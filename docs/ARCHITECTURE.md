# EPR Imaging: architecture overview

This is the maintainer document for the delivered app. It explains how the app turns one dive's navigation, sensor and video files into georeferenced products. It covers the modules, the on-disk contract, the data flow, the threads and subprocesses, and the known weak points.

- Written against commit `f114ce2` (branch `main`). Every file name, function name and constant here was checked against that commit. Line numbers are approximate.
- The typeset PDF version is `EPR_2026_PROCESSED/docs_final/architecture/EPR_Imaging_Architecture_Overview.pdf`.
- The previous architecture document, for the legacy app, is at `archive/legacy_ui/docs/ARCHITECTURE.md`.

**Contents:**

1. [System overview](#1-system-overview)
2. [Module inventory](#2-module-inventory)
3. [The workspace contract](#3-the-workspace-contract)
4. [Data flow](#4-data-flow)
5. [Execution model](#5-execution-model)
6. [External dependencies and failure modes](#6-external-dependencies-and-failure-modes)
7. [Extension points](#7-extension-points)
8. [Known debt](#8-known-debt)

---

## 1. System overview

EPR Imaging is a single-user PySide6 desktop app that works on one dive at a time.

- **Start it with** `launch_simple_ui.sh <dive>.eprproj`, which runs `simple_main.py`.
- **Where it runs:** normally under WSL2 on a Windows workstation.
- **What it drives:** Windows Agisoft Metashape (`metashape.exe`), a licensed MATLAB, and Chrome for PDF printing.

Three rules shape the design:

1. **One master table.** Every product derives from `inputs/interp_full.csv`, a 1 Hz table of navigation and sensor data. It is rebuilt only when its inputs change, and a fingerprint detects the change.
2. **One façade.** The UI talks only to `product_catalog.py`, which has no Qt dependency. For each of the ten product types, `product_catalog` knows three things: how to *find* existing instances on disk (`discover`), how to *make* a new one (`generate`), and which settings it takes (`schema`). It does not implement any science itself; it calls the services.
3. **Scope is a job.** A job is a named set of time intervals. The implicit job `WHOLE_TRACKLINE` (`"__whole__"`) is the whole dive. Products that consume frames are identified by job *and* sampling regime. All other products are identified by job alone.

```
 Navigation CSV          Sensor CSVs                 Video directory
 (renav lat/lon, depth,  (CO2, CH4, O2, T, S;        (start time parsed
  altitude, heading)      units, time_delay_s)        from file name)
        \                    |                        /        :
         \__ File > Import... (auto-saved) _________/         :
                             v                                 :
                      workspace.json                           :
          (config_service; atomic write; workspace.json.lock)  :
                             v                                 :
        ensure_interp() -> pipeline_service build_full_interp  :
     (rebuilt only when the input fingerprint changes;         :
      grid = lat ∩ lon span at 1 Hz; NaN outside each source)  :
                             v                                 :
   inputs/interp_full.csv + interp_full.meta.json (fingerprint):
          every reader clips to the nav span (clip_to_nav)     :
              /                                    \           :
 SAMPLING-INDEPENDENT (identity = job)   SAMPLING-DEPENDENT (identity = job, regime)
   nav_trackline        GeoJSON            sampling_grid  0.25 m, anchored at dive start
   depth_raster         GeoTIFF                 v
   sensor_raster x ch   IDW GeoTIFF        frame recycling  <.... video: uncovered spans only
  [anomaly_detection]   matlab -batch x3        v
   spectrum_trackline   log10 PNG          frame_set  (set_meta.json: in_progress -> complete)
   anomaly_trackline    tier PNG             /                     \
                                      [photogrammetry]         fauna_detection
                                      metashape.exe worker     YOLOv8 zero-shot (torch)
                                      run_status.json          joins anomaly windows
                                      -> merged ortho/DEM
              \                          /
   survey_report -> SURVEY_REPORT.html        [catalog_builder] Run > Build product catalog (PDF)
   (last default-run step; needs              reads products only -> survey/catalog/*.pdf
    merged/ortho_merged.tif)                  via Chrome

 [ ] = starts an external process (MATLAB, Metashape, Chrome)
```

`default_run_all` runs these steps in order:

1. interp
2. nav trackline
3. depth raster
4. one sensor raster per channel
5. frame set
6. photogrammetry, including the merge
7. anomaly detection
8. fauna detection
9. spectrum trackline
10. anomaly trackline
11. survey report

The catalog is a separate action that you run after a run finishes.

---

## 2. Module inventory

The repository root holds 40 Python modules and 4 MATLAB scripts.

- **Reached by the app:** an AST walk of `simple_main.py`'s imports, including lazy imports inside functions, reaches **35 modules** (counting `simple_main` itself).
- **Not imported by the app (5):** `metashape_worker.py`, which `metashape.exe` executes, and four offline CLIs.
- **No cycles:** the import graph is acyclic.
- **Lazy imports:** `product_catalog` imports every service inside functions. Its only module-level import is `batch_service`, inside a `try:`, and it uses it only for `DEFAULT_PHOTO_SETTINGS`.

### UI (Qt, GUI process)

| Module | Responsibility |
|---|---|
| `simple_main.py` | The PySide6 window: Products tree, trackline map and log panel. It also owns the import dialogs, job create/rename/delete, `GenerateDialog`, run confirmation, `ProductViewer`, and the single `Worker` on a `QThread`. It contains no domain logic. |
| `viewer_widget.py` | Optional PyVista/VTK 3-D viewer for PLY and OBJ files. Loaded lazily from `ProductViewer`. |
| `overlay_builders.py` | Builds the trackline and overlay meshes for the 3-D viewer. Only `viewer_widget` imports it. |

### Façade (Qt-free)

| Module | Responsibility |
|---|---|
| `product_catalog.py` | The whole backend of the UI, about 3,800 lines. It holds the `PRODUCT_TYPES` registry, the job store in `workspace.json`, the interp fingerprint and nav-span clipping, the deterministic sampling grid and frame recycling, the altitude gate and chunk planner, fauna orchestration, `_superseded/` backups, the run lock, and `default_run_all`. |

### Core services: configuration, time, paths and interpolation

| Module | Responsibility |
|---|---|
| `config_service.py` | Converts between `workspace.json` and `models` objects. It also provides `atomic_write_text`, and `read_workspace_json`, which raises `WorkspaceFileError` instead of returning `{}` for a corrupt file. It holds the lock primitives (`try_acquire_lock`, `lock_holder`, `release_lock`) and `workspace_json_lock`. |
| `models.py` | Dataclasses: `NavigationConfig`, `SensorFileConfig`, `SensorChannel`, `SelectedTimeRange`, `VideoRecord`, and others. |
| `timeutil.py` | UTC helpers that follow the project's naive-UTC datetime convention. |
| `sensor_service.py` | Loads nav and sensor time series and interpolates them, taking time delays and gaps into account. |
| `video_service.py` | Scans the video directory and parses each file's start time from its name. |
| `pipeline_service.py` | Builds `interp_full.csv` (`build_full_interp`): a positional-span grid, written atomically, with a memory guard. Also runs frame extraction (OpenCV), producing `segment_*/frames/*.jpg` plus `interp.csv`. |
| `dynamicsampling.py` | The along-track distance sampler that `pipeline_service` uses. |
| `workspace_paths.py` → `layout.py` | `PathResolver` handles both workspace layouts: a `.eprproj` bundle or a legacy flat directory. `layout.py` owns the path arithmetic for bundles. |
| `nav_segments.py`, `interval_io.py` | Small helpers. `batch_service` imports `nav_segments`; `anomaly_service` imports `interval_io`. |

### Product modules (Qt-free, called from `product_catalog`)

| Module | Responsibility |
|---|---|
| `output_service.py` | Writes the nav-depth and per-channel sensor GeoTIFFs to `<base>/<sub>/run_NNN/`. It also imports `netcdf_export`, `point_cloud_pipeline`, `qc_report` and `qgis_export`, which are legacy output paths the simple UI does not call. |
| `photogrammetry_service.py` | Checks that Metashape is available and dispatches the run. It prefers the watchdog-supervised `metashape.exe` subprocess. The in-process `import Metashape` path refuses any recipe that needs a DEM, ortho, rotation priors or fixed calibration. It writes `run_status.json`. A COLMAP path exists but the UI does not use it. |
| `merge_products.py`, `ortho_gallery.py` | `merge_products` merges the chunk orthos and DEMs into `survey/photogrammetry/merged/`. Where runs overlap, the newest wins; failed and still-running runs are skipped. It writes `merge_meta.json`. `ortho_gallery` builds the report's close-up gallery. |
| `anomaly_service.py` | Runs the three-stage MATLAB chain under a watchdog. `run_catalog` refuses to fall back to J1754 defaults for any other dive. |
| `build_anomaly_site_catalog.py` | Fuses events into windows, sites and review clips, and writes a QGIS bundle and a PDF report. It is configured through module globals set by `configure()` (see §8). |
| `anomaly_utm.py`, `window_context.py` | `anomaly_utm` writes the UTM GeoJSON layers (pyproj). `window_context` classifies each window as station or transit and writes `window_context.csv`. |
| `fathomnet_detect.py` | YOLOv8 inference per frame directory. It also produces the morphology buckets, exclusions, the points GeoJSON, per-frame density, the fish shortlist, the density-vs-anomaly test and `fauna_provenance.json`. |
| `fauna_timeseries.py`, `fauna_occurrences.py` | `fauna_timeseries` owns the footprint model, the areal-density columns and the time-series figure (`fauna_density.meta.json`). `fauna_occurrences` writes the per-detection occurrence table joined to nav, sensors and windows (`occurrences.meta.json`). |
| `qgis_project.py` | Writes the trackline GeoJSON for `nav_trackline` and assembles the QGIS project. |
| `survey_report.py` | Writes a self-contained HTML survey report. It requires `merged/ortho_merged.tif` and reports position-valid time only. |
| `catalog_builder.py` | Builds the post-run "preview catalog". It walks `survey/`, renders HTML and prints it to PDF with headless Chrome, falling back to matplotlib `PdfPages`. It reads products and computes no science. |
| `reporting_common.py` | New in wave 2. Stdlib-only helpers: `code_version()` (`git describe --dirty`), `file_info`, `provenance()`, `sensor_units`/`channel_label`, and `chunk_status`/`run_status`. |
| `batch_service.py` | A headless full-dive batch runner with its own CLI. The UI imports it only for `DEFAULT_PHOTO_SETTINGS` (recipe1). |

### Worker scripts (run in another interpreter)

| File | Responsibility |
|---|---|
| `metashape_worker.py` | Runs inside Metashape's own Python via `metashape.exe -r`. For each chunk it runs align, dense, mesh, DEM, ortho and report, using nav and rotation priors and fixed intrinsics. It exchanges a JSON params file and a JSON result file with the app. The app never imports it. |
| `GrapherMatrix.m` | The anomaly matrix: 4 configs × 160 strategies per channel. It reads a *relative* `interp_full.csv` from its working directory and calls `navSpeed.m`. |
| `ExportFineConsensusEvents.m`, `ExportDetectorFamilyEvents.m` | Stages 2 and 3. They write the fine-consensus event CSVs and the per-detector-family event CSVs. |

### Offline CLIs (not driven by the UI; `catalog_builder` reads their outputs when present)

| Module | Responsibility |
|---|---|
| `dive_setup.py` | Creates a sensor-only `<dive>_down.eprproj` following the `EPR_2026_DATA` conventions, then builds its interp. |
| `fauna_ortho.py` | Runs the same detector over tiled chunk orthomosaics. |
| `worm_cover.py` | Tile classifier for tube-worm cover. |
| `morphotype.py` | Classifies crustaceans below the bucket level. |

**Tooling:**

- `tools/lint_qt_threading.py` flags lambdas connected to worker signals. That pattern has caused cross-thread crashes that shipped.
- `tools/test_dive_photogrammetry.py`.
- `tests/` holds four files, covering `layout`, `overlay_builders`, `timeutil` and `workspace_paths`.
- `product_catalog` has built-in self-tests (`_selftest_gate`, `_selftest_chunking`, `_selftest_recycling`) that run from its `__main__`.

### What lives in `archive/`

Commit `85eafe8` moved everything the simple UI does not reach into `archive/`. It used `git mv`, so file history is kept. The archived code is kept for reference and the occasional re-run; it is not maintained.

| Directory | Contents |
|---|---|
| `archive/legacy_ui/` | The old `app.py`/`main_window` application: task stack, product graph, workbench, installer, widgets and 9 tests. Also `manifest.py` (the `runs/registry.json` provenance) and the previous `docs/ARCHITECTURE.md`. |
| `archive/analysis_scripts/` | One-off paper figures, cross-dive maps, slide and review decks, multi-view dedup, the frame atlas and the site dossier. |
| `archive/biigle/` | The BIIGLE annotation bridge, export and ingest. |
| `archive/training/` | The YOLO fine-tune (it failed; the zero-shot model is in production), fauna review, and `deploy_predict_offline.py`. |
| `archive/doc_generators/` | The legacy handbook generators and handbook PDF. |
| `archive/legacy_3dvistool/` | An old 3-D example that duplicates the root `point_cloud_pipeline.py`. |

Run an archived script like this (see `archive/README.md`):

```
PYTHONPATH=<repo>:<repo>/archive/analysis_scripts:<repo>/archive/biigle python3 archive/<dir>/<script>.py
```

---

## 3. The workspace contract

A workspace holds one dive in a folder called `<DIVE>_down.eprproj`. `workspace_paths.is_bundle` treats a directory as a bundle if its name ends in `.eprproj` *or* it contains `project.json`.

This is the layout the simple UI uses:

```
<DIVE>_down.eprproj/
├── workspace.json              config + job store (below)
├── workspace.json.lock         transient: held during a job-store / save read-modify-write
├── .epr_run.lock               transient: held for the whole of a generate / default run
├── project.json                optional bundle marker (legacy app)
├── SURVEY_REPORT.html          whole-dive survey report (superseded, never overwritten)
├── inputs/
│   ├── interp_full.csv         1 Hz master table — every product derives from it
│   ├── interp_full.meta.json   input fingerprint + size of the table as written
│   ├── interp_full.stale.csv   left by the UI when an import invalidated the table
│   ├── grapher_matrix_results_<cfg>.mat,
│   │   grapher_matrix_figs/<cfg>/…   MATLAB outputs (MATLAB cwd = here)
│   └── ts_analysis_results.mat optional CTD1 T/S corroboration (disabled when absent)
├── survey/                     WHOLE-TRACKLINE scope (also every legacy/batch product)
│   ├── nav_trackline/          trackline*.geojson + <name>.meta.json
│   ├── nav_depth/run_NNN/      nav_depth.tif + simple_meta.json
│   ├── sensor_2d/<Channel>/run_NNN/   <Channel>_2d.tif + simple_meta.json
│   ├── spectrum_trackline/     *.png + *.meta.json   (job-scoped figures live here too, see §8)
│   ├── anomaly_trackline/      *.png + *.meta.json
│   ├── frame_sets/<whole|job_NNNN>/set_<stamp>/
│   │       set_meta.json, simple_meta.json, segment_*/{frames/*.jpg, interp.csv}
│   ├── photogrammetry/run_<stamp>__<sampling>/
│   │       project.psx, run_status.json, simple_meta.json,
│   │       chunk_NN/{orthomosaic.tif, dem.tif, dense.ply, sparse.ply,
│   │                 mesh.obj, report.pdf, preview_ortho.png}
│   ├── photogrammetry/merged/  ortho_merged.tif, dem_merged.tif, merge_meta.json, previews
│   ├── anomaly/                anomaly_windows_all.csv, anomalous_sites.{csv,geojson},
│   │                           *_utm.geojson, window_context.csv, video_review_clips.csv,
│   │                           *Report.pdf, qgis/, simple_meta.json
│   ├── fauna/                  canonical census (FAUNA_PRODUCT_FILES) + simple_meta.json,
│   │                           fauna_run_summary.json
│   ├── .fauna_staging/         transient: a dive-wide fauna run builds here, then is promoted
│   ├── catalog/                catalog_<DIVE>_<stamp>.{html,pdf}, assets/
│   └── jobs/<job_NNNN>/        JOB scope: same sub-folders, plus
│           interp_job.csv, anomaly_input/interp_full.csv,
│           fauna_runs/<DIVE>_fauna_<stamp>__<sampling>/  (a mini-workspace),
│           SURVEY_REPORT_<stamp>.html
└── _superseded/<UTC-stamp>/    previous canonical files, workspace-relative layout + superseded.json
```

Fauna runs are written in one of two places:

- Only a whole-dive run at the default regime writes the canonical `survey/fauna/` (`_is_census_run`).
- A whole-dive run at any other sampling regime writes to `survey/fauna_runs/…`.

`~/.epr_simple_ui.json` records the last workspace that loaded successfully.

### workspace.json essentials

`ConfigService` reads and writes the configuration keys, and the simple UI adds two keys for its job store. Legacy keys (`job_history`, `pending_job`, `task_stack`, `out_*`, `clahe_*`, `photo_*`, …) are preserved on save but otherwise **ignored**; the UI's own defaults (`product_catalog.DEFAULTS`) take precedence.

| Key | Meaning |
|---|---|
| `navigation_file` | Holds `latitude_source`, `longitude_source`, `altitude_source`, `depth_source`, `heading_source`, and so on. Each source has a `csv_path`, column indices or names, and a parsed `start_time`/`end_time`. The positional span used for clipping is lat ∩ lon. |
| `sensor_files` | A list of `{csv_path, timestamp_column, channels:[{source_column, display_name, units}], time_delay_s, …}`. |
| `depth_source`, `speed_source` | Top-level source selections. Both are part of the interp fingerprint. |
| `video_directory`, `filename_datetime_format`, `frame_quality` | The video input for frame extraction. |
| `simple_jobs` | `[{job_id: "job_NNNN", name, intervals: [[t0, t1], …]}]`, in unix seconds (UTC). |
| `simple_jobs_high_water` | The highest job number ever issued. IDs are never reused, so a new job can never adopt a deleted job's product folders. |

- **Atomic writes.** Every write goes to a unique temp file in the same directory, is `fsync`ed, then moved into place with `os.replace`.
- **Locking.** Job-store read-modify-writes hold `workspace.json.lock`.
- **Corrupt files.** A file that exists but will not parse raises `WorkspaceFileError`. No writer ever replaces it with a jobs-only file.
- **Import auto-save.** The UI saves each import as soon as it succeeds (`_after_import`), because runs read `workspace.json` from disk, never from the UI's memory.

### Meta and sidecar files

| File | Written by | Contents and role |
|---|---|---|
| `simple_meta.json` | `product_catalog._write_meta` | Keys: `type_key`, `job_id`, `job_name`, `intervals`, `characteristic`, `created_at`, `primary`, `views`, plus extras per type. Photogrammetry adds `sampling`, `photo_settings`, `chunk_qc` and `run_status`. Fauna adds `weights`, `bucket_thresholds`, `by_bucket_kept` and `provenance`. **Discovery uses this file to attribute a product to its job.** A product without one is treated as legacy and belongs to the whole trackline. |
| `<file>.meta.json` | nav_trackline, spectrum/anomaly trackline | The same job attribution, for single-file products. |
| `interp_full.meta.json` | `_write_interp_meta` | `fingerprint` (sha256), `schema` (`INTERP_SCHEMA`), `config`, `files` (the size and `mtime_ns` of each input CSV), `interp_size`, `interp_mtime_ns`, `written_at`, `adopted_legacy`. |
| `set_meta.json` | frame_set | `status` moves from `in_progress` to `complete`, or to `failed` with an `error`. Also records `settings` (the sampling triple and reuse options), `intervals`, `n_requested`/`n_frames`/`n_reused`/`n_extracted`, `reuse_methods` and `segments`. Only `complete` sets are discovered or recycled. A set with no `status` predates the status markers and counts as complete. |
| `run_status.json` | `photogrammetry_service._write_run_status` | Written as `running` before Metashape starts, then updated to `ok`, `partial` or `failed`. Also records `started_utc`, `finished_utc`, `n_chunks`, `worker_tag`, `n_ok`, `n_failed`, `metashape_version`, `error`, `code_version`, and a `chunks[]` list. Each chunk entry has `cameras_aligned`/`cameras_total`, `status`, `reason`, `n_products` and `removed_empty`. Discovery hides `failed` runs, and `running` runs that have no meta. Merge skips both. |
| `fauna_provenance.json`, `fauna_density.meta.json`, `occurrences.meta.json` | fathomnet_detect, fauna_timeseries, fauna_occurrences | The `reporting_common.provenance()` block: `generated_utc`, `code_version`, and the path, size and mtime of each input, with a sha256 for the weights. Also column notes and the label caveat. `occurrences.csv` starts with a `#` comment line, so read it with `comment="#"`. |
| `merge_meta.json` | merge_products | Each input raster with its run, size and mtime, plus the overlap policy and provenance. |
| `run.meta.json`, `<tif>.meta.json` | output_service | Raster generation parameters. |

### `_superseded/`: never overwrite a canonical file in place

Some files keep fixed names because other tools read them: the fauna census, `survey/anomaly/*`, `merged/*`, `SURVEY_REPORT.html`, and a stale `interp_full.csv`. Before any of these is replaced:

1. `_supersede` moves the old file to `<ws>/_superseded/<UTC-stamp>/<workspace-relative path>`.
2. It records the move in `superseded.json` (`reason`, `superseded_at_utc`, `moved[{from,to}]`).

If the replacement fails, `_restore_superseded` moves the partial new output into `<stamp>/_failed_partial/`, puts the originals back, and stamps `rolled_back_utc`.

Two products follow a stricter order:

- **Anomaly catalog:** the old files are moved only *after* the MATLAB detector succeeds.
- **Fauna census:** the dive-wide census is built in `survey/.fauna_staging/` and promoted (`_promote_census`) only after the whole build succeeds.

### Locks

- **`.epr_run.lock`** (at the workspace root). `ProductType.generate` and `default_run_all` take it through `run_lock`.
  - It is re-entrant within one process, because `default_run_all` calls `generate`.
  - If another process holds it, it raises a `RuntimeError` naming that process's pid, host, task and start time.
  - A lock whose pid is dead on the same host is stale, and is taken over silently.
- **`workspace.json.lock`**. `workspace_json_lock` holds it around each read-modify-write. It waits up to 15 s, then raises `TimeoutError`. Threads within one process queue on an `RLock`.
- Both locks are files created with `O_CREAT|O_EXCL` that contain a JSON record `{pid, host, task, started_at, max_age_s}`.

---

## 4. Data flow

### 4.1 Import

The three import commands (*File ▸ Import navigation and orientation…*, *Import video…* and *Import new sensor channel…*) build `models` objects and save them to `workspace.json` straight away.

If the navigation or sensor inputs changed, the UI also invalidates the old interp table (`_invalidate_interp`). It deletes the meta sidecar and renames `interp_full.csv` to `interp_full.stale.csv`, so the next run rebuilds it.

### 4.2 Fingerprinted interp build with span clipping

`ensure_interp(ws)` is the first step of every product. It works in four steps:

1. **Fingerprint.** It computes `interp_fingerprint`, a sha256 over three things:
   - `INTERP_SCHEMA`;
   - the saved `navigation_file`, `sensor_files`, `depth_source` and `speed_source` configuration, with the derived `start_time`/`end_time` stripped;
   - the size and `mtime_ns` of every CSV the configuration references.
2. **Compare.** `interp_staleness` compares that fingerprint with `interp_full.meta.json`, and checks the table's size on disk. If both match, the table is reused.
   - A legacy table with no meta is *adopted* as-is, unless an input file is newer than the table.
   - A stale table is superseded and then rebuilt. If the rebuild fails, the old table is restored.
3. **Rebuild.** The rebuild calls `PipelineService.run` with `selected_steps=["build_full_interp"]`. In `pipeline_service._build_full_interp_csv`:
   - The grid covers the **positional span** (latitude ∩ longitude). It falls back to the union of all sources only when no navigation is configured.
   - Each channel is NaN outside its own source's coverage and inside gaps (`NAV_MAX_GAP_S` = 30 s, `SENSOR_MAX_GAP_S` = 60 s).
   - A grid longer than 7 days or 5,000,000 rows is refused.
   - The table is written to a `.partial` file, then moved into place with `os.replace`.
4. **Clip on read.** Tables built before this fix, by the old union-of-sources grid, are not rewritten. Instead, every reader (`track_polyline`, `_interp_df`, `job_interp_csv`) passes them through `clip_to_nav`. This clips rows to the stored lat ∩ lon span (±1 s), then drops leading or trailing runs of at least `HELD_EDGE_MIN_ROWS` (60) identical positions.

### 4.3 Sampling grid and frame recycling

`sampling_grid(ws, spacing_m, min_frequency_hz)` places sample *k* where the cumulative along-track distance from the dive's first fix reaches *k* × spacing (default 0.25 m). A clock floor, 0.1 Hz by default, adds samples during long stationary periods.

The grid depends only on the interp table and the spacing, never on the requested interval. So every job's samples are a strict subset of the whole-dive grid, and that is what makes frame reuse exact.

`_generate_frame_set` then works through these steps:

1. It keeps the grid times that fall inside the job's intervals and inside the nav span.
2. `scan_frame_pool` gathers every frame already on disk with the same sampling mode and spacing. The frames come from complete frame sets and, when `reuse_legacy` is set, from the batch runner's `survey/photogrammetry/*/segment_*`.
3. `coverage_runs` splits the desired times into covered and uncovered runs (tolerance 0.5 s). `coalesce_runs` then merges runs shorter than 15 samples into a neighbouring run, preferring a covered neighbour.
4. Uncovered runs are decoded from video. This uses `_grid_pipeline`, a `PipelineService` subclass that overrides `_get_dynamic_sample_times` with the explicit plan.
5. Covered runs are hard-linked into `segment_rNN_*`, falling back to a symlink and then a copy. Their `interp.csv` is rebuilt from the source manifests.
6. `set_meta.json` is written as `in_progress` before any frames and as `complete` at the end. An empty set is deleted, not left on disk.

### 4.4 Product chains

| Type | Chain |
|---|---|
| `nav_trackline` | For the whole dive: `qgis_project.build_trackline_geojson`. For a job: a MultiLineString built from the clipped interp rows, stamped EPSG:32613. |
| `depth_raster`, `sensor_raster` | `job_interp_csv` feeds `OutputService.generate_nav_2d_geotiff` or `generate_sensor_2d_geotiff`. Defaults: 5 m cells, UTM, IDW. The default run makes one raster per channel listed by `sensor_channels(ws)`. |
| `photogrammetry` | See the steps below this table. |
| `anomaly_detection` | See the steps below this table. |
| `fauna_detection` | See the steps below this table. |
| `spectrum_trackline`, `anomaly_trackline` | Matplotlib (Agg) PNG figures drawn from the clipped interp and the anomaly windows, each with a `.meta.json` sidecar. |
| `survey_report` | Refuses to start unless `merged/ortho_merged.tif` exists. For the whole dive, the old `SURVEY_REPORT.html` is superseded. For a job, it writes a new `SURVEY_REPORT_<stamp>.html` under the job, but the content still covers the whole dive. |
| catalog (not a product type) | Run it from *Run ▸ Build product catalog (PDF)* or from a job's right-click menu. `catalog_builder.build_catalog(ws, job_id, log)` writes `survey/catalog/catalog_<DIVE>_<stamp>.{html,pdf}`. It is never part of `default_run_all`. |

**`photogrammetry`** runs these steps:

1. `resolve_frame_set` picks the newest complete frame set with the same sampling regime, or builds one.
2. `plan_photogrammetry_chunks` applies the altitude gate. It drops frames with altitude over 8 m or NaN, and discards any remaining span shorter than 15 frames.
3. `chunk_sizes` splits what is left into chunks of at most 350 frames. A longer span is cut into ⌈n/350⌉ near-equal parts.
4. `photogrammetry_service.run_metashape_batch` builds one project with many chunks in `run_<stamp>__<sampling>/`.
5. For the whole dive only: the previous `merged/` is superseded, then `merge_survey` and `build_previews` run. If they fail, the old files are rolled back.
6. `simple_meta.json` records `chunk_qc` from `run_status.json`.

**`anomaly_detection`** runs these steps:

1. Prepare the input table:
   - for the whole dive, `ensure_interp`;
   - for a job, `interp_job.csv` is copied to `anomaly_input/interp_full.csv`, because MATLAB opens that exact relative name.
2. `anomaly_service.run_detector` runs three `matlab -batch` stages in order: `GrapherMatrix`, `ExportFineConsensusEvents`, `ExportDetectorFamilyEvents`. MATLAB's working directory is the folder that holds the interp table.
3. The previous catalog is superseded.
4. `run_catalog` configures and runs `build_anomaly_site_catalog` with explicit `interp_csv`, `raw_nav_csv`, `event_root`, `out_dir`, `ts_results` and `dive`. If it fails, the old catalog is rolled back.
5. For the whole dive only, `anomaly_utm.build_utm_layers` and `window_context.classify_windows` then run.

**`fauna_detection`** runs these steps:

1. Find the weights: `$EPR_FAUNA_WEIGHTS`, otherwise the author's path. If the file is missing, the run fails with a clear error.
2. `resolve_frame_set`. Frames always come from a recycled frame set.
3. `fathomnet_detect.detect_frames` runs once per segment on the device chosen by `fauna_device`. On CPU the batch size is capped at 2.
4. A mini-workspace root is set up: `interp_full.csv` is hard-linked into it, and `window_context.csv` is copied in (or an empty placeholder is written).
5. Inside `_nav_from(nav)` the products are built in order:
   1. `to_geojson` (required);
   2. the fish shortlist;
   3. density vs anomaly, only when windows exist;
   4. `fauna_timeseries.add_density_columns` and `build`;
   5. `fauna_occurrences.build`.

   Each optional step runs through `_fauna_step`, so a failure loses that one output, not the run.
6. A dive-wide run at the default regime is promoted into `survey/fauna`.

Per-bucket confidence thresholds from `deploy_config.json` are off by default, so a flat 0.25 floor applies. When they are switched on, detections below their bucket's threshold are kept but marked `excluded="below_conf"`.

### 4.5 Where units and provenance attach

- **Units.** Units are entered at sensor import and stored as `channels[].units` in `workspace.json`. `reporting_common.sensor_units(ws)` reads them back. `channel_label` renders them, for example "pCH4 (µatm)", in the catalog, the survey report and `fauna_occurrences`.
- **Interp provenance.** Recorded in `interp_full.meta.json` (see §3).
- **Product provenance** comes in two layers:
  - `simple_meta.json` records scope, settings and QC for every generate.
  - The `reporting_common.provenance()` block records `code_version` (a `git describe --dirty` of the checkout), the UTC generation time, and input file info. The fauna modules and `merge_products` write it. `code_version` is also stamped into `run_status.json`, the anomaly report, the survey report and the catalog.
- **Time.** All times are naive UTC, and all sources share one timezone. There is no video UTC offset.
- **CRS.** Projected products use EPSG:32613 (WGS 84 / UTM 13N).

---

## 5. Execution model

### GUI thread (Qt event loop)

The GUI thread handles:

- loading and saving the workspace, and imports (with auto-save and interp invalidation);
- job create, rename and delete (short, locked, atomic writes);
- the `discover()` filesystem scans for the Products tree;
- loading the trackline. `_load_track` calls `track_polyline` only when the interp table exists and is not stale, so a rebuild never runs on this thread. A stale table is read directly and flagged as stale.
- parts of `ProductViewer` decoding;
- detached `xdg-open` and `explorer.exe` calls.

### One worker QThread at a time

`MainWindow._start` creates a `QThread` and a `Worker(QObject)` that wraps one callable: `ProductType.generate`, `default_run_all` or `catalog_builder.build_catalog`.

- **Signals.** The worker's `line`, `done` and `failed(cause, traceback)` signals connect to *bound methods* of the main window, so Qt queues them back to the GUI thread. A lambda connected here would run on the worker thread instead; `tools/lint_qt_threading.py` checks for this.
- **Exceptions.** `Worker.run` catches `BaseException`, so a stray `SystemExit` cannot abort the process.
- **What runs here.** All in-process heavy work runs on this one thread: pandas, scipy, OpenCV, rasterio/GDAL, matplotlib (Agg) and torch/ultralytics.
- **While busy.** `_set_busy` disables *Run all*, *Build catalog* and the trackline's *Create job* button. Any Generate or Run request is refused with "a task is already running".
- **Confirmation.** `_confirm_run` shows a dialog, defaulting to No, before every *Run all* and before a whole-trackline Generate that would replace canonical files. The dialog lists the steps, the expected duration, and the files that `replacement_preview(ws, job)` says will be replaced.
- **Closing mid-task.** The window asks first, then detaches the thread into `_ORPHANS`. `main()` waits for the thread before exiting, but the task's remaining log lines are lost.

### The log-prefix contract

Every `log_fn` receives plain strings. `product_catalog` defines three line prefixes. The UI colours and collapses lines by prefix, so the prefixes are part of the interface.

| Constant | Prefix | Meaning |
|---|---|---|
| `LOG_FAIL` | `"!! "` | A failure: one compact line (`one_line(exc)`), shown in red. |
| `LOG_WARN` | `"note: "` | A warning that did not stop the step. |
| `LOG_DETAIL` | `"  · "` | Supporting detail, such as traceback lines. Collapsible. |

`default_run_all` isolates each step. A failed step logs one `!! <step> FAILED: …` line followed by `LOG_DETAIL` traceback lines, and the run carries on with the next step. The last line is always a verdict: either `DEFAULT RUN COMPLETE — n/n steps ok` or `!! DEFAULT RUN FINISHED WITH k OF n STEP(S) FAILED`.

### Subprocesses

**MATLAB: `matlab -batch "addpath(repo); cd(workdir); repo=…; <stage>"`, three stages** (`anomaly_service._matlab_run`).

- Runs in its own session (`start_new_session=True`) with `stdin=DEVNULL`.
- A watchdog daemon thread kills it after 420 s of silence at launch (`_STARTUP_GRACE_S`; this usually means a licensing hang), or at the 24 h deadline (`timeout_s`).
- A regex (`_SIGNIN_RE`) detects the sign-in prompt.
- Kills use `os.killpg(SIGKILL)`.
- At most 200 lines per stage reach the GUI log.

**Metashape: `metashape.exe -r metashape_worker_<tag>.py params_<tag>.json`** (`photogrammetry_service._run_metashape_batch_subprocess`).

- The exe is found under `/mnt/c`.
- The worker copy, params, result and log files go in `/mnt/c/Users/Public/epr_metashape/`. Each is tagged with an 8-character uuid, so concurrent runs cannot collide.
- Paths are translated with `wslpath`.
- It runs in a new session with `stdin=DEVNULL`.
- The watchdog kills the run when any of these happens:
  - nothing is printed within 300 s (`METASHAPE_STARTUP_GRACE_S`);
  - output goes silent for 45 min (`METASHAPE_SILENCE_S`);
  - the run passes its deadline of 2 h plus 1 h per chunk.
- To kill, it runs `taskkill /T /F` on the PID the worker reported, then `killpg` on the local process handle.
- `run_status.json` is written before launch and on every exit path.
- The call raises if every chunk failed. Zero-byte products are deleted.

**Chrome: headless `--print-to-pdf`** (`catalog_builder.chrome_pdf`).

- 600 s timeout.
- Uses its own `--user-data-dir` (`.chrome_profile`, next to the HTML).
- If Chrome fails, it falls back to a degraded matplotlib `PdfPages` layout.

**Helpers: `wslpath` and `git describe`.** Short calls. `code_version` has a 10 s timeout.

Cancel exists in the services but is not reachable from the UI. `anomaly_service.run_detector` and `_run_metashape_batch_subprocess` accept a `cancel_cb`, but nothing wires it. `product_catalog` exposes no cancel entry point, so the Stop button stays disabled, and its tooltip says why.

---

## 6. External dependencies and failure modes

| Dependency | Used by | If absent or broken |
|---|---|---|
| Agisoft Metashape Pro 1.x/2.x: the Windows `metashape.exe` driven from WSL, or `import Metashape` | photogrammetry | The step fails with `metashape_unavailable_reason()`. A licence or activation dialog is killed by the startup watchdog, with a readable reason. A host with only the Python module refuses the full recipe with a clear error. Without a merged ortho, `survey_report` then fails its precondition. |
| MATLAB (`matlab` on PATH, licensed) plus the four `.m` files in the repo root | anomaly_detection | The step fails with `MATLAB unavailable: …`. A signed-out online licence is detected, or the process is killed after 420 s. The catalog alone can still be built from existing event CSVs through `anomaly_service.run_catalog`. |
| ultralytics + torch (CUDA optional); the weights `mbari_315k_yolov8.pt` via `EPR_FAUNA_WEIGHTS`; `deploy_config.json` via `EPR_FAUNA_DEPLOY_CONFIG` | fauna_detection | Missing weights: the step fails and names the variable to set. No CUDA: inference runs on CPU with batch ≤ 2, which is slow. Missing deploy config (only read when per-bucket thresholds are switched on): a flat 0.25 floor applies, with a log note. |
| Google Chrome / Chromium | catalog PDF | The catalog falls back to a degraded matplotlib PDF, labelled as such. |
| Video directory that is reachable and that OpenCV can decode | frame_set | Uncovered spans are skipped with a note. If no frames were reused either, the set is deleted and the step fails ("frame set is empty"). |
| Anomaly windows (`survey/anomaly/window_context.csv`) | fauna_detection | Fauna degrades: the density-vs-anomaly figure is skipped, and the occurrence table's window columns stay blank. |
| Merged orthomosaic | survey_report | The report refuses to start and points to photogrammetry. |
| PySide6, matplotlib, numpy, pandas, scipy, OpenCV, rasterio/GDAL, utm, pyproj, Pillow | core | Required. `requirements.txt` matches the import graph; `requirements-lock.txt` pins the tested set (Python 3.13). |
| pyvista / pyvistaqt / vtk | 3-D viewer | The 3-D button is disabled. The matplotlib preview still works. |
| `git` on PATH, running in a git checkout | provenance | `code_version()` returns `"unknown (no git checkout)"`. A tree with uncommitted changes is reported as `<hash>-dirty`. |
| Hard-coded data roots `/mnt/f/EPR_2026_PROCESSED` and `/home/troyboland/biigle/storage/images` | `fathomnet_detect`/`window_context` CLI defaults, catalog frame crops | Only the CLI entry points and the catalog's optional frame crops use these paths. The UI passes workspace paths explicitly. |

---

## 7. Extension points

### The ProductType contract

```python
@dataclass
class ProductType:                       # product_catalog.py
    key: str                             # stable id, also the tree / meta key
    label: str                           # tree label
    settings_schema: list[tuple]         # (key, label, kind, default, extra)
    _discover: Callable[[ws, Job], list[ProductInstance]]
    _generate: Callable[[ws, Job, settings: dict, log], ProductInstance]
    _schema_fn: Callable[[ws], list[tuple]] | None   # workspace-dependent choices
    last_error: str | None               # set when discover() swallowed an exception

    def schema(ws=None)   -> list[tuple] # _schema_fn(ws) if given, else settings_schema
    def defaults(ws=None) -> dict
    def discover(ws, job) -> list[ProductInstance]    # newest first; never raises
    def generate(ws, job, settings=None, log_fn=None) # merges defaults, logs a header,
                                                      # holds run_lock, calls _generate
```

- **Settings kinds.** `kind` is one of `float`, `int`, `str`, `choice` or `bool`. For `choice`, `extra` holds the list of options. `GenerateDialog` builds its form from `schema(ws)`.
- **ProductInstance** has the fields `type_key`, `label`, `path`, `created_at`, `job_id` and `view_paths`. Build one with `_instance()`, which keeps only the view paths that exist.

### Adding a product type: checklist

1. **Generate.** Write `_generate_x(ws, job, settings, log)`:
   - Get input through `ensure_interp`, `job_interp_csv` or `_interp_df`, which are already clipped. Never read `interp_full.csv` directly.
   - Write under `_scope_root(ws, job)`, with `_unique_dir`/`_unique_file` and `_stamp()`, so two runs never share a folder.
   - Call `_write_meta(...)` or write a `.meta.json` sidecar, then return `_instance(...)`.
   - Raise on failure. Do not log the error and return.
   - Use the `LOG_WARN`/`LOG_DETAIL` prefixes.
2. **Discover.** Write `_discover_x(ws, job)`:
   - Filter by owner with `_meta_job(dir)`, or the sidecar's `job_id`.
   - Skip partial outputs, the way frame sets and photogrammetry do with their status files.
   - It runs on the GUI thread, so keep it to cheap directory scans.
3. **Register.** Add a `ProductType(...)` to `PRODUCT_TYPES`. The list order is the order shown in the tree.
4. **Default suite** (optional):
   - Add a `step(...)` in `_default_run_all_locked`.
   - Add the key to `MainWindow.RUN_ALL_ORDER` and `DURATION` in `simple_main.py`; the confirmation dialog uses both.
5. **Canonical files.** If the type writes fixed-name files that other tools read:
   - Call `_supersede` before writing, and `_restore_superseded` on failure.
   - List the files in `replacement_preview`.
6. **Provenance.** Put a `reporting_common.provenance(inputs, hashed=…)` block in the sidecar. Use `channel_label` for any axis that shows a sensor value.
7. **Hygiene:**
   - Import heavy modules lazily, inside the function.
   - Call `matplotlib.use("Agg")` before pyplot. `simple_main` selects QtAgg, so pyplot on the worker thread would otherwise create Qt objects.
   - Never import Qt.
   - Run `python3 tools/lint_qt_threading.py`.

### The sampling identity rule

The first question for a new type is whether it consumes frames.

**If it consumes frames, it is sampling-dependent.** Its identity is (job, regime), and one job can hold several instances. To implement one:

1. Add its key to `SAMPLING_DEPENDENT`, which is `{"frame_set", "photogrammetry", "fauna_detection"}` today.
2. Get frames through `resolve_frame_set(ws, job, settings, log)`, which reuses only a set with the same regime.
3. Take the identity from the set actually used (`frame_set_sampling(set_dir)`).
4. Put `sampling_slug()` in the directory name, `sampling_label()` at the *start* of the characteristic, and `sampling` in the meta.

**If it does not consume frames, it is sampling-independent.** Its identity is the job alone. Compute it once per job from the 1 Hz table, and never mention sampling in its path, meta or label.

The sampling regime is the triple `SAMPLING_KEYS = ("sampling_mode", "spacing_m", "min_frequency_hz")`.

- Only `"dynamic"` is offered today.
- `sampling_label` can already render `"fixed 1 s"`, so a fixed-interval sampler can be added without changing the identity machinery.
- `simple_main._sampling_dependent` asks `pc.is_sampling_dependent`, so the UI follows the backend.

### Other seams

- **A new sensor channel** needs no code. Import it with a `display_name` and `units`. The rasters, the spectrum trackline and the default run pick it up through `sensor_channels(ws)`.
- **A new Metashape setting** goes into three places:
  - recipe1 (`batch_service.DEFAULT_PHOTO_SETTINGS`);
  - `PHOTO_SETTING_KEYS` and `_photogrammetry_schema`, if users should see it;
  - `metashape_worker.py`.

  §8 lists the other copies of the recipe that must be kept in step.
- **A new external tool** should copy the `_matlab_run`/`_run_metashape_batch_subprocess` pattern:
  - `stdin=DEVNULL` and a new session;
  - a watchdog thread with startup-silence, silence and deadline limits;
  - a status file written before launch and on every exit;
  - a one-line failure reason.

---

## 8. Known debt

These items were deliberately deferred at delivery. The first four are the P1 findings of the architecture review (`paper/final_review/03_architecture.md`). The full backlog, with effort estimates, is in `docs/POST_DELIVERY.md`.

### 8.1 Two Metashape pipelines and four copies of the recipe (03 P1-5 · POST_DELIVERY #4)

`metashape_worker.process_chunk` is the only full recipe: DEM, ortho, rotation priors and fixed calibration. `photogrammetry_service._run_metashape_batch_inproc` is a second, reduced pipeline.

The recipe defaults exist in four places:

- `batch_service.DEFAULT_PHOTO_SETTINGS`, which is what the app sends;
- the fallback dict in `product_catalog`;
- the subprocess `o()` defaults, which disagree with recipe1 (for example `build_mesh=False`, `mesh_surface="Arbitrary"` and `build_dem=False`);
- the in-process keyword defaults.

*Mitigation today:*

- The exe path is used whenever `metashape.exe` exists.
- The in-process path refuses any run that asks for `_INPROC_UNSUPPORTED` options.
- The app always sends the full recipe.

*Fix:* make the in-process path call `process_chunk`, and keep one `RECIPE1` dict that every other place imports.

### 8.2 Leftover module-global rebinding (03 P1-7)

- `build_anomaly_site_catalog.configure()` still rebinds six globals (`REPO`, `EVENT_ROOT`, `INTERP`, `RAW_NAV`, `TS_RESULTS`, `OUT`) whose defaults are J1754 files.
  - *Mitigated:* `anomaly_service.run_catalog` now requires explicit run-scoped paths. For any dive other than J1754, it refuses a missing raw nav file and any resolved path that is a J1754 or repo default.
  - The globals are still shared across the whole process.
- `product_catalog._nav_from` monkey-patches `load_frame_nav` in both `fathomnet_detect` and `fauna_occurrences` while a fauna build runs, and restores it on exit. That is safe only while each process has one worker.
- `ProductType.last_error` is mutable state on shared, module-level registry instances.

*Fix:* pass a `CatalogPaths` dataclass into `run()`, and add a `nav=` parameter to the fauna functions.

### 8.3 No process isolation and no cancel (03 P1-3 · POST_DELIVERY #6)

All in-process work runs on the GUI process's single worker `QThread`. A segfault or out-of-memory in torch or GDAL takes down the window, and with it the only copy of the log.

- The log exists only in the `QPlainTextEdit` (`setMaximumBlockCount(20000)`). There is no file log, and no `file_log_fn` is passed to the services.
- There is no cancel: the services' `cancel_cb` hooks exist but are not wired, and the Stop button is disabled.
- Closing the window mid-run detaches the task.
- The watchdogs cover only the external processes.

*Fix:* run each generate in a spawned child process that streams log lines over a pipe, and kill its process group on Stop. Also copy the log to `<ws>/logs/`.

### 8.4 Split path authority (03 P1-6 · POST_DELIVERY #7)

`product_catalog` resolves paths through `workspace_paths.PathResolver`, which understands both the bundle layout and the legacy flat layout (`ws/outputs`, `ws/anomaly_site_catalog`). Several readers hard-code the bundle layout instead (`ws/survey/…`, `ws/inputs/interp_full.csv`):

- `catalog_builder._Ctx`
- `window_context`
- `fathomnet_detect`, `fauna_occurrences` and `fauna_timeseries`
- `merge_products`
- `survey_report`

The UI's *Open workspace* accepts a non-`.eprproj` folder if it contains a `workspace.json`, and `main()` accepts any directory given on the command line. On a legacy flat workspace, the writers and these readers look in different places.

The simple UI also ignores two parts of `layout.py`:

- its job layout (`jobs/job_NNN/products/`);
- its provenance registry (`manifest.py`, now archived).

*Fix:* first, refuse or convert non-bundle roots in `simple_main`. Then either route every reader through `PathResolver`, or declare the §3 layout the authority and delete the legacy branches.

### 8.5 Smaller items a maintainer will meet

- **Confirmation text is stale.** `_confirm_run` says replaced files are overwritten "in place (no backup is kept)", but the backend now moves them to `_superseded/`.
- **Generate falls back to the UI's file list.** For a single-type Generate, `_replacement_preview` calls `replacement_preview(…, type_keys=…)`. The backend does not accept that keyword, so the UI falls back to its own list (`_canonical_targets`).
- **Job scope is partial.**
  - Job-scoped figures (`spectrum_trackline`, `anomaly_trackline`) are written under `survey/<folder>/` and attributed to the job only through their sidecar's `job_id`.
  - A job-scoped survey report contains dive-wide content.
  - UTM layers and `window_context` cover the whole dive only.
- **Hard-coded site and user values:**
  - EPSG:32613 appears in the job trackline, `merge_products`, `qgis_project`, `anomaly_utm`, `fathomnet_detect` and the `metashape_worker` fallback.
  - `fathomnet_detect.ROOT`/`FRAMES_ROOT` and `window_context.ROOT` are absolute paths.
  - The fallback weight paths are under `/home/troyboland/models`.
- **Duplicated conventions:**
  - Tier colours disagree: SCREEN is `#e0a800` in `product_catalog` but `#3d6ea8` in `anomaly_service.TIER_COLORS`.
  - The dive name is derived with `split("_")[0]` in some places and with `J\d{4}` in others.
- **Private override.** `_grid_pipeline` overrides the private method `PipelineService._get_dynamic_sample_times`.
- **Size and tests.** `product_catalog.py` is about 3,800 lines. Neither it nor `simple_main` has pytest coverage; only the `_selftest_*` functions run, from `__main__`.
- **I/O on the GUI thread.** Discovery scans, and mesh and point-cloud preview parsing in the viewer (POST_DELIVERY #2–3).

### 8.6 POST_DELIVERY.md in brief

| Area | Items (numbers as in `docs/POST_DELIVERY.md`) |
|---|---|
| Performance | #1 frame extraction seeks once per target (about 5× the decode cost). #2 mesh and PLY previews block the GUI. #3 discovery runs on the GUI thread, and the viewer keeps about 66 MB of 3-D state per chunk. |
| Architecture | #4 consolidate the recipe. #5 Metashape pre-flight probe. #6 real Cancel. #7 path authority. |
| Science and integrity | #8 edge-held nav: the build now uses the correct grid, but existing tables and anomaly outputs are not rebuilt automatically. #9 DEM vertical convention. #10 statistics on non-independent frames. #11 IDW extrapolation mask. #12 cross-dive site IDs. #13 deduplicated abundance. #14 taxon plausibility and morphotype labels. #15 full input provenance (sha256, renav correction, tool versions). #16 UTC everywhere. #17 per-product data dictionary. #18 anomaly baselines and tier definitions in the catalog. |
| Reliability | #19 atomic census and superseded folders: now implemented for the census, anomaly, merged and report files. #20 run-status metas for every type: photogrammetry is done, including discovery; rasters and fauna runs are pending. #21 disk-space preflight. |
