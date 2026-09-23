# Post-delivery backlog

These items were agreed as **out of scope for the final delivery**. They are listed here so they do not live only in conversation. Sources: `/mnt/f/EPR_2026_PROCESSED/paper/final_review/0*_*.md` (review ID in brackets). Each item notes what the delivery already does to mitigate it.

## Performance

1. **Frame extraction seeks once per target frame** [05 P2-3]. In `pipeline_service.py`, around lines 1182-1189 and 1607-1617, the code calls `cap.set(CAP_PROP_POS_FRAMES)` and then `cap.read()` for every target. Each target frame therefore decodes from the previous keyframe, which costs about 5x.
   - Fix: when consecutive targets are within about one GOP, advance with `cap.grab()` and `retrieve()` only the targets; seek only for larger jumps.
   - Effort: M.
2. **Mesh and point-cloud previews block the GUI thread**: `mesh.obj` takes 5-96 s and `dense.ply` / `sparse.ply` take 3-7 s cold [05 P1-1, P1-2]. Move parsing to a worker and use `pread`.
3. **Product-tree discovery runs on the GUI thread** and is repeated after every task [05 P2-1]. The viewer also keeps about 66 MB of 3-D state per chunk [05 P2-2].

## Architecture / pipeline

4. **Consolidate the Metashape recipe** [03 P1-5].
   - Make `metashape_worker.process_chunk` the only chunk pipeline, with one `RECIPE1` dict imported everywhere.
   - Today the recipe is duplicated in:
     - `batch_service.DEFAULT_PHOTO_SETTINGS`
     - the `product_catalog` fallback
     - the subprocess `o()` defaults
     - the in-process keyword defaults
   - Delivery mitigation: the in-process route now refuses loudly ("full pipeline requires Windows Metashape via WSL") and prefers the worker whenever `metashape.exe` exists.
5. **Metashape pre-flight probe** [04 P1-8]. Run `metashape.exe -r probe.py` with a 120 s timeout to report the version and activation state, cached per session.
   - Delivery mitigation: the startup-silence, silence and deadline watchdog kills a stuck run with a readable reason.
6. **Real Cancel** threaded through `ProductType.generate` / `default_run_all` [04 P0-1 (M part)].
   - `photogrammetry_service._run_metashape_batch_subprocess` already accepts `cancel_cb`.
   - `anomaly_service.run_detector` already accepts `cancel_cb`.
   - Neither is wired from the UI.
7. **Path authority**: `catalog_builder._Ctx`, `window_context` and others hard-code the bundle layout instead of `workspace_paths.PathResolver` [03 P1-6].

## Science / data integrity

8. **Fabricated edge-held nav** [02 P0-1, 06 P1-2]. The root fix is to stop extrapolating outside each source's time range in `_build_full_interp_csv`, then rebuild `interp_full.csv` and rerun anomaly detection on every dive.
   - Delivery mitigation: the survey report computes position-valid time from non-held rows only.
   - Delivery mitigation: fauna density is NaN outside the 3-8 m altitude band, which excludes the 0.6 m sentinel rows in the held span.
9. **DEM vertical convention** [02 P1-3].
   - Pick elevation = -depth (positive up) consistently across the DEM, `nav_depth.tif` and the CSVs.
   - Report the per-chunk vertical offset against nav.
   - Delivery mitigation: the catalog says "relative relief only", and the merged DEM keeps the chunk unit tag.
10. **Statistics on non-independent frames** [02 P1-7]. Use site-level or block-bootstrap units for `fauna_vs_anomaly` and the report's imagery-by-chemistry section.
    - Delivery mitigation: the figure and catalog are labelled "descriptive".
11. **IDW sensor rasters extrapolate** [02 P1-9]. Add a distance-to-data mask and clip negative partial pressures.
    - Delivery mitigation: the catalog and report captions say "interpolated from along-track measurements; not a synoptic field".
12. **Cross-dive unique site IDs** [02 P1-10]. Use `J1754-S108`, plus an optional `site_group`.
13. **Deduplicated abundance** [02 P1-8, M part]. Add a multiview `track_id` to frame detections.
    - Delivery mitigation: every count is labelled "frame detections, not individuals".
14. **Taxon plausibility filter and morphotype classifier** [02 P0-3]. Extend the midwater/habitat exclusion list, for example *Octopus rubescens*, *Merluccius productus*, *Apostichopus* and *Swima*. Replace zero-shot labels with the morphotype classifier.
    - Delivery mitigation: labels are shown as "model label, unaudited", with the bucket first.
15. **Full provenance** [02 P1-2]. Still missing:
    - sha256 of every nav/sensor input
    - the renav correction ("25 m @ 041 deg") recorded in the sidecars
    - `ultralytics` and Metashape versions in every meta
    - a catalog "Methods and provenance" appendix

    Delivery mitigation: sidecars carry `code_version`, input paths, sizes and mtimes, plus the weights path, size and sha256.
16. **UTC everywhere** [02 P2-1]. Covers the file-name stamps, the `interp_full.csv` `timestamp_iso` column without `Z`, and `meta.json` "generated". The catalog cover and the report footer are now UTC.
17. **Per-product data dictionary** (`README_columns.csv`) [02 P2-6]. The occurrences and density sidecars now carry column notes.
18. **Anomaly terminology and baselines** [02 P2-7]. Add a per-channel baseline/threshold statement and the tier definitions to the catalog anomaly section.

## Reliability

19. **Atomic census writes and superseded folders** [06 P0-1, P1-3]. Also: move-aside before overwriting the canonical products (`merged/`, `survey/anomaly`, `SURVEY_REPORT.html`).
20. **Run-status metas for every product type** [06 P1-6, 04 P2-7]. Photogrammetry writes `run_status.json` now. Discovery still has to honour it (see `handoff_to_backend.md` H1). Rasters and fauna runs still need the same.
21. **Disk-space preflight** before extraction and photogrammetry [04 P2-6].
