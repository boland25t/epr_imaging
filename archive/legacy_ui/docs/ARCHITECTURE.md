# EPR Imaging — Software Architecture

**Status:** current as of 2026-07-24
**Scope:** 28,438 lines of application Python (33 modules), 609 lines in the
legacy standalone 3-D example, 1,817 lines of active MATLAB, and 382 lines of
tests. Generated survey data, experimental MATLAB scripts, and archived reports
were removed in the 2026-07-24 repository cleanup.

This document describes how the system is put together, the invariants that hold
it together, and where the remaining risk sits. Findings were verified by
execution, not inspection alone.

---

## 1. What the system does

A PySide6 desktop application that turns **raw survey inputs** into **science
products**:

```
INPUTS                          PRODUCTS
video files            ─┐       interpolated master table (interp_full.csv)
navigation CSVs        ─┼──►    extracted + annotated frames
sensor CSVs            ─┘       point clouds (PLY), rasters (GeoTIFF), NetCDF
                                QGIS projects, QC reports
                                photogrammetry models (Metashape / COLMAP)
                                anomaly catalogs (PDF, CSV, GeoJSON, QGIS)
```

The organising abstraction is the **Job** — a named set of time intervals over
the survey. Nearly every product can be generated either for the *full dataset*
or per *job*.

---

## 2. Layering

The dependency graph is acyclic and enforced by two conventions. This is the
codebase's single most valuable property: it is why the system is testable at
all.

```
┌──────────────────────────────────────────────────────────────┐
│ app.py                                                        │
├──────────────────────────────────────────────────────────────┤
│ GUI (15 Qt modules)                                           │
│   main_window.py ── the hub: 8,845 lines, 23 internal imports │
│   panels/dialogs: stack_panel, workspace_panel, chat_panel,   │
│                   one_click_dialog, task_config_dialog, …     │
│   widgets/: map_widget, timeline_widget, …                    │
├──────────────────────────────────────────────────────────────┤
│ stack_runner.py ── bridge: Qt Signals, but imports ONLY       │
│                    services (never GUI)                       │
├──────────────────────────────────────────────────────────────┤
│ Qt-free services (19 modules)                                 │
│   plan_service      pipeline_service    output_service        │
│   sensor_service    video_service       photogrammetry_service│
│   anomaly_service   point_cloud_pipeline qgis_export          │
│   netcdf_export     qc_report           interval_io           │
│   config_service    preset_service      frame_stats           │
│   dynamicsampling   timeutil            build_anomaly_site_…  │
├──────────────────────────────────────────────────────────────┤
│ models.py ── leaf: 14 dataclasses, imports nothing internal   │
└──────────────────────────────────────────────────────────────┘
```

**Rule 1 — `models.py` imports nothing internal.** 14 dataclasses with 20
`to_dict`/`from_dict` methods. The entire workspace round-trips to JSON, which is
what makes save/load and job history work.

**Rule 2 — services never import Qt.** They accept a `log_fn` callable instead of
emitting signals. This is why the catalog builder, plan builder, raster export
and output queue can all be exercised headlessly.

`stack_runner` is the deliberate exception: it *is* a `QObject` (it emits
progress signals) but imports only services. It is the seam between the two
worlds.

---

## 3. The domain model (`models.py`)

| Class | Role |
|---|---|
| `VideoRecord` | one video file: path, start/end time, fps |
| `SensorChannel` / `SensorFileConfig` | one column / one CSV of sensor data |
| `TimeValueSourceConfig` / `NavigationConfig` | lat, lon, alt sources |
| `SelectedTimeRange` | a `[start, end]` interval — the atom of a Job |
| `Job` | named set of intervals + settings snapshot |
| `Task` / `TaskStack` | a unit of work + the ordered list of them |
| `SegmentRecord`, `PhotogrammetryRun` | processing history |
| `ThresholdConfig`, `AnnotationConfig` | analysis / render settings |

`TASK_INFO` registers **15 task types** across 6 categories (Prepare, Sampling,
Outputs, Photogrammetry, Anomaly, Export), each declaring its data requirements
(`nav`, `sensor`, `interp`, `video`) and whether it fans out per channel.

---

## 4. Execution model

### Worker pattern
Every long operation follows one shape:

```
QObject worker  ──moveToThread──►  QThread
      │  signals: log / status / progress / finished / error
      ▼
   MainWindow slots (main thread)
```

Five workers: `PipelineWorker`, `OutputWorker`, `AnomalyWorker`,
`PhotogrammetryWorker`, `StackWorker`.

### ⚠ The load-bearing threading rule

PySide6 decides **where a slot runs from the type of the connected object**.
This was measured, not assumed:

| Connection target | Runs on |
|---|---|
| bound method of a `QObject` | **main thread** (auto-queued) ✅ |
| real C++ slot (`label.setText`) | **main thread** (auto-queued) ✅ |
| **lambda** | **emitting (worker) thread** ❌ |
| **plain function** | **emitting (worker) thread** ❌ |

A lambda on a worker signal therefore executes on the worker thread. If it
touches a widget — or constructs a `QObject` parented to one — Qt aborts:

```
QObject: Cannot create children for a parent that is in a different thread.
```

**Three shipped bugs came from exactly this.** It is now enforced by
`tools/lint_qt_threading.py` in CI.

### Teardown
The house pattern, used by every worker:

```python
worker.finished.connect(thread.quit)
worker.error.connect(thread.quit)
thread.finished.connect(_cleanup_this_thread)   # guarded ref clear
thread.finished.connect(worker.deleteLater)
thread.finished.connect(thread.deleteLater)
```

Omitting `deleteLater` leaks one `QThread` per run, because `QThread(self)`
parents it to the window.

---

## 5. Task Stack — the orchestration core

The strongest design idea in the application.

```
Task objects            plan_service.build_plan()          StackWorker
(type, target,   ──►    ① resolve targets → Scopes   ──►   sequential
 settings,              ② expand → flat step dicts         execution
 channels,
 depends_on)
```

**Stage ① — scope resolution.** A target of `{"kind": "full"|"job"|"jobs"|
"all_jobs"}` expands to zero or more `Scope(scope_id, interp_path, output_dir,
label, job)`. Jobs without intervals are dropped and reported as skips.

**Stage ② — step expansion.** Each (task, scope) pair produces one or more step
dicts. Per-channel types fan out: one "Sensor 3D PLY" task × 5 channels × 2 jobs
= 10 steps.

### 🔑 Critical invariant
> **Every target is resolved BEFORE the stack runs.**
> A task can never target a job that a later task in the same stack creates.

This is why One-Click *materialises* its anomaly job at generation time rather
than chaining it behind the detector step. Any future "detect → immediately
sample" feature requires late-binding target resolution in `plan_service`.

### One-Click
A **task generator**, not a parallel execution path. It emits ordinary `Task`
objects into the normal stack, which stay individually editable. There is one
execution engine, and that is deliberate.

---

## 6. The MATLAB boundary

```
GrapherMatrix.m ─► .mat ─► Export*.m ─► event CSVs ─► Python catalog ─► PDF/CSV/GeoJSON/QGIS
└──────────── MATLAB (optional) ────────┘ └─────────── Python (always) ──────────┘
```

The seam is **event CSVs on disk**. Only the left half needs MATLAB, so the app
degrades gracefully when it is absent: `anomaly_service.matlab_unavailable_reason()`
explains why, and the catalog still rebuilds from existing CSVs.

Invocation is `matlab -batch` with:
- `cd(workdir)` — `GrapherMatrix.m` opens a **relative** `interp_full.csv`
- `repo` injected — the two exporters honour it instead of a hardcoded path

The detector evaluates 4 baselines × 4 detectors × 5 smoothers × 2 regimes per
channel across 4 run configurations; the Python side fuses those into windows,
spatial sites and a video-review queue.

---

## 7. Invariants and conventions

### Time — the one that bites
**All datetimes are naive and represent UTC.** Survey timestamps, job intervals,
video ranges and sensor tables are all one timezone by project convention.

- `interval_io._parse_timestamp` is the **single** parser; it strips a trailing `Z`.
- Unix conversion treats naive as UTC (matching pandas `astype("int64")`).
  Using `datetime.timestamp()` instead applies the machine's local offset — that
  bug would have shifted every anomaly band by 5 h on an EST machine.
- `timeutil.utc_now()` / `utc_from_timestamp()` replace the deprecated
  `datetime.utcnow()` / `utcfromtimestamp()` **while preserving naive
  semantics**. The API the deprecation warning suggests returns *aware*
  datetimes, which raise `TypeError` against everything else in the app.

### Other conventions
- **`log_fn` injection** is the universal service-logging contract.
- Products are **workspace-relative**; the workspace JSON is the unit of persistence.
- **`matplotlib.use("Agg")` before importing pyplot** in any module the GUI may
  run on a worker thread. `qc_report` and `point_cloud_pipeline` avoid pyplot
  entirely for this reason.

---

## 8. Testing & CI

```
tests/test_plan_service.py   26 tests — scope resolution, fan-out, ordering,
                                        skips, workspace-level invariants
tests/test_timeutil.py       11 tests — bit-for-bit equivalence with the
                                        deprecated APIs, naive-ness, arithmetic
```
37 tests, ~0.02 s. They need no `QApplication` — that is the payoff of the
service layering.

`.github/workflows/checks.yml` runs four gates:

1. **byte-compile** every module
2. **pyflakes** — undefined names are hard errors (this class shipped once:
   matplotlib symbols used in a function that never imported them, silently
   breaking raster map export)
3. **`tools/lint_qt_threading.py`** — AST checker (not grep; multi-line
   `connect(` defeats grep) enforcing the two whole-app rules: no lambda/plain-
   function slots on worker signals, no deprecated datetime APIs
4. **pytest**

CI installs only `pytest` + `pyflakes` — no Qt, no geo stack, no display server.

---

## 9. Where to start reading

| Goal | Read |
|---|---|
| The data vocabulary | `models.py` (leaf, no deps) |
| How work is orchestrated | `plan_service.py` → `stack_runner.py` |
| How products get made | `output_service.py`, `pipeline_service.py` |
| The GUI hub | `main_window.py` — start at `_build_task_plan`, `_run_stack` |
| Anomaly pipeline | `anomaly_service.py` → `build_anomaly_site_catalog.py` |

---

## 10. Risk register

| Risk | Severity | Notes |
|---|---|---|
| **`main_window.py` still 8,845 lines / 237 methods / ~320 attributes** | High | The dominant structural debt. It is view + controller + thread manager. Next extraction candidates: the export/render functions (`_raster_export_map`, `_viz_export_map`) and the config builders. |
| **No integration tests in-repo** | Medium | The 8 integration smoke tests written during review live in scratch. Promoting them to `tests/` (with a Qt marker) would guard the worker paths. |
| **`_build_task_plan` is a 260-line `elif` chain** | Medium | Now isolated and tested, so safe to refactor. A per-task-type registry (`{task_type: builder_fn}`) would make adding types additive. |
| **61% of anomaly windows are HIGH** | Medium | 300/492. Worth revisiting the `score ≥ 65` threshold if HIGH is meant to be a triage shortlist. |
| **Silent `except Exception:` blocks** (45) | Low–Med | Mostly guarding optional features. Each hides a real failure mode; worth auditing the ones in `viewer_widget` (7) and `output_service` (3) first. |
| **MATLAB not testable in CI** | Low | Accepted; the seam is file-based so the Python half is fully covered. |
| **Metashape untestable under WSL** | Low | Known constraint; validated on Windows. |

### Strengths worth protecting
The service/GUI separation, dataclass round-tripping, a single execution engine,
graceful degradation for every optional external tool (MATLAB, Metashape,
COLMAP), and honest in-code documentation of hazards — `qc_report`'s pyplot
warning was correct and prescient.

**Every bug found in review was at a boundary**, not in core logic: copy-paste
between sibling functions, a lambda in a callback chain, a CLI script pulled onto
a GUI thread. That is a good sign about the interior.

---

## Appendix — bugs found and fixed (2026-07-24 review)

| # | Bug | Impact |
|---|---|---|
| 1 | `_raster_export_map` used 5 undefined matplotlib names | Raster map export **never worked** |
| 2 | Lambda slot on `worker.finished` in the per-channel output queue | **Process abort** on multi-channel sensor products |
| 3 | `build_anomaly_site_catalog` imported pyplot with no backend guard | **Abort** when run from the GUI thread |
| 4 | `AnomalyWorker` missing `deleteLater`; blocking `wait()` on GUI thread | QThread leak per run, UI stall |
| 5 | `viewer_widget` updated the status bar from the worker thread | Undefined behaviour during Potree conversion |
| 6 | `chat_panel` ran `_on_finished` on the worker thread | Chat history mutated off-thread |
| 7 | `installer/create_icon.py` invalid escape `installer\ ` | `SyntaxWarning` now, **`SyntaxError`** in a future Python |
| 8 | 21 deprecated `datetime.utcnow`/`utcfromtimestamp` call sites | Scheduled removal; naive fix would have broken interval maths |

Bugs 5 and 6 were found **by the new CI linter**, after it was written to prevent
recurrences of bug 2.
