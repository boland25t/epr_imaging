# Simple UI overhaul — module contract (2026-09-18)

Two new modules; legacy UI untouched. `simple_main.py` (Qt) imports only from
`product_catalog.py` (Qt-free) plus stdlib/Qt.

## product_catalog.py (backend, Qt-free)

```python
WHOLE_TRACKLINE = "__whole__"          # job id sentinel

@dataclass
class Interval:            # UTC unix seconds
    t0: float
    t1: float

@dataclass
class Job:
    job_id: str            # "job_0001" or WHOLE_TRACKLINE
    name: str              # display, e.g. "Job 3 (2 intervals)"
    intervals: list[Interval]   # empty for WHOLE_TRACKLINE

@dataclass
class ProductInstance:
    type_key: str
    label: str             # "2026-09-18 14:02 · dynamic 0.25 m" (time + characteristic)
    path: str              # primary artifact path
    created_at: float
    job_id: str
    view_paths: list[str]  # displayable files (png/tif/pdf/csv/geojson)

@dataclass
class ProductType:
    key: str
    label: str
    settings_schema: list[tuple]   # (key, label, kind 'float|int|str|choice', default, extra)
    # methods:
    def discover(ws: str, job: Job) -> list[ProductInstance]
    def generate(ws: str, job: Job, settings: dict, log_fn) -> ProductInstance

PRODUCT_TYPES: list[ProductType]   # order = display order
```

Types (exactly): `frame_set`, `photogrammetry`, `fauna_detection`, `nav_trackline`,
`depth_raster`, `sensor_raster`, `anomaly_detection`, `spectrum_trackline`,
`anomaly_trackline`, `survey_report`. NO sensor interpolation cloud (3-D PLY) anywhere.

Sampling identity rule (2026-09-21): SAMPLING-DEPENDENT types (`frame_set`,
`photogrammetry`, `fauna_detection` — anything consuming frames) have identity
(job_id, sampling regime); the regime leads the instance label ("dynamic 0.25 m · ...")
and a job may hold multiple sampling runs, each with its own interp.csv/frame set/
meshes/fauna products. SAMPLING-INDEPENDENT types (all others; computed from the 1 Hz
table) have job-only identity, are computed once per job, and are listed once no matter
how many sampling runs exist. `fauna_detection` wraps fathomnet_detect + fauna_timeseries
+ fauna_occurrences on the recycled frame set (zero-shot MBARI weights, deploy_config
per-bucket thresholds; sub-threshold rows kept flagged `below_conf`); dive-wide default-
sampling runs write to <ws>/survey/fauna/ (canonical, REPLACES census — loud note first),
job/non-default runs to <scope>/fauna_runs/<DIVE>_fauna_<stamp>__<sampling>/.
Catalog: `catalog_builder.build_catalog(ws, job_id, log)` wired at Run menu + job
right-click (post-run step, never inside default_run_all).
- `spectrum_trackline` = log10 colour-spectrum trackline per channel (turbo), from
  interp_full — the slide_deck.render_log_trackline treatment, single dive, saved PNG
  under survey/spectrum_trackline/.
- `anomaly_trackline` = trackline with tier-coloured window strokes (multi_dive_map
  single-dive treatment) saved PNG under survey/anomaly_trackline/; discover() also
  surfaces the anomaly geojsons.

## Job store
```python
def load_jobs(ws) -> list[Job]                 # always includes WHOLE_TRACKLINE first
def create_job(ws, intervals, base_job=None) -> Job
    # base_job given => NEW job with base intervals + new ones (spec: never mutate)
def track_polyline(ws) -> np.ndarray           # Nx3 [unix_time, easting, northing]
```
Jobs persist in workspace.json under "simple_jobs": [{job_id,name,intervals:[[t0,t1],...]}].

## Execution + policies
```python
def default_run_all(ws, log_fn, job=None) -> None   # full default suite for job/whole
DEFAULTS = {photogrammetry: chunks TARGET 250-350 frames (hard cap 350; a span >350 splits into near-equal chunks within the band; spans <=350 stay one chunk), altitude gate: EXCLUDE
            spans where alt > 8 m (drop frames alt>8; split intervals accordingly),
            sampling: dynamic 0.25 m}
```
Frame recycling (core): frame_set.generate samples per contiguous covered span into
survey/photogrammetry/-style segment dirs keyed by time range; before sampling, scan
existing frame sets whose [t0,t1] covers/overlaps the request and REUSE frames whose
timestamps fall inside (union, no duplicate extraction). Reuse via existing
segment_*/interp.csv manifests. Sampling grid must be deterministic from unix_time so
subsets compose (grid anchored at floor(t0 of dive)).

## simple_main.py (Qt, PySide6)
- QMainWindow "EPR Imaging", QSplitter: [Products | Trackline | Logger].
- Menu File: Save workspace / Save workspace as / Import navigation and orientation /
  Import video / Import new sensor channel. Menu Run: "Default run on all".
- Products panel: top = job selector QComboBox (from load_jobs, auto-refresh);
  below = QTreeWidget, one top-level item per ProductType; expanding shows
  "＋ Generate New" child + discovered instances (label). Double-click/right-click
  instance → "View" opens ProductViewer window (image/pdf/csv/geojson-aware).
  Right-click also offers "Open containing folder".
- Trackline panel: matplotlib FigureCanvas of easting/northing track (time-ordered);
  ZOOMABLE: mouse-wheel zoom centred on cursor + drag-pan (and a Reset View button);
  click twice = interval (nearest track point time); each interval-defining click drops a
  visible star/dot marker AT THE CLICK POINT immediately (cleared with the pending
  interval); pending intervals drawn; two
  QDateTimeEdit (UTC) + "Add typed interval"; buttons: "Add interval to job" (appends
  to pending set), "Create job from intervals" (create_job; if a base job is selected
  in the selector, pass it → new derived job), "Clear". Selected job's intervals
  always highlighted.
- Logger panel: QPlainTextEdit read-only; all generate/default_run_all run in a
  QThread; log_fn marshalled via signal. One task at a time; Run menu disabled while
  running.
- Selecting a job re-discovers and filters every type's instances to that job
  (WHOLE_TRACKLINE shows whole-track products only).
- Entry: `python3 simple_main.py <workspace>`; also loads last workspace from
  ~/.epr_simple_ui.json.

## Ownership
- Agent BACKEND: product_catalog.py only.
- Agent FRONTEND: simple_main.py only (code against this contract; do not import
  legacy main_window; reuse ConfigService for save/import wiring).
Integration, testing, fixes: coordinator.
