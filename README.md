# EPR Sampling Tool

Desktop GUI (PySide6) for turning underwater survey data — video, navigation,
and sensor channels — into geo-referenced 2D/3D products and photogrammetric
reconstructions. Built for Woods Hole Oceanographic Institution.

Version is tracked in [`version.txt`](version.txt) and shown in the window title.

---

## What it does

**Ingest**
- Scans a directory of video clips, parsing start times from filenames (with
  common-pattern and file-mtime fallbacks) and reading durations from metadata.
- Imports navigation (lat / lon / altitude / depth / heading / pitch / roll
  from any CSV columns) and any number of sensor channels, including `.ppi`
  files and decimal-minute coordinate repair.

**Select intervals**
- Threshold-based (auto-detect windows meeting sensor/nav criteria) or manual
  trackline picking on an interactive map.
- Group intervals into named **Jobs**.

**Task Stack** — the orchestration surface
- Build an ordered, instance-based pipeline of tasks (sampling, sensor/nav 2D &
  3D, depth slices, QGIS export, photogrammetry) and run them top-to-bottom.
- Each task targets the **Full dataset**, **selected jobs**, or **all jobs**
  (batching), and runs sequentially with per-step failure isolation and a
  complete on-disk task log.

**Products**
- Frame extraction (fixed-rate or distance-adaptive) with per-frame nav/sensor
  interpolation, optional CLAHE and annotation overlays.
- 3D PLYs (nav trackline, sensor fields with IDW/Kriging/RBF fill).
- 2D GeoTIFFs, depth-banded slices, PNG slices.
- **Photogrammetry** via COLMAP or Metashape — sparse/dense clouds, meshes,
  camera trajectory, georeferenced to navigation (UTM E/N/−depth) so outputs
  overlay the sensor products.
- One-click QGIS project generation.

**Visualize**
- Embedded 3D viewer (PyVista) for overlaying clouds/meshes in one scene, with
  downsampling budget, translucency (depth peeling), and log color scaling.

---

## Workspace layout

A workspace is one survey/dive. Products are written under it:

```text
<workspace>/
├── interp_full.csv                 # per-frame nav+sensor record
├── inputs/  nav/  sensor/          # copied source CSVs
├── outputs/                        # full-dataset products
│   ├── nav_trackline/  sensor_3d/  sensor_2d/  photogrammetry/  …
├── job_<NNN>_<name>/outputs/       # per-job (batched) products
└── logs/  task_log_<timestamp>.txt # complete run logs
```

---

## Install (end users)

Download and run `EPRSamplingToolSetup.exe`. It installs Python, a virtual
environment, and all dependencies to `%LOCALAPPDATA%\EPRSamplingTool` (no admin
required) and creates a desktop shortcut. See
[`installer/BUILD.md`](installer/BUILD.md) for building the installer.

## Run from source (developers)

```bash
pip install -r requirements.txt
python app.py
```

Full two-instance setup (production install + editable dev clone, native
Windows vs. WSL) is in [`DEV_SETUP.md`](DEV_SETUP.md).

### Optional photogrammetry engines

- **COLMAP** — install a CUDA build, put `colmap` on `PATH`; auto-detected.
- **Metashape Professional** — install it and `pip install` Agisoft's Metashape
  Python 3 module into the venv; then it's detected and driven headlessly.
  (Not importable from WSL unless a Linux Metashape is installed there — see
  [`DEV_SETUP.md`](DEV_SETUP.md).)

---

## Docs

- [`DEV_SETUP.md`](DEV_SETUP.md) — dev environments, two-instance model, shipping checklist
- [`installer/BUILD.md`](installer/BUILD.md) — building the Windows installer
