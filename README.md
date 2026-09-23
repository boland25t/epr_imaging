# EPR Imaging

Desktop app (PySide6) that turns one ROV dive — downward video, navigation and
sensor channels — into georeferenced products: tracklines, sensor rasters,
frame sets, fauna detections, anomaly detections, photogrammetry (clouds,
meshes, DEM, orthomosaics) and a survey report. Built for Woods Hole
Oceanographic Institution. Version: [`version.txt`](version.txt).

**The app is the simple UI, `simple_main.py`.** One window per dive
workspace (`<dive>.eprproj`): pick a product, generate it (or run the Default
Run for all of them), and view it in place. What the UI promises is written
down in [`docs/simple_ui_contract.md`](docs/simple_ui_contract.md). A user
manual is on the way.

---

## Launch

```bash
./launch_simple_ui.sh /path/to/<dive>.eprproj   # open one dive workspace
./launch_simple_ui.sh --pick                     # choose a workspace in a folder dialog
```

The launcher works from any directory. It runs `simple_main.py` with
`$EPR_PYTHON` if set, else `<repo>/.venv/bin/python` if present, else `python3`.
On WSL it sets `QT_QPA_PLATFORM=xcb` (a Wayland window renders blank) unless
you set it yourself. Direct equivalent: `python simple_main.py [workspace]`.

## Install (WSL2 or native Linux, Python 3.12/3.13)

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -U pip
# torch first, matched to your hardware:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # NVIDIA GPU
#   or: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`requirements.txt` gives floors at the tested versions.
[`requirements-lock.txt`](requirements-lock.txt) pins the exact tested set
(Python 3.13.13, Linux x86-64, CUDA 13.0 torch). Use it for a reproducible or
offline install: build a wheelhouse with `pip download` on a connected machine
before sailing.

## Prerequisites

| Component | Needed for | Tested version | Without it |
|---|---|---|---|
| Python + PySide6, numpy, pandas, matplotlib | the app itself | 3.13.13 · 6.11.1 · 2.4.6 · 3.0.3 · 3.10.9 | app won't start |
| scipy, rasterio, opencv-python, Pillow, utm, pykrige, plyfile, netCDF4 | rasters, frame extraction, PLY/NetCDF output | see `requirements.txt` | the affected step fails with a red `!!` line |
| pyproj | anomaly UTM layers | 3.7.2 | layers skipped (`note:` line) |
| pyvista + pyvistaqt (+vtk) | in-app 3-D viewer | 0.49.0 / 0.13.1 / 9.7.0 | 3-D view not offered; everything else works |
| torch + torchvision + ultralytics | fauna detection (YOLOv8) | 2.14.0 / 0.29.0 / 8.4.155 | fauna step fails |
| Fauna weights `mbari_315k_yolov8.pt` (132 MB, not in git) | fauna detection | — | fauna step fails. See `FAUNA_WEIGHTS` in `product_catalog.py` for where the Default Run looks. For a single run, set the path under Generate New… ▸ Fauna detection |
| NVIDIA GPU + driver (Windows driver with WSL CUDA on WSL) | fast fauna inference, Metashape | RTX 4500 Ada | CPU fallback (slower) |
| Agisoft Metashape Professional, Windows install, activated | photogrammetry, orthos, DEM, survey report | 2.3.1 | photogrammetry and survey report fail. Supported setup: WSL2 + Windows Metashape |
| MATLAB on PATH + Signal Processing Toolbox | anomaly detection (`GrapherMatrix.m`) | R2026a (needs R2019a or later) | anomaly step fails. Use a licence that works offline |
| Chrome / Chromium / Edge | high-quality catalogue PDF | — | falls back to a plainer matplotlib PDF |
| QGIS | opening the generated `.qgs` projects | — | projects are still written |

ffmpeg and COLMAP are not needed. Keep workspaces on a drive Windows can see
(`/mnt/<x>`) when Metashape runs from WSL.

## Layout

- `simple_main.py` (UI) → `product_catalog.py` (the product list and how each
  product is generated) → the `*_service.py` modules and product builders in
  the repo root.
- `metashape_worker.py` is copied next to the Windows Metashape exe and run
  there. `GrapherMatrix.m`, `ExportFineConsensusEvents.m`,
  `ExportDetectorFamilyEvents.m` and `navSpeed.m` are run by MATLAB from the
  repo root. Keep all of these in the root.
- `worm_cover.py`, `morphotype.py`, `fauna_ortho.py`, `dive_setup.py` are
  standalone CLIs that the UI does not call.
- `tests/` — `python -m pytest tests`.
- [`archive/`](archive/README.md) holds code the app no longer uses. It is
  kept for history and not maintained.

Legacy: the old `app.py` UI and its Windows installer (`EPRSamplingToolSetup.exe`,
`setup.ps1`, `launch.bat`) are archived under `archive/legacy_ui/` and unsupported.

## Docs

- [`docs/simple_ui_contract.md`](docs/simple_ui_contract.md) — what the simple UI does and guarantees
- [`docs/DEV_SETUP.md`](docs/DEV_SETUP.md) — developer environment notes (partly legacy)
- [`docs/anomaly/`](docs/anomaly/) — anomaly detector design notes
