# archive/

Code the simple UI (`launch_simple_ui.sh` -> `simple_main.py`) does not use. Kept for history and
occasional re-runs, not maintained. See the final-review archive plan (09_archive_plan.md) for why each
file moved.

- `legacy_ui/` - the old `app.py` / `main_window` application, its installer, and its tests
- `analysis_scripts/` - one-off paper figures, cross-dive maps, decks, multi-view dedup
- `biigle/` - BIIGLE annotation bridge / export / ingest
- `training/` - YOLO fauna fine-tune + review (keep `deploy_predict_offline.py` beside `fauna_finetune.py`)
- `doc_generators/` - legacy handbook generators and PDF
- `legacy_3dvistool/` - old 3-D example (duplicates root `point_cloud_pipeline.py`)

Archived scripts import modules from the repo root and from each other. Run them with:

```bash
PYTHONPATH=<repo>:<repo>/archive/analysis_scripts:<repo>/archive/biigle python3 archive/<dir>/<script>.py
```

Legacy app (best effort, unsupported): `cd archive/legacy_ui && PYTHONPATH=../..:. python3 app.py`
