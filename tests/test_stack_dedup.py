"""Integration test: StackWorker consults the registry to skip duplicate runs.

The manifest/registry primitives are unit-tested in test_manifest.py; this pins
the wiring in stack_runner — that a recorded run makes an identical future step a
skip, and that changing settings, editing input data, or losing the outputs each
re-opens the step for execution.  Requires Qt (offscreen).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _worker(ws: Path):
    from stack_runner import StackWorker
    w = StackWorker.__new__(StackWorker)
    StackWorker.__init__(w, plan=[], skip_existing=True, workspace_dir=str(ws))
    w._log_fh = None
    w._emit = lambda m="": None
    w._init_provenance()
    return w


def _step(interp: Path, outdir: Path, cell=5.0):
    return {"product_type": "nav_2d", "scope_id": "job_2", "channel": None,
            "kwargs": {"interp_path": str(interp), "output_dir": str(outdir),
                       "cell_size_m": cell}}


def _record_one(ws: Path, interp: Path, outdir: Path):
    w = _worker(ws)
    run_dir = outdir / "nav" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    prod = run_dir / "nav.tif"
    prod.write_bytes(b"T" * 64)
    w._record_run_manifest(_step(interp, outdir), "completed", [str(prod)], 0.0, 1.0)
    w._save_registry()
    return prod


def test_identical_step_is_deduped(tmp_path, qapp):
    interp = tmp_path / "interp_full.csv"
    interp.write_text("t,co2\n2026-01-01T00:00:00,1000\n")
    outdir = tmp_path / "outputs"; outdir.mkdir()
    _record_one(tmp_path, interp, outdir)
    w = _worker(tmp_path)
    dup = w._registry_duplicate(_step(interp, outdir))
    assert dup is not None and dup["run_id"].startswith("r_")


def test_changed_setting_reruns(tmp_path, qapp):
    interp = tmp_path / "interp_full.csv"
    interp.write_text("t,co2\n2026-01-01T00:00:00,1000\n")
    outdir = tmp_path / "outputs"; outdir.mkdir()
    _record_one(tmp_path, interp, outdir)
    w = _worker(tmp_path)
    assert w._registry_duplicate(_step(interp, outdir, cell=10.0)) is None


def test_edited_input_reruns(tmp_path, qapp):
    interp = tmp_path / "interp_full.csv"
    interp.write_text("t,co2\n2026-01-01T00:00:00,1000\n")
    outdir = tmp_path / "outputs"; outdir.mkdir()
    _record_one(tmp_path, interp, outdir)
    interp.write_text("t,co2\n2026-01-01T00:00:00,9999\n")     # edited in place
    os.utime(interp, (time.time() + 5, time.time() + 5))
    w = _worker(tmp_path)
    assert w._registry_duplicate(_step(interp, outdir)) is None


def test_deleted_outputs_reruns(tmp_path, qapp):
    interp = tmp_path / "interp_full.csv"
    interp.write_text("t,co2\n2026-01-01T00:00:00,1000\n")
    outdir = tmp_path / "outputs"; outdir.mkdir()
    prod = _record_one(tmp_path, interp, outdir)
    prod.unlink()                                              # user deleted it
    w = _worker(tmp_path)
    assert w._registry_duplicate(_step(interp, outdir)) is None
