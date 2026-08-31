"""Tests for workspace_migrate.py — OLD-layout → .eprproj migration.

Pins the two things a migration must never get wrong: destinations are minted
exactly where WorkspaceLayout says they go, and the copy is non-destructive
(the original workspace is untouched, so a failed migration loses nothing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import layout
from layout import WorkspaceLayout
import workspace_migrate as wm


# --------------------------------------------------------------------------
# fixture: a realistic old-layout workspace
# --------------------------------------------------------------------------
def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def old_ws(tmp_path) -> Path:
    ws = tmp_path / "J1756 Dive 12"
    _write(ws / "interp_full.csv", "t,lat,lon\n1,2,3\n")
    _write(ws / "outputs" / "nav_depth" / "run_001" / "nav_depth.tif")
    _write(ws / "job_002_downdive" / "outputs" / "sensor_2d" / "co2"
           / "run_001" / "co2_2d.tif")
    _write(ws / "job_002_downdive" / "filtered_interp.csv")
    _write(ws / "job_002_downdive" / "filtered_interp.csv.meta.json", "{}")
    _write(ws / "sampling_5_job_002_downdive" / "segment_001_x_y" / "frames" / "f.jpg")
    _write(ws / "sampling_9_full" / "segment_001_a_b" / "frames" / "g.jpg")
    _write(ws / "anomaly_site_catalog" / "site.csv")
    _write(ws / "logs" / "20260101_000000" / "run.log")
    _write(ws / "weird_stuff" / "x.txt", "surprise")
    _write(ws / "workspace.json", json.dumps({"name": "J1756 Dive 12", "jobs": [2]}))
    return ws


# --------------------------------------------------------------------------
# plan destinations
# --------------------------------------------------------------------------
def test_plan_destinations(old_ws, tmp_path):
    bundle = tmp_path / "out.eprproj"
    lo = WorkspaceLayout(bundle)
    plan = wm.build_migration_plan(str(old_ws), str(bundle))
    dsts = {op.src: op.dst for op in plan}

    # interp_full → inputs/interp_full.csv
    assert dsts[str(old_ws / "interp_full.csv")] == str(lo.interp_full)

    # survey products keep their subfolder tree under survey/
    assert dsts[str(old_ws / "outputs" / "nav_depth")] == \
        str(lo.products_dir("survey") / "nav_depth")

    # per-job products land under jobs/job_002/products/
    assert dsts[str(old_ws / "job_002_downdive" / "outputs" / "sensor_2d")] == \
        str(lo.products_dir(2) / "sensor_2d")

    # per-job filtered interp + its meta sibling
    assert dsts[str(old_ws / "job_002_downdive" / "filtered_interp.csv")] == \
        str(lo.job_filtered_interp(2))
    assert dsts[str(old_ws / "job_002_downdive" / "filtered_interp.csv.meta.json")] == \
        str(lo.job_dir(2) / "filtered_interp.csv.meta.json")

    # per-job frames under jobs/job_002/frames/run_001/segments/
    assert dsts[str(old_ws / "sampling_5_job_002_downdive" / "segment_001_x_y")] == \
        str(lo.frames_run_dir(2, run_id=1) / "segment_001_x_y")

    # full-dataset frames under survey/frames/run_001/segments/
    assert dsts[str(old_ws / "sampling_9_full" / "segment_001_a_b")] == \
        str(lo.products_dir("survey") / "frames" / "run_001" / "segments"
            / "segment_001_a_b")

    # anomaly catalog → survey/anomaly/
    assert dsts[str(old_ws / "anomaly_site_catalog" / "site.csv")] == \
        str(lo.products_dir("survey") / "anomaly" / "site.csv")

    # logs → logs/
    assert dsts[str(old_ws / "logs" / "20260101_000000")] == \
        str(lo.logs_dir() / "20260101_000000")

    # workspace.json is folded into project.json, not copied verbatim
    assert str(old_ws / "workspace.json") not in dsts


def test_unrecognized_goes_to_archive(old_ws, tmp_path):
    bundle = tmp_path / "out.eprproj"
    lo = WorkspaceLayout(bundle)
    plan = wm.build_migration_plan(str(old_ws), str(bundle))
    dsts = {op.src: op.dst for op in plan}

    assert dsts[str(old_ws / "weird_stuff")] == \
        str(lo.archive_dir() / "imported" / "weird_stuff")


def test_job_id_parsed(old_ws):
    # sampling + job dirs both key off the leading job_(\d+) integer
    assert wm._JOB_DIR_RE.match("job_002_downdive").group(1) == "002"
    assert int(wm._JOB_DIR_RE.match("job_002_downdive").group(1)) == 2
    assert int(wm._SAMPLING_JOB_RE.match("sampling_5_job_002_downdive").group(1)) == 2
    assert wm._JOB_DIR_RE.match("job_7") is not None
    assert wm._SAMPLING_FULL_RE.match("sampling_9_full") is not None


def test_plan_is_deterministic(old_ws, tmp_path):
    bundle = tmp_path / "out.eprproj"
    p1 = wm.build_migration_plan(str(old_ws), str(bundle))
    p2 = wm.build_migration_plan(str(old_ws), str(bundle))
    assert [(o.src, o.dst, o.kind) for o in p1] == \
        [(o.src, o.dst, o.kind) for o in p2]


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------
def test_execute_copies_everything_non_destructively(old_ws, tmp_path):
    bundle = tmp_path / "out.eprproj"
    lo = WorkspaceLayout(bundle)
    plan = wm.build_migration_plan(str(old_ws), str(bundle))
    pj = wm.build_project_json(str(old_ws))

    result = wm.execute_plan(plan, pj)

    assert result["errors"] == []
    assert result["copied"] == len(plan)
    assert result["bytes"] > 0

    # destinations exist in the new bundle
    assert lo.interp_full.is_file()
    assert (lo.products_dir("survey") / "nav_depth" / "run_001" / "nav_depth.tif").is_file()
    assert (lo.products_dir(2) / "sensor_2d" / "co2" / "run_001" / "co2_2d.tif").is_file()
    assert lo.job_filtered_interp(2).is_file()
    assert (lo.job_dir(2) / "filtered_interp.csv.meta.json").is_file()
    # job frames landed under jobs/job_002/frames/run_001/segments/
    assert (lo.frames_run_dir(2, run_id=1) / "segment_001_x_y" / "frames" / "f.jpg").is_file()
    assert (lo.products_dir("survey") / "frames" / "run_001" / "segments"
            / "segment_001_a_b" / "frames" / "g.jpg").is_file()
    assert (lo.products_dir("survey") / "anomaly" / "site.csv").is_file()
    assert (lo.logs_dir() / "20260101_000000" / "run.log").is_file()
    assert (lo.archive_dir() / "imported" / "weird_stuff" / "x.txt").is_file()

    # NON-DESTRUCTIVE: every original still present
    assert (old_ws / "interp_full.csv").is_file()
    assert (old_ws / "outputs" / "nav_depth" / "run_001" / "nav_depth.tif").is_file()
    assert (old_ws / "job_002_downdive" / "filtered_interp.csv").is_file()
    assert (old_ws / "sampling_5_job_002_downdive" / "segment_001_x_y" / "frames" / "f.jpg").is_file()
    assert (old_ws / "weird_stuff" / "x.txt").is_file()

    # project.json written at bundle root with provenance
    pj_path = bundle / "project.json"
    assert pj_path.is_file()
    saved = json.loads(pj_path.read_text(encoding="utf-8"))
    assert saved["schema"] == "eprproj"
    assert saved["migrated_from"] == str(old_ws.resolve())
    assert "migrated_at" in saved
    assert saved["name"] == "J1756 Dive 12"  # carried over from workspace.json


def test_execute_progress_callback(old_ws, tmp_path):
    bundle = tmp_path / "out.eprproj"
    plan = wm.build_migration_plan(str(old_ws), str(bundle))
    seen = []
    wm.execute_plan(plan, None, progress=lambda done, total, op: seen.append((done, total)))
    assert seen[-1] == (len(plan), len(plan))
    assert [d for d, _ in seen] == list(range(1, len(plan) + 1))


def test_execute_skips_missing_source_without_raising(tmp_path):
    # A source that does not exist is skipped, not copied, and never raises.
    bundle = tmp_path / "out.eprproj"
    plan = [
        wm.MoveOp(str(tmp_path / "does_not_exist"), str(bundle / "inputs" / "x"), "file"),
    ]
    result = wm.execute_plan(plan, None)
    assert result["skipped"] == 1
    assert result["copied"] == 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def test_default_bundle_path_slugifies(tmp_path):
    ws = tmp_path / "J1756 Dive 12"
    ws.mkdir()
    out = wm.default_bundle_path(str(ws))
    assert Path(out).name == "j1756_dive_12.eprproj"
    assert Path(out).parent == tmp_path


def test_summarize_plan_groups_by_area(old_ws, tmp_path):
    bundle = tmp_path / "out.eprproj"
    plan = wm.build_migration_plan(str(old_ws), str(bundle))
    text = wm.summarize_plan(plan)
    assert "Inputs: interp_full.csv" in text
    assert "Survey products:" in text
    assert "Survey anomaly:" in text
    assert "Job 002:" in text
    assert "frames" in text and "products" in text
    assert "Logs:" in text
    assert "Archive (unrecognized):" in text


def test_build_project_json_without_workspace_json(tmp_path):
    ws = tmp_path / "bare"
    ws.mkdir()
    pj = wm.build_project_json(str(ws))
    assert pj["schema"] == "eprproj"
    assert pj["migrated_from"] == str(ws.resolve())
    assert "migrated_at" in pj
