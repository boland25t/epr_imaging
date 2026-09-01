"""Tests for the Qt-free core of product_tree_widget.

Covers the registry → produced-node mapping the Product Tree badges/expansion
rely on: scope matching (incl. the full/survey alias), status filtering (only
completed runs count), the photogrammetry shared-task-type coarseness, the
per-node run listing, and the synthesised node descriptions.

Pure and Qt-free: importing product_tree_widget pulls in PySide6 but constructs
no widgets, so these run headless without a QApplication.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import product_graph as pg
import product_tree_widget as ptw


def _reg(runs):
    """A stand-in for manifest.Registry: anything with a .runs list works."""
    return SimpleNamespace(runs=runs)


def _run(task_type, scope_id="full", status="completed", channel=None,
         run_id="r_1", finished_at="2026-08-31T10:00:00", output_dir="/out"):
    return {"task_type": task_type, "scope_id": scope_id, "status": status,
            "channel": channel, "run_id": run_id, "finished_at": finished_at,
            "output_dir": output_dir}


# --------------------------------------------------------------------------
# produced_node_ids
# --------------------------------------------------------------------------
def test_empty_registry_produces_nothing():
    assert ptw.produced_node_ids(_reg([]), "full") == set()


def test_completed_run_marks_its_node_produced():
    reg = _reg([_run("sensor_2d")])
    assert "sensor_raster" in ptw.produced_node_ids(reg, "full")


def test_survey_alias_matches_full_scope():
    reg = _reg([_run("nav_3d", scope_id="full")])
    # The tree views the scope as "survey"; a run recorded under "full" still counts.
    assert "trackline" in ptw.produced_node_ids(reg, "survey")
    assert "trackline" in ptw.produced_node_ids(reg, "full")


def test_job_scope_does_not_leak_into_survey():
    reg = _reg([_run("sensor_2d", scope_id="job_2")])
    assert "sensor_raster" not in ptw.produced_node_ids(reg, "full")
    assert "sensor_raster" in ptw.produced_node_ids(reg, "job_2")


def test_none_scope_matches_any():
    reg = _reg([_run("sensor_2d", scope_id="job_2")])
    assert "sensor_raster" in ptw.produced_node_ids(reg, None)


def test_only_completed_runs_count():
    for bad in ("failed", "partial", "skipped"):
        reg = _reg([_run("sensor_2d", status=bad)])
        assert ptw.produced_node_ids(reg, "full") == set()


def test_channel_ignored_any_channel_marks_produced():
    reg = _reg([_run("sensor_2d", channel="co2")])
    assert "sensor_raster" in ptw.produced_node_ids(reg, "full")


def test_photogrammetry_run_marks_all_photogrammetry_nodes():
    # Documented coarseness: the shared "photogrammetry" task type lights up every
    # node that maps to it from a single completed run.
    reg = _reg([_run("photogrammetry")])
    produced = ptw.produced_node_ids(reg, "full")
    assert {"alignment", "mesh", "orthomosaic", "dem", "dense"} <= produced


def test_aggregate_and_root_nodes_never_produced():
    # No task maps to report/video/nav/sensors, so no run can mark them produced.
    reg = _reg([_run("sensor_2d"), _run("nav_3d"), _run("photogrammetry")])
    produced = ptw.produced_node_ids(reg, "full")
    for nid in ("report", "video", "nav", "sensors"):
        assert nid not in produced


# --------------------------------------------------------------------------
# runs_for_node
# --------------------------------------------------------------------------
def test_runs_for_node_newest_first():
    reg = _reg([
        _run("sensor_2d", run_id="r_old", finished_at="2026-08-01T00:00:00"),
        _run("sensor_2d", run_id="r_new", finished_at="2026-08-31T00:00:00"),
    ])
    runs = ptw.runs_for_node(reg, "sensor_raster", "full")
    assert [r["run_id"] for r in runs] == ["r_new", "r_old"]


def test_runs_for_node_filters_status_and_scope():
    reg = _reg([
        _run("sensor_2d", run_id="ok"),
        _run("sensor_2d", run_id="bad", status="failed"),
        _run("sensor_2d", run_id="other", scope_id="job_9"),
    ])
    runs = ptw.runs_for_node(reg, "sensor_raster", "full")
    assert [r["run_id"] for r in runs] == ["ok"]


def test_runs_for_node_empty_for_aggregate():
    reg = _reg([_run("sensor_2d")])
    assert ptw.runs_for_node(reg, "report", "full") == []


# --------------------------------------------------------------------------
# node_description
# --------------------------------------------------------------------------
def test_description_mentions_requirements_and_task():
    desc = ptw.node_description(pg.get_node("mesh"))
    assert "Alignment" in desc          # its required parent's label
    assert "photogrammetry" in desc     # its task type
    assert desc                          # non-empty


def test_description_for_root_and_aggregate():
    assert "root" in ptw.node_description(pg.get_node("video")).lower()
    assert "aggregate" in ptw.node_description(pg.get_node("report")).lower()
