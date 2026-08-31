"""Tests for layout.py — the workspace path authority.

The whole point of centralizing paths is that these rules are pinned: slugs
never contain spaces, IDs (not human names) go in the path, runs are numbered,
and stored paths are workspace-relative so a renamed workspace stays valid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import layout
from layout import WorkspaceLayout, slugify


# --------------------------------------------------------------------------
# slugify
# --------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("CO2 Concentration", "co2_concentration"),
    ("Down Data w Orientation", "down_data_w_orientation"),
    ("J1756 Job 2 (v5)", "j1756_job_2_v5"),
    ("temperature", "temperature"),
    ("  spaced  out  ", "spaced_out"),
    ("CH4/µatm", "ch4_atm"),
    ("already_slug", "already_slug"),
    ("Nav-Depth", "nav_depth"),
])
def test_slugify(raw, expected):
    assert slugify(raw) == expected


def test_slugify_never_blank():
    assert slugify("") == "item"
    assert slugify("!!!") == "item"
    assert slugify("...", default="x") == "x"


def test_slug_has_no_spaces_ever():
    for name in ("CO2 Concentration", "a b c", "  ", "Weird / Name : here"):
        assert " " not in slugify(name)


# --------------------------------------------------------------------------
# path construction
# --------------------------------------------------------------------------
def L(tmp_path) -> WorkspaceLayout:
    return WorkspaceLayout(tmp_path / "J1756_down.eprproj")


def test_top_level_paths(tmp_path):
    lo = L(tmp_path)
    assert lo.interp_full.as_posix().endswith("/inputs/interp_full.csv")
    assert lo.project_json.name == "project.json"
    assert lo.registry_json.as_posix().endswith("/runs/registry.json")


def test_job_dir_uses_id_not_name(tmp_path):
    lo = L(tmp_path)
    # the human name never appears in the path — only the zero-padded id
    assert lo.job_dir(7).name == "job_007"
    assert lo.job_json(7).as_posix().endswith("/jobs/job_007/job.json")
    assert lo.job_filtered_interp(7).name == "filtered_interp.csv"


def test_product_run_dir_slugifies_channel(tmp_path):
    lo = L(tmp_path)
    p = lo.product_run_dir(7, "sensor_2d", 3, channel="CO2 Concentration")
    posix = p.as_posix()
    assert "/jobs/job_007/products/rasters/co2_concentration/run_003" in posix
    assert " " not in posix                       # the whole point


def test_survey_scope_vs_job_scope(tmp_path):
    lo = L(tmp_path)
    survey = lo.product_run_dir("survey", "nav_2d", 1)
    job = lo.product_run_dir(2, "nav_2d", 1)
    assert "/survey/rasters/nav_depth/run_001" not in survey.as_posix()  # nav_2d folder is 'rasters'
    assert "/survey/rasters/run_001" in survey.as_posix()
    assert "/jobs/job_002/products/rasters/run_001" in job.as_posix()


def test_nav_trackline_folder(tmp_path):
    lo = L(tmp_path)
    p = lo.product_run_dir(1, "nav_3d", 2)
    assert "/jobs/job_001/products/tracklines/run_002" in p.as_posix()


def test_photogrammetry_per_chunk_layout(tmp_path):
    lo = L(tmp_path)
    proj = lo.photogrammetry_project(3, 2)
    chunk = lo.chunk_dir(3, 2, 0)
    assert proj.as_posix().endswith("/products/photogrammetry/run_002/project.psx")
    assert chunk.as_posix().endswith("/products/photogrammetry/run_002/chunks/chunk_000")


def test_frames_run_layout(tmp_path):
    lo = L(tmp_path)
    seg = lo.frames_run_dir(7, 3)
    assert seg.as_posix().endswith("/jobs/job_007/frames/run_003/segments")


# --------------------------------------------------------------------------
# run allocation
# --------------------------------------------------------------------------
def test_next_run_id_on_empty(tmp_path):
    lo = L(tmp_path)
    parent = lo.product_type_dir(1, "nav_2d", create=True)
    assert lo.next_run_id(parent) == 1
    assert lo.latest_run_id(parent) is None


def test_next_run_id_increments(tmp_path):
    lo = L(tmp_path)
    parent = lo.product_type_dir(1, "nav_2d")
    (parent / "run_001").mkdir(parents=True)
    (parent / "run_002").mkdir()
    (parent / "not_a_run").mkdir()               # ignored
    assert lo.next_run_id(parent) == 3
    assert lo.latest_run_id(parent) == 2


# --------------------------------------------------------------------------
# relative-path portability
# --------------------------------------------------------------------------
def test_relative_and_resolve_roundtrip(tmp_path):
    lo = L(tmp_path)
    p = lo.product_run_dir(2, "nav_2d", 1, create=True)
    rel = lo.relative(p)
    assert not rel.startswith("/")               # stored relative
    assert lo.resolve(rel).resolve() == p.resolve()


def test_relative_survives_workspace_rename(tmp_path):
    """A stored relative path must resolve correctly after the bundle is renamed."""
    lo = L(tmp_path)
    p = lo.product_run_dir(2, "nav_2d", 1, create=True)
    rel = lo.relative(p)
    # simulate moving/renaming the whole .eprproj bundle
    moved_root = tmp_path / "renamed.eprproj"
    lo.root.rename(moved_root)
    lo2 = WorkspaceLayout(moved_root)
    assert lo2.resolve(rel).is_dir()             # still valid


def test_relative_for_outside_path_is_passthrough(tmp_path):
    lo = L(tmp_path)
    outside = tmp_path / "elsewhere" / "x.csv"
    assert lo.relative(outside) == outside.as_posix()


def test_create_flag_makes_dirs(tmp_path):
    lo = L(tmp_path)
    assert not lo.job_dir(1).exists()
    made = lo.job_dir(1, create=True)
    assert made.is_dir()
