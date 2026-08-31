"""Tests for PathResolver — the seam between the writers and the two layouts.

The safety-critical property: in LEGACY mode every resolved path equals the exact
string the app builds today, so routing an existing workspace through the resolver
cannot move anyone's data.  In BUNDLE mode the same calls yield the layout.py tree.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workspace_paths import PathResolver, is_bundle, default_bundle_dir


@dataclass
class FakeJob:
    job_id: int
    name: str = ""


# --------------------------------------------------------------------------
# mode detection
# --------------------------------------------------------------------------
def test_plain_dir_is_legacy(tmp_path):
    assert is_bundle(tmp_path) is False
    assert PathResolver(tmp_path).bundle is False


def test_suffix_is_bundle(tmp_path):
    b = tmp_path / "dive.eprproj"
    b.mkdir()
    assert is_bundle(b) is True


def test_project_json_marks_bundle(tmp_path):
    (tmp_path / "project.json").write_text("{}")
    assert is_bundle(tmp_path) is True


# --------------------------------------------------------------------------
# LEGACY mode reproduces today's exact paths
# --------------------------------------------------------------------------
def test_legacy_interp_and_survey(tmp_path):
    r = PathResolver(tmp_path)
    assert r.interp_full() == tmp_path / "interp_full.csv"
    assert r.survey_products() == tmp_path / "outputs"


def test_legacy_job_paths_match_old_formula(tmp_path):
    r = PathResolver(tmp_path)
    job = FakeJob(2, "Down Dive")
    # MainWindow._job_output_dirname: job_002_Down_Dive (spaces→_, \w-preserving)
    assert r.job_dir(job) == tmp_path / "job_002_Down_Dive"
    assert r.job_products(job) == tmp_path / "job_002_Down_Dive" / "outputs"
    assert r.job_filtered_interp(job) == tmp_path / "job_002_Down_Dive" / "filtered_interp.csv"


def test_legacy_job_without_name(tmp_path):
    r = PathResolver(tmp_path)
    assert r.job_dir(FakeJob(7)) == tmp_path / "job_007"


def test_legacy_sampling_dir(tmp_path):
    r = PathResolver(tmp_path)
    assert r.sampling_dir(5, FakeJob(2, "Down Dive")) == tmp_path / "sampling_5_job_002_Down_Dive"
    assert r.sampling_dir(9, None) == tmp_path / "sampling_9_full"


def test_legacy_anomaly(tmp_path):
    assert PathResolver(tmp_path).anomaly_dir() == tmp_path / "anomaly_site_catalog"


def test_products_for_dispatches_scope(tmp_path):
    r = PathResolver(tmp_path)
    assert r.products_for(None) == r.survey_products()
    assert r.products_for(FakeJob(3, "x")) == r.job_products(FakeJob(3, "x"))


# --------------------------------------------------------------------------
# BUNDLE mode yields the layout.py tree
# --------------------------------------------------------------------------
def test_bundle_interp_and_survey(tmp_path):
    b = tmp_path / "dive.eprproj"; b.mkdir()
    r = PathResolver(b)
    assert r.interp_full() == b / "inputs" / "interp_full.csv"
    assert r.survey_products() == b / "survey"


def test_bundle_job_products_use_id_not_name(tmp_path):
    b = tmp_path / "dive.eprproj"; b.mkdir()
    r = PathResolver(b)
    job = FakeJob(2, "Down Dive")
    assert r.job_products(job) == b / "jobs" / "job_002" / "products"     # no name in path
    assert r.job_filtered_interp(job) == b / "jobs" / "job_002" / "filtered_interp.csv"


def test_bundle_sampling_under_job_frames(tmp_path):
    b = tmp_path / "dive.eprproj"; b.mkdir()
    r = PathResolver(b)
    p = r.sampling_dir(5, FakeJob(2, "Down Dive"))
    assert p == b / "jobs" / "job_002" / "frames" / "run_001" / "segments"
    assert " " not in str(p)


def test_bundle_anomaly(tmp_path):
    b = tmp_path / "dive.eprproj"; b.mkdir()
    assert PathResolver(b).anomaly_dir() == b / "survey" / "anomaly"


# --------------------------------------------------------------------------
# new-bundle naming
# --------------------------------------------------------------------------
def test_default_bundle_dir_slugifies(tmp_path):
    p = default_bundle_dir("Down Data w Orientation", tmp_path)
    assert p == tmp_path / "down_data_w_orientation.eprproj"
