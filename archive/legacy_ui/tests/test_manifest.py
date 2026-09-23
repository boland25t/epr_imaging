"""Tests for manifest.py — run provenance + dedup.

Pins the properties the whole dedup system depends on: identical intent →
identical signature; volatile paths ignored; changed data → changed signature;
the registry answers "already done?" only for completed runs whose outputs still
exist; and re-runs supersede their predecessors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import manifest as m
from manifest import Registry, RunRecord, run_signature


# --------------------------------------------------------------------------
# signature stability & sensitivity
# --------------------------------------------------------------------------
def test_signature_is_deterministic():
    s = dict(cell_size=1.0, aggregation="mean")
    a = run_signature("sensor_2d", "job_2", s, "sha256:abc", "colmap 4.1", "co2")
    b = run_signature("sensor_2d", "job_2", dict(aggregation="mean", cell_size=1.0),
                      "sha256:abc", "colmap 4.1", "co2")
    assert a == b                       # key order in settings must not matter


def test_signature_ignores_volatile_paths():
    base = dict(cell_size=1.0)
    a = run_signature("sensor_2d", "job_2", {**base, "output_dir": "/ws/a", "interp_path": "/ws/x.csv"},
                      "sha256:abc", "e")
    b = run_signature("sensor_2d", "job_2", {**base, "output_dir": "/other/b", "interp_path": "/other/y.csv"},
                      "sha256:abc", "e")
    assert a == b                       # location must not change identity


def test_signature_changes_with_data():
    s = dict(cell_size=1.0)
    a = run_signature("sensor_2d", "job_2", s, "sha256:DATA_A", "e")
    b = run_signature("sensor_2d", "job_2", s, "sha256:DATA_B", "e")
    assert a != b                       # edited interp data → different run


def test_signature_changes_with_settings_engine_scope_channel():
    base = ("sensor_2d", "job_2", dict(cell_size=1.0), "sha256:abc", "colmap 4.1", "co2")
    sig = run_signature(*base)
    assert sig != run_signature("sensor_2d", "job_2", dict(cell_size=2.0), "sha256:abc", "colmap 4.1", "co2")
    assert sig != run_signature("sensor_2d", "job_2", dict(cell_size=1.0), "sha256:abc", "metashape 2.3", "co2")
    assert sig != run_signature("sensor_2d", "job_9", dict(cell_size=1.0), "sha256:abc", "colmap 4.1", "co2")
    assert sig != run_signature("sensor_2d", "job_2", dict(cell_size=1.0), "sha256:abc", "colmap 4.1", "ch4")


def test_callables_stripped_from_settings():
    s = dict(cell_size=1.0, log_fn=lambda x: None, status_callback=print)
    assert m.settings_for_signature(s) == {"cell_size": 1.0}


# --------------------------------------------------------------------------
# input fingerprints
# --------------------------------------------------------------------------
def test_file_content_hash_cached(tmp_path):
    f = tmp_path / "interp.csv"
    f.write_text("a,b\n1,2\n")
    cache = tmp_path / "cache.json"
    h1 = m.file_content_hash_cached(f, cache)
    assert cache.is_file()
    h2 = m.file_content_hash_cached(f, cache)     # served from cache
    assert h1 == h2
    f.write_text("a,b\n1,2\n3,4\n")               # content changes
    import os, time
    os.utime(f, (time.time() + 1, time.time() + 1))
    assert m.file_content_hash_cached(f, cache) != h1


def test_frame_set_hash_stable_and_order_independent(tmp_path):
    for n in ("f2.jpg", "f1.jpg", "f3.jpg"):
        (tmp_path / n).write_bytes(b"x" * 10)
    a = m.frame_set_hash([str(tmp_path / "f1.jpg"), str(tmp_path / "f2.jpg"), str(tmp_path / "f3.jpg")])
    b = m.frame_set_hash([str(tmp_path / "f3.jpg"), str(tmp_path / "f1.jpg"), str(tmp_path / "f2.jpg")])
    assert a == b
    (tmp_path / "f4.jpg").write_bytes(b"x" * 10)
    c = m.frame_set_hash([str(tmp_path / n) for n in ("f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg")])
    assert c != a                                 # adding a frame changes it


def test_intervals_fingerprint_order_independent():
    class IV:
        def __init__(self, a, b):
            from datetime import datetime
            self.start_time = datetime.fromtimestamp(a)
            self.end_time = datetime.fromtimestamp(b)
    a = m.intervals_fingerprint([IV(100, 200), IV(300, 400)])
    b = m.intervals_fingerprint([IV(300, 400), IV(100, 200)])
    assert a == b
    assert m.intervals_fingerprint([(100, 200), (300, 400)]) == a   # tuple form matches


# --------------------------------------------------------------------------
# manifest + registry
# --------------------------------------------------------------------------
def make_rec(sig, task="sensor_2d", scope="job_2", channel="co2", status="completed"):
    return RunRecord(run_id=m.new_run_id(sig), signature=sig, task_type=task,
                     scope_id=scope, channel=channel, status=status,
                     finished_at="2026-08-31T11:00:00")


def test_write_manifest_and_describe_outputs(tmp_path):
    out = tmp_path / "prod.tif"
    out.write_bytes(b"y" * 2048)
    rec = make_rec("sha256:sig1")
    rec.outputs = m.describe_outputs([str(out)])
    assert rec.outputs[0]["size"] == 2048 and rec.outputs[0]["sha256"].startswith("sha256:")
    p = m.write_manifest(tmp_path / "run_001", rec)
    assert p.name == "run.json" and p.is_file()


def test_describe_outputs_skips_hash_for_huge_files(tmp_path):
    big = tmp_path / "dense.ply"
    big.write_bytes(b"z" * 4096)
    outs = m.describe_outputs([str(big)], hash_max_bytes=1024)   # below file size
    assert outs[0]["size"] == 4096 and "sha256" not in outs[0]


def test_registry_dedup_lookup(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    rec = make_rec("sha256:sigA")
    mp = m.write_manifest(tmp_path / "run_001", rec)
    reg.record(rec, mp, tmp_path / "run_001")
    reg.save()
    reloaded = Registry(tmp_path / "registry.json")
    hit = reloaded.lookup("sha256:sigA")
    assert hit and hit["run_id"] == rec.run_id
    assert reloaded.lookup("sha256:nope") is None


def test_failed_run_not_in_by_signature(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    rec = make_rec("sha256:failed", status="failed")
    reg.record(rec, tmp_path / "run.json", tmp_path)
    assert reg.lookup("sha256:failed") is None    # a failure must not suppress a retry


def test_rerun_supersedes_prior(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    r1 = make_rec("sha256:v1")
    reg.record(r1, tmp_path / "run_001" / "run.json", tmp_path / "run_001")
    r2 = make_rec("sha256:v2")                     # same group, changed inputs
    reg.record(r2, tmp_path / "run_002" / "run.json", tmp_path / "run_002")
    e1 = next(e for e in reg.runs if e["run_id"] == r1.run_id)
    assert e1["superseded_by"] == r2.run_id        # meaningful versioning, not v3/v5/v6


def test_outputs_exist_gate(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    out = tmp_path / "run_001" / "prod.tif"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"w" * 512)
    rec = make_rec("sha256:sigX")
    rec.outputs = m.describe_outputs([str(out)])
    mp = m.write_manifest(tmp_path / "run_001", rec)
    entry = reg.record(rec, mp, tmp_path / "run_001")
    assert reg.outputs_exist(entry) is True
    out.unlink()                                   # user deleted the product
    assert reg.outputs_exist(entry) is False       # → must re-run
