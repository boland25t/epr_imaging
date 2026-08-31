"""Tests for product_browser discovery — the Qt-free half.

Pins the guarantees the browser depends on: registry-recorded outputs surface
with their provenance; bare files on disk still surface (pre-registry / hand-made);
frames and other input noise are never surfaced as products; a file recorded in
the registry is not also duplicated by the filesystem scan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import product_browser as pb


def _write_registry(ws: Path, runs: list, manifests: dict):
    (ws / "runs").mkdir(parents=True, exist_ok=True)
    (ws / "runs" / "registry.json").write_text(json.dumps({"runs": runs}))
    for mp, man in manifests.items():
        Path(mp).parent.mkdir(parents=True, exist_ok=True)
        Path(mp).write_text(json.dumps(man))


def test_registry_products_carry_provenance(tmp_path):
    out = tmp_path / "outputs" / "rasters" / "co2" / "run_001" / "co2.tif"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"TIFF" * 100)
    mp = out.parent / "run.json"
    _write_registry(
        tmp_path,
        runs=[{"run_id": "r_1", "signature": "s", "task_type": "sensor_2d",
               "scope_id": "job_2", "channel": "co2", "status": "completed",
               "finished_at": "2026-08-31T10:00:00", "engine": "colmap 4.1",
               "duration_s": 12.0, "manifest_path": str(mp), "superseded_by": None}],
        manifests={str(mp): {"outputs": [{"path": str(out), "size": 400,
                                          "sha256": "sha256:x"}]}},
    )
    items = pb.discover_products(tmp_path)
    assert len(items) == 1
    it = items[0]
    assert it.source == "registry"
    assert it.task_type == "sensor_2d" and it.channel == "co2"
    assert it.engine == "colmap 4.1" and it.kind == "raster"
    assert it.scope_label == "Job 2"


def test_dem_and_ortho_kind_from_task_type(tmp_path):
    for task, name in (("dem", "dem.tif"), ("orthomosaic", "ortho.tif")):
        out = tmp_path / task / "run.json_out" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"T")
        mp = out.parent / "run.json"
        (tmp_path / "runs").mkdir(exist_ok=True)
        # append per-iteration
        reg = tmp_path / "runs" / "registry.json"
        runs = json.loads(reg.read_text())["runs"] if reg.is_file() else []
        runs.append({"run_id": f"r_{task}", "task_type": task, "scope_id": "survey",
                     "status": "completed", "manifest_path": str(mp),
                     "finished_at": "2026-08-31T10:00:00"})
        reg.write_text(json.dumps({"runs": runs}))
        mp.write_text(json.dumps({"outputs": [{"path": str(out), "size": 1}]}))
    kinds = {it.name: it.kind for it in pb.discover_products(tmp_path)}
    assert kinds["dem.tif"] == "dem"
    assert kinds["ortho.tif"] == "orthomosaic"


def test_scan_surfaces_bare_files(tmp_path):
    # No registry at all — a hand-made product must still appear.
    p = tmp_path / "outputs" / "mesh.obj"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"v 0 0 0\n")
    items = pb.discover_products(tmp_path)
    names = {it.name for it in items}
    assert "mesh.obj" in names
    assert next(it for it in items if it.name == "mesh.obj").source == "scan"


def test_scan_never_surfaces_frames_or_jpg(tmp_path):
    frames = tmp_path / "jobs" / "job_001" / "frames" / "run_001" / "segments"
    frames.mkdir(parents=True)
    for i in range(5):
        (frames / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8")
    # a real product alongside
    (tmp_path / "jobs" / "job_001" / "products").mkdir(parents=True)
    (tmp_path / "jobs" / "job_001" / "products" / "ortho.tif").write_bytes(b"T")
    items = pb.discover_products(tmp_path)
    names = {it.name for it in items}
    assert "ortho.tif" in names
    assert not any(n.endswith(".jpg") for n in names)   # frames excluded entirely


def test_registry_output_not_duplicated_by_scan(tmp_path):
    out = tmp_path / "outputs" / "co2.tif"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"TIFF")
    mp = tmp_path / "outputs" / "run.json"
    _write_registry(
        tmp_path,
        runs=[{"run_id": "r_1", "task_type": "sensor_2d", "scope_id": "job_2",
               "status": "completed", "manifest_path": str(mp),
               "finished_at": "2026-08-31T10:00:00"}],
        manifests={str(mp): {"outputs": [{"path": str(out), "size": 4}]}},
    )
    items = [it for it in pb.discover_products(tmp_path) if it.name == "co2.tif"]
    assert len(items) == 1                     # registry wins; scan doesn't dupe it
    assert items[0].source == "registry"


def test_failed_run_outputs_not_shown(tmp_path):
    out = tmp_path / "outputs" / "partial.tif"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"T")
    mp = tmp_path / "outputs" / "run.json"
    _write_registry(
        tmp_path,
        runs=[{"run_id": "r_1", "task_type": "sensor_2d", "scope_id": "job_2",
               "status": "failed", "manifest_path": str(mp),
               "finished_at": "2026-08-31T10:00:00"}],
        manifests={str(mp): {"outputs": [{"path": str(out), "size": 1}]}},
    )
    # failed run contributes nothing via registry, but the file exists so the
    # scan surfaces it as a bare product (source=scan, no task metadata).
    items = pb.discover_products(tmp_path)
    assert len(items) == 1 and items[0].source == "scan"
    assert items[0].task_type is None


def test_missing_workspace_returns_empty(tmp_path):
    assert pb.discover_products(tmp_path / "nope") == []


def test_grouping(tmp_path):
    items = [
        pb.ProductItem(path="/a/co2.tif", kind="raster", scope_id="job_2"),
        pb.ProductItem(path="/a/dem.tif", kind="dem", scope_id="job_2"),
        pb.ProductItem(path="/a/ortho.tif", kind="orthomosaic", scope_id="survey"),
    ]
    tree = pb.group_products(items)
    assert set(tree["Job 2"].keys()) == {"raster", "dem"}
    assert set(tree["Survey"].keys()) == {"orthomosaic"}


def test_run_history_all_statuses_newest_first(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "registry.json").write_text(json.dumps({"runs": [
        {"run_id": "r_a", "task_type": "sensor_2d", "scope_id": "job_2",
         "status": "completed", "finished_at": "2026-08-31T09:00:00",
         "superseded_by": "r_c"},
        {"run_id": "r_b", "task_type": "photogrammetry", "scope_id": "job_2",
         "status": "failed", "finished_at": "2026-08-31T10:00:00"},
        {"run_id": "r_c", "task_type": "sensor_2d", "scope_id": "job_2",
         "status": "completed", "finished_at": "2026-08-31T11:00:00"},
    ]}))
    hist = pb.read_run_history(tmp_path)
    assert [h["run_id"] for h in hist] == ["r_c", "r_b", "r_a"]   # newest first
    assert {h["status"] for h in hist} == {"completed", "failed"}  # failures kept
    assert hist[-1]["superseded_by"] == "r_c"                      # supersession kept


def test_run_history_empty_without_registry(tmp_path):
    assert pb.read_run_history(tmp_path) == []


def test_scan_excludes_png_slices_and_frame_dirs(tmp_path):
    # Archived PNG depth-slices and annotated/clahe frame images must not appear.
    for sub in ("sensor_3d/co2/run_001/slices/run_001", "sensor_slices/co2/run_001",
                "sampling_1_job_002/segment_001_x_y/frames_annotated"):
        d = tmp_path / "outputs" / sub
        d.mkdir(parents=True)
        (d / "slice.png").write_bytes(b"\x89PNG")
    # a real GeoTIFF product alongside
    (tmp_path / "outputs" / "sensor_2d").mkdir(parents=True)
    (tmp_path / "outputs" / "sensor_2d" / "co2.tif").write_bytes(b"II*\0")
    items = pb.discover_products(tmp_path)
    names = {it.name for it in items}
    assert "co2.tif" in names
    assert not any(n.endswith(".png") for n in names)     # no slice/frame PNGs


def test_scan_attributes_legacy_job_scope(tmp_path):
    d = tmp_path / "job_023_FullTest1" / "outputs" / "sensor_2d"
    d.mkdir(parents=True)
    (d / "co2.tif").write_bytes(b"II*\0")
    items = pb.discover_products(tmp_path)
    it = next(i for i in items if i.name == "co2.tif")
    assert it.scope_id == "job_023" and it.scope_label == "Job 023"
