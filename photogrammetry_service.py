"""
photogrammetry_service.py — Photogrammetry pipeline service

Supports two engines:
  - Metashape Professional (Agisoft) via Python API  (import Metashape)
  - COLMAP via CLI subprocess

Both engines share the same call interface used by PhotogrammetryWorker in
main_window.py.  All long-running methods accept a log_fn callable so the
caller can route progress text to a QTextEdit or similar widget.

Output directory layout (under outputs/photogrammetry/{job_id}/run_NNN/):
  meta.json          — run settings and product paths
  sparse_cloud.ply   — SfM sparse point cloud
  dense_cloud.ply    — MVS dense point cloud (if requested)
  mesh.obj           — triangulated mesh (if requested)
  mesh.mtl           — material file for mesh
  texture.png        — texture image (if requested)
  cameras.json       — per-camera poses in nav (UTM) coordinate frame
  report.pdf         — Metashape processing report (Metashape engine only)
  project.psx        — Metashape project file (Metashape engine only)
  colmap/            — COLMAP workspace (COLMAP engine only)
    database.db
    sparse/
    dense/
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Quality preset mappings
# ---------------------------------------------------------------------------

# Metashape integer constants (stable across versions):
#   downscale:  0=Ultra, 1=High, 2=Medium, 4=Low, 8=Lowest
#   filter:     0=Disabled, 1=Aggressive, 2=Moderate, 3=Mild
_META_ALIGN_ACC: dict[str, int] = {
    "highest": 0, "high": 1, "medium": 2, "low": 4, "lowest": 8,
}
_META_DENSE_QUAL: dict[str, int] = {
    "ultra": 0, "high": 1, "medium": 2, "low": 4, "lowest": 8,
}
_META_DEPTH_FILTER: dict[str, int] = {
    "disabled": 0, "aggressive": 1, "moderate": 2, "mild": 3,
}
# These are looked up via getattr(Metashape, name) at runtime
_META_SURFACE_TYPE: dict[str, str] = {
    "arbitrary": "Arbitrary", "height field": "HeightField",
}
_META_FACE_COUNT: dict[str, str] = {
    "low": "LowFaceCount", "medium": "MediumFaceCount", "high": "HighFaceCount",
}
_META_BLENDING: dict[str, str] = {
    "mosaic": "MosaicBlending", "average": "AverageBlending",
    "min": "MinBlending", "max": "MaxBlending", "disabled": "DisabledBlending",
}

# COLMAP matcher CLI command names
_COLMAP_MATCHER_CMD: dict[str, str] = {
    "exhaustive": "exhaustive_matcher",
    "sequential": "sequential_matcher",
    "vocab tree": "vocab_tree_matcher",
    "spatial":    "spatial_matcher",
}


# ---------------------------------------------------------------------------
# Metashape API version handling
# ---------------------------------------------------------------------------
#
# Metashape 2.0 renamed the dense-cloud API.  The same identifier even changes
# meaning between majors, so we branch on the version rather than probing
# attribute names:
#
#                       sparse / tie points        dense cloud
#   Metashape 1.x       PointCloudData             DenseCloudData
#   Metashape 2.x       TiePointsData              PointCloudData   (!)
#
#   build method        1.x: chunk.buildDenseCloud()
#                       2.x: chunk.buildPointCloud()
#
# Because PointCloudData means "sparse" in 1.x but "dense" in 2.x, naive
# getattr-fallback probing would silently export the wrong cloud.  _meta_api()
# resolves the correct bindings for the running version.

def _metashape_major(Metashape) -> int:
    """Return the Metashape major version (e.g. 2), defaulting to 2 if unknown."""
    for getter in (lambda: Metashape.app.version, lambda: Metashape.version):
        try:
            return int(str(getter()).split(".")[0])
        except Exception:
            continue
    return 2  # assume modern API when the version can't be read


def _meta_api(Metashape) -> dict:
    """Resolve version-appropriate dense/sparse bindings.

    Returns a dict with:
        build_dense    — callable(chunk) that builds the dense/point cloud
        dense_source   — DataSource enum for the dense cloud
        sparse_source  — DataSource enum for the sparse / tie-point cloud
    """
    ds = Metashape.DataSource
    if _metashape_major(Metashape) >= 2:
        return {
            "build_dense":   lambda chunk: chunk.buildPointCloud(),
            "dense_source":  ds.PointCloudData,
            # TiePointsData in 2.x; fall back to PointCloudData on the rare
            # pre-release 2.0 build that hadn't renamed it yet.
            "sparse_source": getattr(ds, "TiePointsData", getattr(ds, "PointCloudData", None)),
        }
    return {
        "build_dense":   lambda chunk: chunk.buildDenseCloud(),
        "dense_source":  ds.DenseCloudData,
        "sparse_source": ds.PointCloudData,
    }


# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------

def _is_wsl() -> bool:
    """True when running inside Windows Subsystem for Linux."""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def detect_engines() -> dict[str, str | None]:
    """Return paths to available photogrammetry engines.

    Returns a dict with keys "metashape" and "colmap"; each value is the
    path/command string if found, or None if not available.
    """
    return {
        "metashape": _detect_metashape(),
        "colmap":    _detect_colmap(),
    }


def metashape_driver() -> str | None:
    """How Metashape can be driven here: "inprocess", "subprocess", or None.

    "inprocess"  — ``import Metashape`` works in this interpreter.
    "subprocess" — the module can't be imported (e.g. the app runs under WSL)
                   but a Windows metashape.exe is present, so the batch pipeline
                   drives it via ``metashape.exe -r metashape_worker.py``.
    None         — no usable Metashape at all.
    """
    try:
        import Metashape  # noqa: F401
        return "inprocess"
    except ImportError:
        pass
    return "subprocess" if _find_windows_metashape_exe() else None


def _detect_metashape() -> str | None:
    """Return a Metashape executable path when the engine is usable by ANY means.

    Usable means either the Python module is importable in this interpreter, or
    (the common WSL case) a Windows metashape.exe exists and the batch pipeline
    can drive it as a subprocess.  See metashape_driver().
    """
    driver = metashape_driver()
    if driver == "inprocess":
        return _find_metashape_exe()
    if driver == "subprocess":
        return _find_windows_metashape_exe()
    return None


def metashape_unavailable_reason() -> str | None:
    """Explain why the Metashape engine is unavailable, or None if it's usable.

    Returns None whenever Metashape can be driven either in-process OR via the
    Windows-exe subprocess path (which works from WSL).  Only returns a message
    when neither is possible.
    """
    if metashape_driver() is not None:
        return None
    return (
        "No Metashape found. Install Agisoft Metashape Professional (the app can "
        "drive a Windows install from WSL via metashape.exe), or use the COLMAP "
        "engine instead."
    )


def _find_windows_metashape_exe() -> str | None:
    """Locate a Windows Metashape.exe, including via /mnt/c when under WSL.

    Agisoft has shipped the folder under several names over the years
    ("Metashape Professional", "Metashape Pro", "Metashape").  Rather than
    hardcode one, glob every "Agisoft\\*" directory under each Program Files
    root and look for metashape.exe (case-insensitive) inside it.  This is what
    fixed the WSL install here, which lives in "Agisoft\\Metashape Pro".
    """
    program_dirs = [
        r"C:\Program Files",
        r"C:\Program Files (x86)",
    ]
    roots = [Path(p) for p in program_dirs]
    if _is_wsl():
        roots += [Path("/mnt/c/Program Files"), Path("/mnt/c/Program Files (x86)")]

    for root in roots:
        agisoft = root / "Agisoft"
        if not agisoft.is_dir():
            continue
        for sub in sorted(agisoft.iterdir()):
            if not sub.is_dir():
                continue
            for name in ("metashape.exe", "Metashape.exe"):
                exe = sub / name
                if exe.exists():
                    return str(exe)
    return None


def _find_metashape_exe() -> str:
    """Best-effort search for the Metashape GUI executable."""
    candidates = []
    if platform.system() == "Windows":
        win = _find_windows_metashape_exe()
        if win:
            candidates.append(win)
    elif platform.system() == "Darwin":
        candidates.append("/Applications/Metashape Professional.app/Contents/MacOS/Metashape")
    else:
        # Linux / WSL: prefer a native Linux install; fall back to the Windows
        # exe under /mnt/c so "Open in Metashape GUI" still works from WSL.
        candidates.extend([
            "/opt/metashape-pro/metashape.sh",
            "/usr/local/bin/metashape",
            shutil.which("metashape") or "",
        ])
        win = _find_windows_metashape_exe()
        if win:
            candidates.append(win)
    for c in candidates:
        if c and os.path.exists(c):
            return c
    found = shutil.which("metashape") or shutil.which("Metashape")
    return found or "metashape"  # best-guess command even if not verified


def _detect_colmap() -> str | None:
    """Return "colmap" command string if COLMAP is on PATH, else None."""
    return shutil.which("colmap")


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

def prepare_run_dir(output_root: str, job_id: int) -> Path:
    """Create and return the next sequential run_NNN directory.

    output_root is the photogrammetry root (callers pass
    <outputs>/photogrammetry).  The layout produced is:
        <output_root>/job_{job_id:03d}/run_NNN
    i.e. <outputs>/photogrammetry/job_NNN/run_NNN.
    """
    base = Path(output_root) / f"job_{job_id:03d}"
    base.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        candidate = base / f"run_{idx:03d}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        idx += 1


def save_meta(run_dir: Path, data: dict) -> None:
    with open(run_dir / "meta.json", "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Detailed-logging helpers (shared by both engines)
# ---------------------------------------------------------------------------

def _fsize(path: str | Path) -> str:
    """Human-readable file size for log lines ('' when the file is missing)."""
    try:
        n = os.path.getsize(str(path))
    except OSError:
        return "size unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


@contextmanager
def _stage_timer(log: Callable[[str], None], name: str):
    """Log the start of a processing stage and its elapsed wall time on exit.

    The elapsed line is emitted even when the stage raises, so the log always
    shows how long a failing stage ran before it died.
    """
    t0 = time.time()
    log(f"      ▶ {name}…")
    try:
        yield
    except Exception:
        log(f"      ✗ {name} FAILED after {time.time() - t0:.1f} s")
        raise
    log(f"      ✓ {name} done in {time.time() - t0:.1f} s")


def _progress_logger(log_fn: Callable[[str], None],
                     file_fn: Optional[Callable[[str], None]],
                     stage: str) -> Callable[[float], None]:
    """Return a Metashape-style progress callback (float percent).

    GUI log (log_fn) gets a line every ≥5%; the uncapped task log file
    (file_fn) gets every ≥1% tick.  Never raises — a broken progress line must
    not abort a multi-hour reconstruction.
    """
    state = {"gui": -5.0, "file": -1.0}

    def cb(pct: float) -> None:
        try:
            pct = float(pct)
            if file_fn is not None and (pct - state["file"] >= 1.0 or pct >= 100.0):
                state["file"] = pct
                file_fn(f"        [{stage}] {pct:.1f}%")
            if pct - state["gui"] >= 5.0 or pct >= 100.0:
                state["gui"] = pct
                log_fn(f"        [{stage}] {pct:.0f}%")
        except Exception:  # noqa: BLE001
            pass

    return cb


def _call_with_progress(fn: Callable, kwargs: dict,
                        progress_cb: Optional[Callable[[float], None]]):
    """Invoke a Metashape API call, attaching a progress callback when supported.

    Older API builds reject the ``progress`` keyword with TypeError — retry
    without it rather than failing the stage.
    """
    if progress_cb is not None:
        try:
            return fn(**kwargs, progress=progress_cb)
        except TypeError:
            pass
    return fn(**kwargs)


def build_chunk_sets(frame_dirs: list[str], chunk_size: int) -> list[dict]:
    """Split interval frame dirs into Metashape chunk specs of ≤ chunk_size images.

    Chunks never span interval (frame-dir) boundaries: each frame dir's sorted
    photo list is cut into sequential groups of at most chunk_size.  A
    chunk_size of 0 (or less) means unlimited — one chunk per frame dir.

    Returns a list of dicts:
        {"frame_dir": str, "photos": [str], "interval_idx": int (1-based),
         "part_idx": int (1-based), "n_parts": int, "label": str}
    """
    sets: list[dict] = []
    for i_idx, frame_dir in enumerate(frame_dirs, start=1):
        photos = _collect_frames(frame_dir)
        if not photos:
            sets.append({
                "frame_dir": frame_dir, "photos": [], "interval_idx": i_idx,
                "part_idx": 1, "n_parts": 1,
                "label": f"interval{i_idx:02d}",
            })
            continue
        size = chunk_size if chunk_size and chunk_size > 0 else len(photos)
        groups = [photos[k:k + size] for k in range(0, len(photos), size)]
        for p_idx, group in enumerate(groups, start=1):
            label = (f"interval{i_idx:02d}_part{p_idx:02d}"
                     if len(groups) > 1 else f"interval{i_idx:02d}")
            sets.append({
                "frame_dir": frame_dir, "photos": group, "interval_idx": i_idx,
                "part_idx": p_idx, "n_parts": len(groups), "label": label,
            })
    return sets


# ---------------------------------------------------------------------------
# Metashape engine
# ---------------------------------------------------------------------------

def run_metashape(
    run_dir: Path,
    frame_dir: str,
    # Alignment
    align_accuracy: str = "High",
    key_point_limit: int = 40000,
    tie_point_limit: int = 10000,
    generic_preselect: bool = True,
    reference_preselect: bool = True,
    adaptive_fitting: bool = True,
    reset_cameras: bool = False,
    # Dense cloud
    build_dense: bool = True,
    dense_quality: str = "Medium",
    depth_filter: str = "Moderate",
    reuse_depth: bool = False,
    # Mesh
    build_mesh: bool = False,
    mesh_surface: str = "Arbitrary",
    mesh_faces: str = "Medium",
    mesh_source: str = "Dense cloud",
    mesh_vertex_colors: bool = True,
    # Texture
    build_texture: bool = False,
    texture_size: int = 4096,
    texture_blending: str = "Mosaic",
    texture_fill_holes: bool = True,
    # Export
    export_dense_ply: bool = False,   # ARCHIVED deliverable; dense build stays on
    export_mesh_obj: bool = False,
    # Georeference
    nav_csv: Optional[str] = None,
    use_nav_reference: bool = True,
    nav_accuracy_h: float = 0.1,
    nav_accuracy_v: float = 0.5,
    # Project
    save_project: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, str]:
    """Run the full Metashape headless pipeline with fine-grained control.

    All string parameters are case-insensitive and match the labels shown in
    the TaskConfigDialog (e.g. align_accuracy="High", depth_filter="Moderate").

    Returns a dict mapping product keys to absolute file paths.
    """
    import Metashape

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    # Version-appropriate dense/sparse bindings (1.x vs 2.x — see _meta_api).
    api   = _meta_api(Metashape)
    major = _metashape_major(Metashape)

    # Point limits: 0 in the UI means "Auto" — translate to Metashape's own
    # documented defaults rather than passing 0, which Metashape reads as
    # "unlimited" (very slow, not what the user intends).
    if key_point_limit <= 0:
        key_point_limit = 40000
    if tie_point_limit <= 0:
        tie_point_limit = 4000

    photos = _collect_frames(frame_dir)
    if not photos:
        raise FileNotFoundError(f"No images found in {frame_dir}")

    log(f"Metashape {major}.x: {len(photos)} frames from {frame_dir}")

    psx_path = str(run_dir / "project.psx")
    doc = Metashape.Document()
    doc.save(psx_path)
    chunk = doc.addChunk()
    chunk.addPhotos(photos)
    if save_project:
        doc.save()
    log(f"Project created: {psx_path}")

    opts = dict(
        align_accuracy=align_accuracy,
        key_point_limit=key_point_limit,
        tie_point_limit=tie_point_limit,
        generic_preselect=generic_preselect,
        reference_preselect=reference_preselect,
        adaptive_fitting=adaptive_fitting,
        reset_cameras=reset_cameras,
        build_dense=build_dense,
        dense_quality=dense_quality,
        depth_filter=depth_filter,
        reuse_depth=reuse_depth,
        build_mesh=build_mesh,
        mesh_surface=mesh_surface,
        mesh_faces=mesh_faces,
        mesh_source=mesh_source,
        mesh_vertex_colors=mesh_vertex_colors,
        build_texture=build_texture,
        texture_size=texture_size,
        texture_blending=texture_blending,
        texture_fill_holes=texture_fill_holes,
        export_dense_ply=export_dense_ply,
        export_mesh_obj=export_mesh_obj,
        nav_csv=nav_csv,
        use_nav_reference=use_nav_reference,
        nav_accuracy_h=nav_accuracy_h,
        nav_accuracy_v=nav_accuracy_v,
    )
    products = _process_metashape_chunk(
        Metashape, doc, chunk, run_dir, api=api, major=major,
        opts=opts, save_project=save_project, log=log,
    )
    products["metashape_psx"] = psx_path
    if save_project:
        doc.save()
    log("Metashape run complete.")
    return products


def _process_metashape_chunk(Metashape, doc, chunk, run_dir, *, api, major,
                             opts, save_project, log, file_log=None):
    """Process ONE Metashape chunk: georeference seed -> align -> dense ->
    mesh -> texture -> exports.  Returns the products dict (without the .psx,
    which the caller adds).  Shared by run_metashape and run_metashape_batch.

    file_log, when given, receives fine-grained progress ticks (every ≥1%)
    destined for the uncapped task log file."""
    align_accuracy = opts['align_accuracy']
    key_point_limit = opts['key_point_limit']
    tie_point_limit = opts['tie_point_limit']
    generic_preselect = opts['generic_preselect']
    reference_preselect = opts['reference_preselect']
    adaptive_fitting = opts['adaptive_fitting']
    reset_cameras = opts['reset_cameras']
    build_dense = opts['build_dense']
    dense_quality = opts['dense_quality']
    depth_filter = opts['depth_filter']
    reuse_depth = opts['reuse_depth']
    build_mesh = opts['build_mesh']
    mesh_surface = opts['mesh_surface']
    mesh_faces = opts['mesh_faces']
    mesh_source = opts['mesh_source']
    mesh_vertex_colors = opts['mesh_vertex_colors']
    build_texture = opts['build_texture']
    texture_size = opts['texture_size']
    texture_blending = opts['texture_blending']
    texture_fill_holes = opts['texture_fill_holes']
    export_dense_ply = opts['export_dense_ply']
    export_mesh_obj = opts['export_mesh_obj']
    nav_csv = opts['nav_csv']
    use_nav_reference = opts['use_nav_reference']
    nav_accuracy_h = opts['nav_accuracy_h']
    nav_accuracy_v = opts['nav_accuracy_v']
    dense_source = api['dense_source']
    # ── Georeference: pre-seed camera positions ────────────────────────────────
    # seeded_nav is only True when at least one camera actually received a
    # reference location — passing reference_preselection with zero references
    # would silently degrade matching.
    seeded_nav = False
    if nav_csv and use_nav_reference and Path(nav_csv).exists():
        n_seeded = _seed_camera_locations(chunk, nav_csv, log, nav_accuracy_h, nav_accuracy_v)
        seeded_nav = n_seeded > 0
    elif use_nav_reference:
        log(f"      nav seeding skipped: nav CSV not found ({nav_csv or 'none set'})")

    # ── Alignment ─────────────────────────────────────────────────────────────
    acc_int = _META_ALIGN_ACC.get(align_accuracy.lower(), 1)
    log(f"Aligning cameras (accuracy={align_accuracy}, downscale={acc_int}, "
        f"keypoints={key_point_limit}, tiepoints={tie_point_limit}, "
        f"generic_preselect={generic_preselect}, "
        f"reference_preselect={bool(seeded_nav and reference_preselect)})…")
    match_kwargs = dict(
        downscale=acc_int,
        keypoint_limit=key_point_limit,
        tiepoint_limit=tie_point_limit,
        generic_preselection=generic_preselect,
        reference_preselection=bool(seeded_nav and reference_preselect),
        reset_matches=reset_cameras,
    )
    # Explicitly request "source" reference preselection (use the seeded
    # coordinates) when the enum is available — preferred over the bare bool in
    # Metashape 2.x.  Guarded so older APIs that lack the enum don't break.
    if seeded_nav and reference_preselect:
        mode_enum = getattr(Metashape, "ReferencePreselectionMode", None)
        if mode_enum is not None and hasattr(mode_enum, "ReferencePreselectionSource"):
            match_kwargs["reference_preselection_mode"] = mode_enum.ReferencePreselectionSource
    with _stage_timer(log, "matchPhotos"):
        _call_with_progress(chunk.matchPhotos, match_kwargs,
                            _progress_logger(log, file_log, "matchPhotos"))
    with _stage_timer(log, "alignCameras"):
        _call_with_progress(chunk.alignCameras, {"adaptive_fitting": adaptive_fitting},
                            _progress_logger(log, file_log, "alignCameras"))
    if save_project:
        doc.save()

    aligned = sum(1 for c in chunk.cameras if c.transform)
    total   = len(chunk.cameras)
    log(f"Alignment: {aligned}/{total} cameras aligned")
    if aligned == 0:
        raise RuntimeError(
            "No cameras aligned. Check image overlap and quality. "
            "Consider opening the project in Metashape GUI for diagnostics."
        )

    products: dict[str, str] = {}

    # ── Sparse cloud export ───────────────────────────────────────────────────
    # Best-effort: the sparse/tie-point cloud is a diagnostic extra, not the
    # primary product, and exporting it via exportPointCloud is unreliable
    # across versions (esp. 2.x tie points).  Never let it abort the run.
    sparse_source = api["sparse_source"]
    if sparse_source is not None:
        try:
            sparse_path = str(run_dir / "sparse_cloud.ply")
            chunk.exportPointCloud(sparse_path, source_data=sparse_source, save_colors=True)
            products["sparse_ply"] = sparse_path
            log(f"Sparse cloud: {sparse_path}")
        except Exception as exc:
            log(f"Sparse cloud export skipped (non-fatal): {exc}")

    # ── Dense cloud ───────────────────────────────────────────────────────────
    dq    = _META_DENSE_QUAL.get(dense_quality.lower(), 2)
    dfilt = _META_DEPTH_FILTER.get(depth_filter.lower(), 2)
    depth_maps_built = False
    if build_dense:
        log(f"Building depth maps (quality={dense_quality}, filter={depth_filter})…")
        with _stage_timer(log, "buildDepthMaps"):
            _call_with_progress(chunk.buildDepthMaps, dict(
                downscale=dq, filter_mode=dfilt, reuse_depth=reuse_depth,
            ), _progress_logger(log, file_log, "buildDepthMaps"))
        depth_maps_built = True
        with _stage_timer(log, f"build {'point' if major >= 2 else 'dense'} cloud"):
            api["build_dense"](chunk)   # buildPointCloud() (2.x) / buildDenseCloud() (1.x)
        if save_project:
            doc.save()

        if export_dense_ply:
            dense_path = str(run_dir / "dense_cloud.ply")
            with _stage_timer(log, "export dense cloud"):
                chunk.exportPointCloud(dense_path, source_data=dense_source, save_colors=True)
            products["dense_ply"] = dense_path
            log(f"Dense cloud: {dense_path}  ({_fsize(dense_path)})")

    # ── Mesh ──────────────────────────────────────────────────────────────────
    # Mesh from "Depth maps" does NOT require the dense-cloud stage — build the
    # depth maps on demand when dense was disabled.  Mesh from "Dense cloud"
    # genuinely needs build_dense on.
    mesh_from_depth = mesh_source.lower() == "depth maps"
    if build_mesh and not build_dense and not mesh_from_depth:
        log("  ⚠ Mesh skipped: mesh source is 'Dense cloud' but the Dense Cloud "
            "stage is disabled. Enable it, or switch the mesh source to 'Depth maps'.")
    if build_mesh and (build_dense or mesh_from_depth):
        surf_attr = _META_SURFACE_TYPE.get(mesh_surface.lower(), "Arbitrary")
        face_attr = _META_FACE_COUNT.get(mesh_faces.lower(), "MediumFaceCount")
        if mesh_from_depth:
            mesh_src_enum = Metashape.DataSource.DepthMapsData
            if not depth_maps_built:
                log(f"Building depth maps for mesh (quality={dense_quality}, filter={depth_filter})…")
                with _stage_timer(log, "buildDepthMaps"):
                    _call_with_progress(chunk.buildDepthMaps, dict(
                        downscale=dq, filter_mode=dfilt, reuse_depth=reuse_depth,
                    ), _progress_logger(log, file_log, "buildDepthMaps"))
                depth_maps_built = True
        else:
            mesh_src_enum = dense_source
        log(f"Building mesh (surface={mesh_surface}, faces={mesh_faces}, source={mesh_source})…")
        with _stage_timer(log, "buildMesh"):
            _call_with_progress(chunk.buildMesh, dict(
                source_data=mesh_src_enum,
                surface_type=getattr(Metashape, surf_attr),
                face_count=getattr(Metashape, face_attr),
                vertex_colors=mesh_vertex_colors,
            ), _progress_logger(log, file_log, "buildMesh"))
        if save_project:
            doc.save()

        if export_mesh_obj:
            mesh_path = str(run_dir / "mesh.obj")
            with _stage_timer(log, "export mesh"):
                chunk.exportModel(mesh_path, save_texture=False)
            products["mesh_obj"] = mesh_path
            log(f"Mesh: {mesh_path}  ({_fsize(mesh_path)})")

        # ── Texture ───────────────────────────────────────────────────────────
        if build_texture:
            blend_attr = _META_BLENDING.get(texture_blending.lower(), "MosaicBlending")
            log(f"Building texture (size={texture_size}, blending={texture_blending})…")
            with _stage_timer(log, "buildUV"):
                chunk.buildUV()
            with _stage_timer(log, "buildTexture"):
                _call_with_progress(chunk.buildTexture, dict(
                    blending_mode=getattr(Metashape, blend_attr),
                    texture_size=texture_size,
                    fill_holes=texture_fill_holes,
                ), _progress_logger(log, file_log, "buildTexture"))
            if save_project:
                doc.save()
            tex_mesh_path = str(run_dir / "mesh_textured.obj")
            with _stage_timer(log, "export textured mesh"):
                chunk.exportModel(tex_mesh_path, save_texture=True)
            products["mesh_textured_obj"] = tex_mesh_path
            products["texture_png"]       = str(run_dir / "mesh_textured.png")
            log(f"Textured mesh: {tex_mesh_path}  ({_fsize(tex_mesh_path)})")

    # ── Camera poses export ───────────────────────────────────────────────────
    cameras_path = str(run_dir / "cameras.json")
    _export_cameras_json(chunk, cameras_path)
    products["cameras_json"] = cameras_path
    log(f"Camera poses: {cameras_path}")

    # ── Processing report ─────────────────────────────────────────────────────
    report_path = str(run_dir / "report.pdf")
    try:
        chunk.exportReport(report_path)
        products["report_pdf"] = report_path
        log(f"Report: {report_path}")
    except Exception:
        pass  # non-fatal

    return products


def _to_windows_path(p) -> str:
    """Translate a path to the form a native Windows Metashape process needs.

    Under WSL, ``wslpath -w`` maps /home/... → \\\\wsl.localhost\\<distro>\\home\\...
    and /mnt/c/... → C:\\...  Off WSL the path is already native.
    """
    import subprocess as _sp
    s = str(p)
    if not _is_wsl():
        return s
    try:
        return _sp.check_output(["wslpath", "-w", s], text=True).strip()
    except Exception:                                          # noqa: BLE001
        # Fallback for /mnt/<drive>/... → <DRIVE>:\...
        if s.startswith("/mnt/") and len(s) > 6 and s[6] == "/":
            return s[5].upper() + ":" + s[6:].replace("/", "\\")
        return s


def _win_temp_dir() -> "Path":
    """A writable directory on the Windows filesystem for the worker script and
    params — Metashape reads its ``-r`` script far more reliably from a real
    Windows path than from a \\\\wsl.localhost UNC path."""
    base = Path("/mnt/c/Users/Public/epr_metashape")
    if _is_wsl() and Path("/mnt/c/Users/Public").is_dir():
        base.mkdir(parents=True, exist_ok=True)
        return base
    tmp = Path(tempfile.gettempdir()) / "epr_metashape"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def _run_metashape_batch_subprocess(exe, project_psx, frame_sets, *,
                                    log_fn=None, file_log_fn=None, **opts):
    """Drive Metashape via ``metashape.exe -r metashape_worker.py params.json``.

    Builds a params file (all paths translated to Windows form), launches the
    worker in Metashape's own interpreter, streams its stdout to the app log,
    then reads back the result JSON.  Returns {run_dir: [product paths]}.
    """
    import subprocess
    import uuid

    def log(msg):
        if log_fn:
            log_fn(msg)
        if file_log_fn:
            file_log_fn(msg)

    worker_src = Path(__file__).resolve().parent / "metashape_worker.py"
    win_dir = _win_temp_dir()
    tag = uuid.uuid4().hex[:8]
    # Copy the worker next to the params so both live on the Windows filesystem.
    worker_dst = win_dir / "metashape_worker.py"
    shutil.copyfile(worker_src, worker_dst)
    params_path = win_dir / f"params_{tag}.json"
    result_path = win_dir / f"result_{tag}.json"
    log_path    = win_dir / f"worker_{tag}.log"

    # Map option kwargs (defaults mirror the in-process signature) into the flat
    # options block the worker understands.
    def o(k, d):
        return opts.get(k, d)

    options = {
        "quality_threshold": float(o("quality_threshold", 0.5)),
        "align_accuracy": o("align_accuracy", "High"),
        "key_point_limit": int(o("key_point_limit", 40000)),
        "tie_point_limit": int(o("tie_point_limit", 10000)),
        "generic_preselect": bool(o("generic_preselect", True)),
        "adaptive_fitting": bool(o("adaptive_fitting", True)),
        "build_dense": bool(o("build_dense", True)),
        "dense_quality": o("dense_quality", "Medium"),
        "depth_filter": o("depth_filter", "Moderate"),
        "export_dense_ply": bool(o("export_dense_ply", False)),
        "build_mesh": bool(o("build_mesh", False)),
        "mesh_surface": o("mesh_surface", "Arbitrary"),
        "mesh_faces": o("mesh_faces", "Medium"),
        "mesh_source": o("mesh_source", "Dense cloud"),
        "mesh_interpolation": o("mesh_interpolation", "Enabled"),
        "ortho_surface": o("ortho_surface", "DEM"),
        "mesh_vertex_colors": bool(o("mesh_vertex_colors", True)),
        "build_texture": bool(o("build_texture", False)),
        "texture_size": int(o("texture_size", 4096)),
        "texture_blending": o("texture_blending", "Mosaic"),
        "texture_fill_holes": bool(o("texture_fill_holes", True)),
        "export_mesh_obj": bool(o("export_mesh_obj", False)),
        # DEM / orthomosaic (new products — off unless asked)
        "build_dem": bool(o("build_dem", o("build_orthomosaic", False))),
        "export_dem": bool(o("export_dem", False)),
        "build_orthomosaic": bool(o("build_orthomosaic", False)),
        # georeference
        "use_nav_reference": bool(o("use_nav_reference", True)),
        "nav_accuracy_h": float(o("nav_accuracy_h", 0.1)),
        "nav_accuracy_v": float(o("nav_accuracy_v", 0.5)),
        "make_report": bool(o("make_report", o("save_project", True))),
        "save_project": bool(o("save_project", True)),
    }

    chunks = []
    run_dir_by_win = {}
    for photos, run_dir, nav_csv, label in frame_sets:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        win_run = _to_windows_path(run_dir)
        run_dir_by_win[win_run] = run_dir
        chunks.append({
            "label": label,
            "run_dir": win_run,
            "photos": [_to_windows_path(p) for p in photos],
            "nav_csv": _to_windows_path(nav_csv) if nav_csv else None,
        })

    Path(project_psx).parent.mkdir(parents=True, exist_ok=True)
    params = {
        "project_psx": _to_windows_path(project_psx),
        "log_path": _to_windows_path(log_path),
        "result_path": _to_windows_path(result_path),
        "options": options,
        "chunks": chunks,
    }
    params_path.write_text(json.dumps(params, indent=1), encoding="utf-8")
    if result_path.exists():
        result_path.unlink()

    cmd = [str(exe), "-r", _to_windows_path(worker_dst), _to_windows_path(params_path)]
    log(f"  $ metashape.exe -r metashape_worker.py params_{tag}.json")
    log(f"  driving Metashape as a subprocess ({len(chunks)} chunk(s))")

    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
    except OSError as e:
        raise RuntimeError(f"Failed to launch Metashape: {e}") from e

    emitted = 0
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if file_log_fn:
            file_log_fn(f"    {line}")
        # Keep the GUI log readable: forward worker/progress lines, cap the rest.
        if line.startswith("[worker]") or emitted < 400:
            if log_fn:
                log_fn(f"    {line}")
            emitted += 1
    proc.wait()
    log(f"  Metashape subprocess finished in {time.time() - t0:.0f}s (exit {proc.returncode})")

    if not result_path.exists():
        raise RuntimeError(
            "Metashape produced no result file — the subprocess likely failed to "
            f"start the script. Check {log_path}. Exit code {proc.returncode}.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result.get("ok"):
        raise RuntimeError(f"Metashape pipeline failed: {result.get('error', 'unknown')}")

    # Translate product paths back to WSL and group by run_dir.
    out: dict[str, list[str]] = {}
    for ch in result.get("chunks", []):
        wsl_run = run_dir_by_win.get(ch.get("run_dir"), ch.get("run_dir"))
        paths = []
        for _key, winpath in (ch.get("products") or {}).items():
            paths.append(_from_windows_path(winpath))
        out[wsl_run] = paths
        log(f"  chunk '{ch.get('label')}' — {ch.get('cameras_aligned')}/"
            f"{ch.get('cameras_total')} cameras aligned, {len(paths)} product(s)")
        if ch.get("error"):
            log(f"    chunk error: {ch['error']}")
    return out


def _from_windows_path(p: str) -> str:
    """Translate a Windows path the worker wrote back into a WSL path."""
    s = str(p)
    if not _is_wsl():
        return s
    try:
        import subprocess as _sp
        return _sp.check_output(["wslpath", "-u", s], text=True).strip()
    except Exception:                                          # noqa: BLE001
        return s


def run_metashape_batch(project_psx, frame_sets, **opts):
    """Batch photogrammetry dispatcher: ONE project, many chunks.

    Chooses HOW to drive Metashape:
      * in-process  — when ``import Metashape`` works in this interpreter
                      (native Windows/Linux with the module installed);
      * subprocess  — otherwise, when a Windows metashape.exe is found (the
                      common WSL case: the module can't be imported here, but
                      the exe runs the pipeline in its own interpreter via
                      ``metashape.exe -r metashape_worker.py``).

    Both paths honour the same options and return {run_dir: [product paths]}.
    """
    try:
        import Metashape  # noqa: F401
    except ImportError:
        exe = _find_windows_metashape_exe()
        if not exe:
            raise RuntimeError(metashape_unavailable_reason() or
                               "Metashape is not available.")
        return _run_metashape_batch_subprocess(exe, project_psx, frame_sets, **opts)
    return _run_metashape_batch_inproc(project_psx, frame_sets, **opts)


def _run_metashape_batch_inproc(
    project_psx,
    frame_sets,                       # list of (frame_dir, run_dir, nav_csv)
    *,
    align_accuracy: str = "High",
    key_point_limit: int = 40000,
    tie_point_limit: int = 10000,
    generic_preselect: bool = True,
    reference_preselect: bool = True,
    adaptive_fitting: bool = True,
    reset_cameras: bool = False,
    build_dense: bool = True,
    dense_quality: str = "Medium",
    depth_filter: str = "Moderate",
    reuse_depth: bool = False,
    build_mesh: bool = False,
    mesh_surface: str = "Arbitrary",
    mesh_faces: str = "Medium",
    mesh_source: str = "Dense cloud",
    mesh_vertex_colors: bool = True,
    build_texture: bool = False,
    texture_size: int = 4096,
    texture_blending: str = "Mosaic",
    texture_fill_holes: bool = True,
    export_dense_ply: bool = False,   # ARCHIVED deliverable; dense build stays on
    export_mesh_obj: bool = False,
    use_nav_reference: bool = True,
    nav_accuracy_h: float = 0.1,
    nav_accuracy_v: float = 0.5,
    save_project: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
    file_log_fn: Optional[Callable[[str], None]] = None,
    **_unused,   # DEM/orthomosaic flags are handled by the subprocess worker;
                 # absorbed here so the dispatcher can forward one option set to
                 # either backend without TypeError. (The in-process path does
                 # not yet build ortho/DEM — those run via metashape_worker.py.)
) -> dict[str, list[str]]:
    """Batch photogrammetry: ONE Metashape project, many chunks.

    This is the headless equivalent of Metashape's Batch Process — no GUI: a
    single Document holds every chunk, each chunk is processed with the same
    per-chunk pipeline as run_metashape, and each chunk's products are exported
    into ITS OWN run directory in the app's file tree.  One *dataset* → one
    .psx; its intervals (split into ≤N-image parts by the caller) → chunks.

    frame_sets : list of (photos, run_dir, nav_csv, label) where
        photos  — explicit list of image paths for this chunk (the caller has
                  already applied interval-respecting ≤chunk_size splitting
                  via build_chunk_sets)
        run_dir — directory this chunk's products are exported into
        nav_csv — that interval's interp.csv (per-chunk georeferencing)
        label   — chunk label shown in the Metashape GUI (e.g. interval01_part02)

    Returns {run_dir: [product paths]} — one entry per chunk.
    """
    import Metashape

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    t_batch = time.time()
    api   = _meta_api(Metashape)
    major = _metashape_major(Metashape)
    if key_point_limit <= 0:
        key_point_limit = 40000
    if tie_point_limit <= 0:
        tie_point_limit = 4000

    project_psx = Path(project_psx)
    project_psx.parent.mkdir(parents=True, exist_ok=True)
    doc = Metashape.Document()
    doc.save(str(project_psx))
    n_photos_total = sum(len(fs[0]) for fs in frame_sets)
    log(f"Metashape {major}.x batch project: {project_psx}")
    log(f"  {len(frame_sets)} chunk(s), {n_photos_total} image(s) total — "
        "all chunks live in this one project")

    results: dict[str, list[str]] = {}
    for idx, (photos, run_dir, nav_csv, label) in enumerate(frame_sets, start=1):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if not photos:
            log(f"  ⚠ chunk {idx} ({label}): no images — skipping")
            results[str(run_dir)] = []
            continue

        t_chunk = time.time()
        chunk = doc.addChunk()
        chunk.label = label or run_dir.name
        chunk.addPhotos(list(photos))
        if save_project:
            doc.save()
        log("")
        log(f"  ── chunk {idx}/{len(frame_sets)} '{chunk.label}': "
            f"{len(photos)} frames → {run_dir}")
        log(f"      images: {Path(photos[0]).name} … {Path(photos[-1]).name}")
        log(f"      georef: {nav_csv or '(none)'}")

        opts = dict(
            align_accuracy=align_accuracy, key_point_limit=key_point_limit,
            tie_point_limit=tie_point_limit, generic_preselect=generic_preselect,
            reference_preselect=reference_preselect, adaptive_fitting=adaptive_fitting,
            reset_cameras=reset_cameras, build_dense=build_dense,
            dense_quality=dense_quality, depth_filter=depth_filter, reuse_depth=reuse_depth,
            build_mesh=build_mesh, mesh_surface=mesh_surface, mesh_faces=mesh_faces,
            mesh_source=mesh_source, mesh_vertex_colors=mesh_vertex_colors,
            build_texture=build_texture, texture_size=texture_size,
            texture_blending=texture_blending, texture_fill_holes=texture_fill_holes,
            export_dense_ply=export_dense_ply, export_mesh_obj=export_mesh_obj,
            nav_csv=nav_csv, use_nav_reference=use_nav_reference,
            nav_accuracy_h=nav_accuracy_h, nav_accuracy_v=nav_accuracy_v,
        )
        try:
            products = _process_metashape_chunk(
                Metashape, doc, chunk, run_dir, api=api, major=major,
                opts=opts, save_project=save_project, log=log,
                file_log=file_log_fn,
            )
            products["metashape_psx"] = str(project_psx)
            for k, v in products.items():
                log(f"        [metashape] {k}: {v}")
            results[str(run_dir)] = list(products.values())
            log(f"  ── chunk {idx}/{len(frame_sets)} '{chunk.label}' "
                f"finished in {time.time() - t_chunk:.1f} s")
        except Exception as exc:  # noqa: BLE001 — one bad chunk must not abort the batch
            log(f"  ⚠ chunk {idx} ('{label}') failed after "
                f"{time.time() - t_chunk:.1f} s (continuing): {exc}")
            results[str(run_dir)] = []

    if save_project:
        doc.save()
    ok = sum(1 for v in results.values() if v)
    log(f"Metashape batch complete in {time.time() - t_batch:.1f} s — "
        f"{ok}/{len(frame_sets)} chunk(s) produced output. Project: {project_psx} "
        f"({_fsize(project_psx)})")
    return results


def _collect_frames(frame_dir: str) -> list[str]:
    """Return sorted list of JPEG/PNG paths in frame_dir."""
    p = Path(frame_dir)
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in exts)


def _seed_camera_locations(
    chunk,
    nav_csv: str,
    log_fn: Callable,
    accuracy_h: float = 0.1,
    accuracy_v: float = 0.5,
) -> int:
    """Pre-populate camera reference locations from an interp CSV.

    Matches each camera to the closest nav timestamp derived from the frame
    filename.  Sets chunk.crs to Metashape.CoordinateSystem("EPSG::4326") and
    assigns lat/lon/alt with per-camera reference accuracy.

    The app's interp CSVs carry the time as ``timestamp_iso`` (plus
    ``unix_time``); a plain ``timestamp`` column is accepted for external
    files.  Returns the number of cameras seeded (0 on any failure) so the
    caller can decide whether reference preselection is meaningful.
    """
    try:
        import Metashape
        import pandas as pd

        nav = pd.read_csv(nav_csv)
        cols = {c.lower(): c for c in nav.columns}
        lat_col = cols.get("lat") or cols.get("latitude")
        lon_col = cols.get("lon") or cols.get("longitude")
        alt_col = cols.get("alt") or cols.get("altitude")
        ts_col  = cols.get("timestamp_iso") or cols.get("timestamp")

        if lat_col is None or lon_col is None:
            log_fn(f"      nav seeding skipped: no lat/lon columns in {nav_csv}")
            return 0
        if ts_col is not None:
            nav["_ts"] = pd.to_datetime(nav[ts_col], utc=False, errors="coerce")
        elif "unix_time" in cols:
            nav["_ts"] = pd.to_datetime(nav[cols["unix_time"]], unit="s", errors="coerce")
        else:
            log_fn(f"      nav seeding skipped: no timestamp_iso/timestamp/unix_time column in {nav_csv}")
            return 0
        nav = nav.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
        if nav.empty:
            log_fn("      nav seeding skipped: no parseable timestamps in nav CSV")
            return 0

        chunk.crs = Metashape.CoordinateSystem("EPSG::4326")

        seeded = 0
        no_ts  = 0
        far    = 0   # cameras whose nearest nav sample is > 2 s away
        for camera in chunk.cameras:
            ts = _timestamp_from_camera_label(camera.label)
            if ts is None:
                no_ts += 1
                continue
            deltas = (nav["_ts"] - ts).abs()
            idx = deltas.idxmin()
            if deltas.loc[idx].total_seconds() > 2.0:
                far += 1
            row = nav.iloc[idx]
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            alt = float(row[alt_col]) if (alt_col and pd.notna(row[alt_col])) else 0.0
            camera.reference.location = Metashape.Vector([lon, lat, alt])
            camera.reference.accuracy  = Metashape.Vector([accuracy_h, accuracy_h, accuracy_v])
            camera.reference.enabled   = True
            seeded += 1

        log_fn(f"Nav seeding: {seeded}/{len(chunk.cameras)} cameras pre-positioned "
               f"(H±{accuracy_h} m, V±{accuracy_v} m)"
               + (f"; {no_ts} label(s) had no parseable timestamp" if no_ts else "")
               + (f"; ⚠ {far} matched >2 s from the nearest nav sample" if far else ""))
        return seeded
    except Exception as exc:
        log_fn(f"Nav seeding skipped: {exc}")
        return 0


def _timestamp_from_camera_label(label: str) -> Optional[datetime]:
    """Parse the FRAME datetime from a frame filename stem.

    Frame names are ``{video_stem}_{YYYYMMDDTHHMMSS}_{mmm}.jpg`` and the video
    stem itself usually contains the video's start time (``YYYYMMDD_HHMMSS``),
    so the label holds TWO timestamps.  The frame's own capture time is the
    LAST one — matching the first would seed every frame of a video to the
    same (start) position.
    """
    import re
    # The millisecond suffix must be a full 3-digit token (not the first digits
    # of a FOLLOWING timestamp) — hence the (?!\d) boundary.  Without it, a
    # label like "20260115_170000_20260115T170600_000" would have its first
    # match swallow "_202" as milliseconds and corrupt the scan for the second
    # (frame) timestamp.
    matches = list(re.finditer(
        r"(\d{4})(\d{2})(\d{2})[_T](\d{2})(\d{2})(\d{2})(?:_(\d{3})(?!\d))?", label
    ))
    for m in reversed(matches):
        try:
            ms = int(m.group(7)) if m.group(7) else 0
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6)),
                ms * 1000,
            )
        except ValueError:
            continue
    return None


def _export_cameras_json(chunk, output_path: str) -> None:
    """Write camera poses to a simple JSON file.

    Each entry has: label, aligned (bool), and if aligned: T (4x4 matrix),
    reference_location ([lon, lat, alt] or null).
    """
    import Metashape
    cameras = []
    for camera in chunk.cameras:
        entry: dict = {"label": camera.label, "aligned": bool(camera.transform)}
        if camera.transform:
            t = camera.transform
            entry["transform_4x4"] = [list(t.row(i)) for i in range(4)]
        if camera.reference.location:
            loc = camera.reference.location
            entry["reference_location"] = [loc.x, loc.y, loc.z]
        cameras.append(entry)
    with open(output_path, "w") as f:
        json.dump({"cameras": cameras}, f, indent=2)


def launch_in_metashape(psx_path: str, exe: Optional[str] = None) -> None:
    """Open a .psx project in the Metashape GUI (non-blocking)."""
    if not exe:
        exe = _find_metashape_exe()
    subprocess.Popen([exe, psx_path], start_new_session=True)


# ---------------------------------------------------------------------------
# COLMAP engine
# ---------------------------------------------------------------------------

def run_colmap(
    run_dir: Path,
    frame_dir: str,
    *,
    nav_csv: Optional[str] = None,
    single_camera: bool = True,
    matcher: str = "Exhaustive",
    max_features: int = 8192,
    georeference: bool = True,
    build_dense: bool = True,
    export_camera_trajectory: bool = True,
    export_undistorted: bool = False,
    export_depth_maps: bool = False,
    build_poisson_mesh: bool = False,
    build_delaunay_mesh: bool = False,
    colmap_bin: str = "colmap",
    log_fn: Optional[Callable[[str], None]] = None,
    file_log_fn: Optional[Callable[[str], None]] = None,
) -> dict[str, str]:
    """Run the COLMAP SfM (+ optional MVS, meshing, georeferencing) pipeline.

    Pipeline (stages gated by the toggles):
      1. feature_extractor          (single_camera optional)
      2. <matcher>                  (exhaustive / sequential / vocab_tree / spatial)
      3. mapper                     → sparse model
      4. model_aligner              → georeference to nav (UTM E/N/alt) if requested
      5. export sparse cloud PLY
      6. export camera trajectory   (PLY of camera centres + JSON)
      7. image_undistorter → patch_match_stereo → stereo_fusion → dense cloud PLY
      8. poisson_mesher             → watertight mesh PLY
      9. delaunay_mesher            → detail-preserving mesh PLY

    All georeferenced products (sparse, trajectory, dense, meshes) come out in
    the navigation UTM frame when `georeference` is on and nav_csv is provided.

    Returns dict mapping product keys to absolute file paths.
    """
    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    photos = _collect_frames(frame_dir)
    if not photos:
        raise FileNotFoundError(f"No images found in {frame_dir}")
    log(f"COLMAP: {len(photos)} frames from {frame_dir}")
    log(f"COLMAP: matcher={matcher}, max_features={max_features}, single_camera={single_camera}")
    log(f"COLMAP: georeference={georeference}, dense={build_dense}, "
        f"poisson={build_poisson_mesh}, delaunay={build_delaunay_mesh}")

    colmap_dir = run_dir / "colmap"
    colmap_dir.mkdir(exist_ok=True)
    db_path    = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    dense_dir  = colmap_dir / "dense"
    sparse_dir.mkdir(exist_ok=True)

    products: dict[str, str] = {}

    # ── 1. Feature extraction ──────────────────────────────────────────────
    log(f"COLMAP: extracting features (max={max_features})…")
    feat_args = [
        "feature_extractor",
        "--database_path", str(db_path),
        "--image_path",    frame_dir,
        "--SiftExtraction.max_num_features", str(max_features),
        "--ImageReader.camera_model", "RADIAL",
    ]
    if single_camera:
        feat_args += ["--ImageReader.single_camera", "1"]
    _colmap_run(colmap_bin, feat_args, log, file_log_fn)

    # ── 1b. Build geo.txt + inject position priors (for georef / spatial) ───
    geo_txt: Optional[Path] = None
    if (georeference or matcher.lower() == "spatial") and nav_csv:
        geo_txt = colmap_dir / "geo.txt"
        n = _build_colmap_geo_txt(nav_csv, photos, geo_txt, log)
        if n == 0:
            log("  ⚠ No frame→position matches found in nav CSV; georeferencing disabled.")
            geo_txt = None
        else:
            products["georef_txt"] = str(geo_txt)
            if matcher.lower() == "spatial":
                _inject_db_positions(db_path, geo_txt, log)

    # ── 2. Matching ────────────────────────────────────────────────────────
    matcher_cmd = _COLMAP_MATCHER_CMD.get(matcher.lower(), "exhaustive_matcher")
    log(f"COLMAP: {matcher.lower()} feature matching…")
    _colmap_run(colmap_bin, [matcher_cmd, "--database_path", str(db_path)], log, file_log_fn)

    # ── 3. Sparse reconstruction ───────────────────────────────────────────
    log("COLMAP: sparse reconstruction (mapper)…")
    _colmap_run(colmap_bin, [
        "mapper",
        "--database_path", str(db_path),
        "--image_path",    frame_dir,
        "--output_path",   str(sparse_dir),
    ], log, file_log_fn)

    sparse_model = sparse_dir / "0"
    if not sparse_model.exists():
        raise RuntimeError(
            "COLMAP mapper produced no reconstruction. "
            "Check image overlap — underwater images may need higher quality preset."
        )

    # ── 4. Georeference to navigation (model_aligner) ───────────────────────
    #     Aligns the sparse model into the nav UTM frame; everything downstream
    #     (dense, meshes, trajectory) inherits the georeferenced coordinates.
    model = sparse_model
    if georeference and geo_txt is not None:
        geo_model = colmap_dir / "sparse_geo"
        geo_model.mkdir(exist_ok=True)
        log("COLMAP: georeferencing to navigation (model_aligner)…")
        try:
            _colmap_run(colmap_bin, [
                "model_aligner",
                "--input_path",          str(sparse_model),
                "--output_path",         str(geo_model),
                "--ref_images_path",     str(geo_txt),
                "--ref_is_gps",          "0",
                "--alignment_type",      "custom",
                "--alignment_max_error", "3.0",
            ], log, file_log_fn)
            if (geo_model / "images.bin").exists() or (geo_model / "images.txt").exists():
                model = geo_model
                log("  ✓ model georeferenced into navigation UTM frame")
            else:
                log("  ⚠ model_aligner produced no output; using un-georeferenced model")
        except Exception as exc:  # noqa: BLE001 — georef is best-effort
            log(f"  ⚠ Georeferencing failed (continuing un-georeferenced): {exc}")

    # ── 5. Export sparse cloud PLY ──────────────────────────────────────────
    sparse_ply = run_dir / "sparse_cloud.ply"
    _colmap_run(colmap_bin, [
        "model_converter",
        "--input_path",  str(model),
        "--output_path", str(sparse_ply),
        "--output_type", "PLY",
    ], log, file_log_fn)
    products["sparse_ply"] = str(sparse_ply)
    log(f"Sparse cloud: {sparse_ply}")

    # ── 6. Camera trajectory (camera centres + poses) ───────────────────────
    if export_camera_trajectory:
        try:
            traj_ply  = run_dir / "camera_trajectory.ply"
            traj_json = run_dir / "cameras.json"
            n = _export_colmap_trajectory(model, traj_ply, traj_json, colmap_bin, log)
            if n > 0:
                products["camera_trajectory_ply"] = str(traj_ply)
                products["cameras_json"]          = str(traj_json)
                log(f"Camera trajectory: {traj_ply} ({n} cameras)")
        except Exception as exc:  # noqa: BLE001 — non-fatal QA product
            log(f"  ⚠ Camera trajectory export skipped: {exc}")

    # ── 7. Dense reconstruction (best-effort: never discards the sparse cloud) ─
    if build_dense:
        try:
            dense_dir.mkdir(exist_ok=True)
            log("COLMAP: undistorting images…")
            _colmap_run(colmap_bin, [
                "image_undistorter",
                "--image_path",  frame_dir,
                "--input_path",  str(model),
                "--output_path", str(dense_dir),
                "--output_type", "COLMAP",
            ], log, file_log_fn)
            if export_undistorted and (dense_dir / "images").is_dir():
                products["undistorted_dir"] = str(dense_dir / "images")
                log(f"Undistorted frames: {dense_dir / 'images'}")

            log("COLMAP: PatchMatch stereo (CUDA GPU required)…")
            _colmap_run(colmap_bin, [
                "patch_match_stereo",
                "--workspace_path", str(dense_dir),
            ], log, file_log_fn)
            if export_depth_maps and (dense_dir / "stereo" / "depth_maps").is_dir():
                products["depth_maps_dir"] = str(dense_dir / "stereo" / "depth_maps")
                products["normal_maps_dir"] = str(dense_dir / "stereo" / "normal_maps")
                log(f"Depth/normal maps: {dense_dir / 'stereo'}")

            log("COLMAP: stereo fusion…")
            dense_ply = run_dir / "dense_cloud.ply"
            _colmap_run(colmap_bin, [
                "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--output_path",    str(dense_ply),
            ], log, file_log_fn)
            products["dense_ply"] = str(dense_ply)
            log(f"Dense cloud: {dense_ply}")

            fused = dense_dir / "fused.ply"
            if not fused.exists():
                fused = dense_ply  # some COLMAP builds write directly to output_path

            # ── 8. Poisson mesh ────────────────────────────────────────────
            if build_poisson_mesh:
                try:
                    mesh_p = run_dir / "mesh_poisson.ply"
                    log("COLMAP: Poisson meshing…")
                    _colmap_run(colmap_bin, [
                        "poisson_mesher",
                        "--input_path",  str(fused),
                        "--output_path", str(mesh_p),
                    ], log, file_log_fn)
                    products["mesh_poisson_ply"] = str(mesh_p)
                    log(f"Poisson mesh: {mesh_p}")
                except Exception as exc:  # noqa: BLE001
                    log(f"  ⚠ Poisson meshing failed: {exc}")

            # ── 9. Delaunay mesh ───────────────────────────────────────────
            if build_delaunay_mesh:
                try:
                    mesh_d = run_dir / "mesh_delaunay.ply"
                    log("COLMAP: Delaunay meshing…")
                    _colmap_run(colmap_bin, [
                        "delaunay_mesher",
                        "--input_path",  str(dense_dir),
                        "--output_path", str(mesh_d),
                    ], log, file_log_fn)
                    products["mesh_delaunay_ply"] = str(mesh_d)
                    log(f"Delaunay mesh: {mesh_d}")
                except Exception as exc:  # noqa: BLE001
                    log(f"  ⚠ Delaunay meshing failed: {exc}")

        except Exception as exc:  # noqa: BLE001 — keep sparse cloud on dense failure
            log(f"  ⚠ Dense reconstruction failed — sparse cloud is preserved. Reason: {exc}")
            log("    (PatchMatch stereo requires a CUDA GPU; check GPU availability in WSL.)")

    # Final product summary with explicit paths.
    log(f"COLMAP run complete. Products written under: {run_dir}")
    for key, path in products.items():
        log(f"  [colmap] {key}: {path}")
    return products


def _build_colmap_geo_txt(nav_csv: str, photos: list[str], out_path: Path,
                          log_fn: Callable) -> int:
    """Write a COLMAP geo.txt (`image_name X Y Z` per line) from an interp CSV.

    Uses UTM easting/northing and Z = -depth — the SAME convention as the app's
    sensor/nav 3-D PLYs (see _neg_depth_csv) — so a georeferenced COLMAP cloud
    overlays the sensor products in one viewer scene.  Falls back to converting
    lat/lon when easting/northing are absent.  Returns the number of lines.
    """
    import math
    import pandas as pd

    df = pd.read_csv(nav_csv)
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("frame_filename") or cols.get("filename")
    if name_col is None:
        log_fn("  ⚠ geo.txt: nav CSV has no frame_filename column.")
        return 0

    e_col, n_col = cols.get("easting"), cols.get("northing")
    lat_col = cols.get("lat") or cols.get("latitude")
    lon_col = cols.get("lon") or cols.get("longitude")
    d_col   = cols.get("depth") or cols.get("water_depth")

    photo_names = {Path(p).name for p in photos}
    out_lines: list[str] = []
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        if not name or name not in photo_names:
            continue
        try:
            if e_col and n_col:
                x, y = float(row[e_col]), float(row[n_col])
            elif lat_col and lon_col:
                import utm
                x, y, *_ = utm.from_latlon(float(row[lat_col]), float(row[lon_col]))
            else:
                continue
            z = -float(row[d_col]) if (d_col and pd.notna(row[d_col])) else 0.0
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        out_lines.append(f"{name} {x:.4f} {y:.4f} {z:.4f}")

    out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    log_fn(f"  geo.txt: {len(out_lines)} frame positions written → {out_path}")
    return len(out_lines)


def _inject_db_positions(db_path: Path, geo_txt: Path, log_fn: Callable) -> None:
    """Write position priors into the COLMAP database for spatial matching.

    Best-effort: updates images.prior_tx/ty/tz keyed by image name.  Silently
    no-ops (with a log line) if the schema differs across COLMAP versions.
    """
    import sqlite3

    pos: dict[str, tuple] = {}
    for line in geo_txt.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try:
                pos[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                continue
    if not pos:
        return
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        have = {r[1] for r in cur.execute("PRAGMA table_info(images)").fetchall()}
        if not {"prior_tx", "prior_ty", "prior_tz"}.issubset(have):
            log_fn("  ⚠ spatial matching: DB lacks prior position columns; skipping injection.")
            con.close()
            return
        updated = 0
        for name, (x, y, z) in pos.items():
            cur.execute(
                "UPDATE images SET prior_tx=?, prior_ty=?, prior_tz=? WHERE name=?",
                (x, y, z, name),
            )
            updated += cur.rowcount
        con.commit()
        con.close()
        log_fn(f"  spatial matching: injected positions for {updated} image(s)")
    except Exception as exc:  # noqa: BLE001 — best-effort
        log_fn(f"  ⚠ spatial matching: DB position injection failed: {exc}")


def _export_colmap_trajectory(model_dir: Path, out_ply: Path, out_json: Path,
                              colmap_bin: str, log_fn: Callable) -> int:
    """Export camera centres + poses from a COLMAP model.

    Converts the model to TXT, parses images.txt (camera centre C = -Rᵀ·t),
    and writes:
      • out_ply  — one coloured point per camera (the camera path)
      • out_json — per-camera name, centre, and quaternion
    Returns the number of cameras exported.
    """
    import tempfile

    txt_dir = Path(tempfile.mkdtemp(prefix="colmap_txt_"))
    _colmap_run(colmap_bin, [
        "model_converter",
        "--input_path",  str(model_dir),
        "--output_path", str(txt_dir),
        "--output_type", "TXT",
    ], log_fn)

    images_txt = txt_dir / "images.txt"
    if not images_txt.exists():
        return 0

    data_lines = [l for l in images_txt.read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
    cams: list[dict] = []
    # COLMAP images.txt: 2 lines per image (header, then 2-D points). Take headers.
    for k in range(0, len(data_lines), 2):
        parts = data_lines[k].split()
        if len(parts) < 10:
            continue
        try:
            qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        except ValueError:
            continue
        name = parts[9]
        cx, cy, cz = _camera_center(qw, qx, qy, qz, tx, ty, tz)
        cams.append({"name": name, "center": [cx, cy, cz], "quat": [qw, qx, qy, qz]})

    if not cams:
        return 0
    cams.sort(key=lambda c: c["name"])  # frame order → connectable path

    # ASCII PLY of camera centres (amber so the path stands out over clouds).
    hdr = [
        "ply", "format ascii 1.0", f"element vertex {len(cams)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header",
    ]
    body = [f"{c['center'][0]:.4f} {c['center'][1]:.4f} {c['center'][2]:.4f} 255 200 0"
            for c in cams]
    out_ply.write_text("\n".join(hdr + body) + "\n")
    out_json.write_text(json.dumps({"cameras": cams}, indent=2))
    return len(cams)


def _camera_center(qw: float, qx: float, qy: float, qz: float,
                   tx: float, ty: float, tz: float) -> tuple:
    """COLMAP camera centre in world coords: C = -Rᵀ·t (R = world→cam rotation)."""
    # Rotation matrix from a Hamilton quaternion (COLMAP convention).
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)
    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qx * qw)
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx * qx + qy * qy)
    # C = -Rᵀ t
    cx = -(r00 * tx + r10 * ty + r20 * tz)
    cy = -(r01 * tx + r11 * ty + r21 * tz)
    cz = -(r02 * tx + r12 * ty + r22 * tz)
    return cx, cy, cz


# Hard cap on log lines forwarded per COLMAP stage.  COLMAP (esp.
# patch_match_stereo / feature_extractor) can emit tens of thousands of
# progress lines; forwarding every one floods the GUI's cross-thread signal
# queue and unbounded QTextEdit → freeze/crash.  We cap and summarise instead.
_COLMAP_MAX_LOG_LINES = 1500


def _colmap_run(colmap_bin: str, args: list[str], log_fn: Callable,
                file_fn: Optional[Callable] = None) -> None:
    """Run one COLMAP CLI stage with a BOUNDED GUI view and a COMPLETE file log.

    COLMAP's console output comes from glog on stderr.  We:
      • wrap in `stdbuf -oL -eL` (when available) to line-buffer;
      • read by NEWLINE only — COLMAP's carriage-return progress bars overwrite
        in place and are intentionally NOT each turned into a log line (that was
        a flood/crash hazard);
      • send the first _COLMAP_MAX_LOG_LINES lines to log_fn (GUI + file, since
        log_fn tees), then send the remainder to file_fn only (the complete,
        uncapped task log file) — so the GUI never floods but the file is whole.
    The exact command (with full paths) is logged first.
    """
    cmd = [colmap_bin] + args
    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL", "-eL"] + cmd
    log_fn(f"  $ {' '.join(cmd)}")

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    emitted = 0
    capped = False
    for raw in proc.stdout:
        # Keep only the final segment of a CR-progress line (drop the in-place
        # progress ticks), then forward the resulting text line.
        line = raw.replace("\r", "\n").rstrip("\n").split("\n")[-1].strip()
        if not line:
            continue
        if emitted < _COLMAP_MAX_LOG_LINES:
            log_fn(f"  {line}")            # GUI + file (log_fn tees to both)
            emitted += 1
        else:
            if file_fn:
                file_fn(f"  {line}")        # file only — keeps the file complete
            if not capped:
                log_fn(f"  … (GUI output capped at {_COLMAP_MAX_LOG_LINES} lines; "
                       "full output continues in the task log file)")
                capped = True
    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log_fn(f"  ✗ colmap {args[0]} exited with code {proc.returncode} "
               f"after {elapsed:.1f} s")
        raise RuntimeError(
            f"COLMAP '{args[0]}' exited with code {proc.returncode}. "
            "See log above for details."
        )
    log_fn(f"  ✓ colmap {args[0]} finished in {elapsed:.1f} s")


def launch_in_colmap_gui(database_path: str, colmap_bin: str = "colmap") -> None:
    """Open the COLMAP GUI with the given database (non-blocking)."""
    subprocess.Popen(
        [colmap_bin, "gui", "--database_path", database_path],
        start_new_session=True,
    )
