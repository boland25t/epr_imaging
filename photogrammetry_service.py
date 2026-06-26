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


def _detect_metashape() -> str | None:
    """Return a Metashape executable path if the Python module is importable.

    The headless pipeline (run_metashape) drives Metashape through its Python
    module, so the engine is only considered "available" when `import Metashape`
    succeeds *in this interpreter*.  A Windows-only Metashape install is NOT
    importable from a WSL/Linux Python — see metashape_unavailable_reason().
    """
    try:
        import Metashape  # noqa: F401
        return _find_metashape_exe()
    except ImportError:
        return None


def metashape_unavailable_reason() -> str | None:
    """Explain why the Metashape engine is unavailable, or None if it's usable.

    Distinguishes the common WSL pitfall (Metashape installed on the Windows
    host but not importable from the Linux Python running this app) from a
    plain "not installed" so the UI can give actionable guidance.
    """
    try:
        import Metashape  # noqa: F401
        return None  # importable → available
    except ImportError:
        pass
    win_exe = _find_windows_metashape_exe()
    if _is_wsl() and win_exe:
        return (
            "Metashape is installed on the Windows host but cannot be imported "
            "from the Linux (WSL) Python running this app. The headless pipeline "
            "needs the Metashape Python module in *this* interpreter. Run the app "
            "on Windows (where the installer targets it), or install Metashape for "
            "Linux inside WSL. COLMAP works either way."
        )
    return (
        "Metashape Python module not found. Install Metashape Professional and "
        "its Python module, or use the COLMAP engine instead."
    )


def _find_windows_metashape_exe() -> str | None:
    """Locate a Windows Metashape.exe, including via /mnt/c when under WSL."""
    bases = [
        r"C:\Program Files\Agisoft\Metashape Professional",
        r"C:\Program Files (x86)\Agisoft\Metashape Professional",
    ]
    roots = [Path(b) for b in bases]
    if _is_wsl():
        # Translate C:\... → /mnt/c/... so we can see the Windows install.
        roots += [Path("/mnt/c/Program Files/Agisoft/Metashape Professional"),
                  Path("/mnt/c/Program Files (x86)/Agisoft/Metashape Professional")]
    for root in roots:
        exe = root / "Metashape.exe"
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
    export_dense_ply: bool = True,
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
    api          = _meta_api(Metashape)
    major        = _metashape_major(Metashape)
    dense_source = api["dense_source"]

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

    # ── Georeference: pre-seed camera positions ────────────────────────────────
    seeded_nav = False
    if nav_csv and use_nav_reference and Path(nav_csv).exists():
        _seed_camera_locations(chunk, nav_csv, log, nav_accuracy_h, nav_accuracy_v)
        seeded_nav = True

    # ── Alignment ─────────────────────────────────────────────────────────────
    acc_int = _META_ALIGN_ACC.get(align_accuracy.lower(), 1)
    log(f"Aligning cameras (accuracy={align_accuracy}, downscale={acc_int})…")
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
    chunk.matchPhotos(**match_kwargs)
    chunk.alignCameras(adaptive_fitting=adaptive_fitting)
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
    if build_dense:
        dq   = _META_DENSE_QUAL.get(dense_quality.lower(), 2)
        df   = _META_DEPTH_FILTER.get(depth_filter.lower(), 2)
        log(f"Building depth maps (quality={dense_quality}, filter={depth_filter})…")
        chunk.buildDepthMaps(
            downscale=dq,
            filter_mode=df,
            reuse_depth=reuse_depth,
        )
        log(f"Building {'point' if major >= 2 else 'dense'} cloud…")
        api["build_dense"](chunk)   # buildPointCloud() (2.x) / buildDenseCloud() (1.x)
        if save_project:
            doc.save()

        if export_dense_ply:
            dense_path = str(run_dir / "dense_cloud.ply")
            chunk.exportPointCloud(dense_path, source_data=dense_source, save_colors=True)
            products["dense_ply"] = dense_path
            log(f"Dense cloud: {dense_path}")

    # ── Mesh ──────────────────────────────────────────────────────────────────
    if build_mesh and build_dense:
        surf_attr = _META_SURFACE_TYPE.get(mesh_surface.lower(), "Arbitrary")
        face_attr = _META_FACE_COUNT.get(mesh_faces.lower(), "MediumFaceCount")
        # Resolve the mesh source enum version-correctly: "dense cloud" maps to
        # whatever the dense cloud is called in this version; "depth maps" is
        # stable across versions.
        if mesh_source.lower() == "depth maps":
            mesh_src_enum = Metashape.DataSource.DepthMapsData
        else:
            mesh_src_enum = dense_source
        log(f"Building mesh (surface={mesh_surface}, faces={mesh_faces})…")
        chunk.buildMesh(
            source_data=mesh_src_enum,
            surface_type=getattr(Metashape, surf_attr),
            face_count=getattr(Metashape, face_attr),
            vertex_colors=mesh_vertex_colors,
        )
        if save_project:
            doc.save()

        if export_mesh_obj:
            mesh_path = str(run_dir / "mesh.obj")
            chunk.exportModel(mesh_path, save_texture=False)
            products["mesh_obj"] = mesh_path
            log(f"Mesh: {mesh_path}")

        # ── Texture ───────────────────────────────────────────────────────────
        if build_texture:
            blend_attr = _META_BLENDING.get(texture_blending.lower(), "MosaicBlending")
            log(f"Building texture (size={texture_size}, blending={texture_blending})…")
            chunk.buildUV()
            chunk.buildTexture(
                blending_mode=getattr(Metashape, blend_attr),
                texture_size=texture_size,
                fill_holes=texture_fill_holes,
            )
            if save_project:
                doc.save()
            tex_mesh_path = str(run_dir / "mesh_textured.obj")
            chunk.exportModel(tex_mesh_path, save_texture=True)
            products["mesh_textured_obj"] = tex_mesh_path
            products["texture_png"]       = str(run_dir / "mesh_textured.png")
            log(f"Textured mesh: {tex_mesh_path}")

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

    products["metashape_psx"] = psx_path
    if save_project:
        doc.save()
    log("Metashape run complete.")
    return products


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
) -> None:
    """Pre-populate camera reference locations from interp_full.csv.

    Matches each camera to the closest nav timestamp derived from the frame
    filename (format YYYYMMDD_HHMMSS or unix-based).  Sets chunk.crs to
    Metashape.CoordinateSystem("EPSG::4326") and assigns lat/lon/alt with
    per-camera reference accuracy.
    """
    try:
        import Metashape
        import pandas as pd

        nav = pd.read_csv(nav_csv)
        if "timestamp" not in nav.columns or "lat" not in nav.columns:
            log_fn("Nav seeding skipped: required columns missing from interp_full.csv")
            return

        nav["timestamp"] = pd.to_datetime(nav["timestamp"], utc=False)
        nav = nav.sort_values("timestamp").reset_index(drop=True)

        chunk.crs = Metashape.CoordinateSystem("EPSG::4326")

        seeded = 0
        for camera in chunk.cameras:
            ts = _timestamp_from_camera_label(camera.label)
            if ts is None:
                continue
            idx = (nav["timestamp"] - ts).abs().idxmin()
            row = nav.iloc[idx]
            lat = float(row.get("lat", row.get("latitude", 0)))
            lon = float(row.get("lon", row.get("longitude", 0)))
            alt = float(row.get("alt", row.get("altitude", 0))) if "alt" in row.index or "altitude" in row.index else 0.0
            camera.reference.location = Metashape.Vector([lon, lat, alt])
            camera.reference.accuracy  = Metashape.Vector([accuracy_h, accuracy_h, accuracy_v])
            camera.reference.enabled   = True
            seeded += 1

        log_fn(f"Nav seeding: {seeded}/{len(chunk.cameras)} cameras pre-positioned "
               f"(H±{accuracy_h} m, V±{accuracy_v} m)")
    except Exception as exc:
        log_fn(f"Nav seeding skipped: {exc}")


def _timestamp_from_camera_label(label: str) -> Optional[datetime]:
    """Parse a datetime from a frame filename stem.

    Supports the video frame naming convention used by the EPR app:
      YYYYMMDD_HHMMSS  (prefix, may have _frame_NNNNN suffix)
    Returns None if the label does not contain a parseable timestamp.
    """
    import re
    m = re.search(r"(\d{4})(\d{2})(\d{2})[_T](\d{2})(\d{2})(\d{2})", label)
    if m:
        try:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)), int(m.group(6)),
            )
        except ValueError:
            pass
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
    if proc.returncode != 0:
        raise RuntimeError(
            f"COLMAP '{args[0]}' exited with code {proc.returncode}. "
            "See log above for details."
        )


def launch_in_colmap_gui(database_path: str, colmap_bin: str = "colmap") -> None:
    """Open the COLMAP GUI with the given database (non-blocking)."""
    subprocess.Popen(
        [colmap_bin, "gui", "--database_path", database_path],
        start_new_session=True,
    )
