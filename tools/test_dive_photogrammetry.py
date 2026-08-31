#!/usr/bin/env python3
"""
test_dive_photogrammetry.py — end-to-end smoke test of the Metashape pipeline.

Extracts a window of overlapping frames from a real dive video, then runs the
app's actual photogrammetry batch (the WSL→Windows subprocess path) to produce
a sparse/dense cloud, mesh, DEM and orthomosaic — the full product chain on real
seafloor imagery.

Usage:
    python3 tools/test_dive_photogrammetry.py VIDEO [options]

    python3 tools/test_dive_photogrammetry.py \
        /home/troyboland/epr_imaging/old/DOWN/20260121T052420.MP4 \
        --start 600 --seconds 60 --fps 3 --out /tmp/dive_test

Options:
    --start SEC     start extraction this many seconds into the video (default 300)
    --seconds SEC   length of the window to extract (default 60)
    --fps N         frames per second to sample within the window (default 3)
    --out DIR       output workspace (default /tmp/dive_test)
    --nav CSV       optional nav/interp CSV for georeferencing (timestamp_iso,lat,lon,alt)
    --no-ortho      skip DEM+orthomosaic (align+dense+mesh only)
    --chunk N       max images per chunk (default 300; keep < 350)

A dense, overlapping window (short seconds × a few fps) reconstructs far better
than frames scattered across the whole dive, so prefer --seconds 60 --fps 3 over
sampling the entire video.
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def extract_window(video, out_dir, start_s, seconds, fps):
    """Write evenly-spaced JPEGs from [start_s, start_s+seconds] at `fps`.

    Filenames embed a YYYYMMDDTHHMMSS stamp derived from the video name + offset
    so the pipeline's georeference step can match them against a nav CSV.
    """
    import re
    from datetime import datetime, timedelta
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = total / vid_fps if vid_fps else 0
    print(f"video: {vid_fps:.2f} fps, {dur/60:.1f} min, {int(total)} frames")

    # Base timestamp from the filename (…YYYYMMDDTHHMMSS… or YYYY_MM_DDTHH_MM_SS)
    stem = Path(video).stem
    base_dt = None
    for rx in (r"(\d{8}T\d{6})", r"(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2})"):
        m = re.search(rx, stem)
        if m:
            s = m.group(1).replace("_", "")
            try:
                base_dt = datetime.strptime(s, "%Y%m%dT%H%M%S")
            except ValueError:
                pass
            break
    if base_dt is None:
        base_dt = datetime(2026, 1, 1, 0, 0, 0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, int(round(vid_fps / max(fps, 0.1))))
    start_f = int(start_s * vid_fps)
    end_f = int(min(total, (start_s + seconds) * vid_fps))
    n = 0
    for fi in range(start_f, end_f, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        ts = base_dt + timedelta(seconds=fi / vid_fps)
        name = f"frame_{ts:%Y%m%dT%H%M%S}_{fi % 1000:03d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n += 1
    cap.release()
    print(f"extracted {n} frames → {out_dir}")
    if n < 8:
        raise SystemExit("Too few frames — widen --seconds or raise --fps.")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=300)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--fps", type=float, default=3)
    ap.add_argument("--out", default="/tmp/dive_test")
    ap.add_argument("--nav", default=None)
    ap.add_argument("--no-ortho", action="store_true")
    ap.add_argument("--chunk", type=int, default=300)
    args = ap.parse_args()

    import photogrammetry_service as ps

    reason = ps.metashape_unavailable_reason()
    if reason:
        raise SystemExit(f"Metashape unavailable: {reason}")
    print(f"Metashape driver: {ps.metashape_driver()}  "
          f"({ps._find_windows_metashape_exe() or 'in-process'})")

    out = Path(args.out)
    frames_dir = out / "frames"
    extract_window(args.video, frames_dir, args.start, args.seconds, args.fps)
    frames = sorted(str(p) for p in frames_dir.glob("*.jpg"))

    run_dir = out / "run_000"
    run_dir.mkdir(parents=True, exist_ok=True)
    project = out / "project.psx"
    want_ortho = not args.no_ortho

    print(f"\nrunning photogrammetry ({len(frames)} frames, "
          f"chunk≤{args.chunk}, ortho={'on' if want_ortho else 'off'}) …\n")
    t0 = time.time()
    results = ps.run_metashape_batch(
        str(project),
        [(frames, str(run_dir), args.nav, "dive_test")],
        align_accuracy="High",
        build_dense=True, dense_quality="Medium", export_dense_ply=True,
        build_mesh=True, mesh_source="Depth maps", mesh_faces="Medium",
        export_mesh_obj=True,
        build_dem=want_ortho, export_dem=want_ortho, build_orthomosaic=want_ortho,
        use_nav_reference=bool(args.nav),
        make_report=True, save_project=True,
        log_fn=lambda m: print(m, flush=True),
    )

    print(f"\n=== DONE in {time.time()-t0:.0f}s ===")
    total = 0
    for rd, paths in results.items():
        for p in paths:
            exists = Path(p).exists()
            total += exists
            print(f"  {'✓' if exists else '✗'} {p}")
    if total == 0:
        raise SystemExit("No products produced — check the log above.")
    print(f"\n{total} product(s) written under {out}")


if __name__ == "__main__":
    main()
