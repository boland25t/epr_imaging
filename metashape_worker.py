#!/usr/bin/env python3
"""
metashape_worker.py — headless photogrammetry, run INSIDE Metashape's interpreter.

This script is NOT imported by the app.  It is executed by the Metashape
executable itself:

    metashape.exe -r metashape_worker.py <params.json>

That is the only interpreter where ``import Metashape`` succeeds, so all the
Metashape work lives here.  The application (which runs under WSL/Linux, where
the module cannot be imported) drives this via a subprocess — see
photogrammetry_service._run_metashape_batch_subprocess.  The two sides agree on
the JSON contract described below; keep them in sync.

Everything this script touches is a WINDOWS path: the caller has already
translated every WSL path (via ``wslpath -w``) before writing the params file,
because Metashape runs as a native Windows process.

Params JSON
-----------
{
  "project_psx": "C:\\...\\project.psx",     # created/overwritten
  "log_path":    "C:\\...\\worker.log",        # progress + errors (tee of stdout)
  "result_path": "C:\\...\\result.json",       # written on exit (see below)
  "options": { align/dense/mesh/dem/ortho flags and quality settings },
  "chunks": [
     { "label": "interval01_part01",
       "run_dir": "C:\\...\\run_000",           # products written here
       "photos": ["C:\\...\\f0001.jpg", ...],
       "nav_csv": "C:\\...\\interp.csv" | null   # optional georeference
     }, ...
  ]
}

Result JSON (always written, even on failure)
---------------------------------------------
{
  "ok": bool,
  "version": "2.3.1",
  "error": "..."               # present when ok is false
  "chunks": [
     { "label": ..., "run_dir": ...,
       "cameras_total": int, "cameras_aligned": int,
       "products": { "sparse_ply": "C:\\...", "dense_ply": ..., "mesh_obj": ...,
                     "orthomosaic_tif": ..., "dem_tif": ..., "report_pdf": ... },
       "error": "..."          # present if this chunk failed (others still run)
     }, ...
  ]
}
"""

import json
import sys
import time
import traceback

import Metashape


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------
_LOG = None


def log(msg):
    line = "[worker] %s" % msg
    print(line, flush=True)                 # streamed to the app via stdout
    if _LOG:
        _LOG.write(line + "\n")
        _LOG.flush()


# --------------------------------------------------------------------------
# version-stable enum lookups (2.x)
# --------------------------------------------------------------------------
def _accuracy(name):
    return {
        "Highest": Metashape.HighestAccuracy, "High": Metashape.HighAccuracy,
        "Medium": Metashape.MediumAccuracy, "Low": Metashape.LowAccuracy,
        "Lowest": Metashape.LowestAccuracy,
    }.get(name, Metashape.HighAccuracy)


def _quality(name):
    return {
        "Ultra High": Metashape.UltraQuality, "High": Metashape.HighQuality,
        "Medium": Metashape.MediumQuality, "Low": Metashape.LowQuality,
        "Lowest": Metashape.LowestQuality,
    }.get(name, Metashape.MediumQuality)


def _depth_filter(name):
    return {
        "Aggressive": Metashape.AggressiveFiltering,
        "Moderate": Metashape.ModerateFiltering,
        "Mild": Metashape.MildFiltering, "Disabled": Metashape.NoFiltering,
    }.get(name, Metashape.ModerateFiltering)


def _surface(name):
    return (Metashape.HeightField if name == "Height Field"
            else Metashape.Arbitrary)


def _faces(name):
    return {"Low": Metashape.LowFaceCount, "Medium": Metashape.MediumFaceCount,
            "High": Metashape.HighFaceCount}.get(name, Metashape.MediumFaceCount)


def _blending(name):
    return {"Mosaic": Metashape.MosaicBlending, "Average": Metashape.AverageBlending,
            "Max": Metashape.MaxBlending, "Min": Metashape.MinBlending,
            "Disabled": Metashape.DisabledBlending}.get(name, Metashape.MosaicBlending)


def _progress(stage):
    """A Metashape progress callback that logs '<stage> NN%' — throttled to at
    most one line per +5% or per 10s so long stages (depth maps, dense, mesh,
    DEM, ortho) stream live progress to the app UI instead of going silent.

    Metashape calls the callback with a float 0..100 very frequently; without the
    throttle it would flood the log."""
    import time
    state = {"pct": -100.0, "t": 0.0}

    def cb(pct):
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            return
        now = time.time()
        if pct - state["pct"] >= 5.0 or now - state["t"] >= 10.0 or pct >= 100.0:
            log("%s %.0f%%" % (stage, pct))
            state["pct"] = pct
            state["t"] = now
    return cb


def _call(fn, **kw):
    """Call a Metashape stage, dropping progress= if the build rejects it."""
    try:
        return fn(**kw)
    except TypeError:
        kw.pop("progress", None)
        return fn(**kw)


# --------------------------------------------------------------------------
# georeference: seed camera locations from an interp.csv (timestamp_iso → cam)
# --------------------------------------------------------------------------
def _utm_epsg(zone_str):
    """EPSG code for a UTM zone string like '13P' / '13N' / '13'.

    UTM latitude bands N..X are northern hemisphere (326xx), C..M southern
    (327xx).  Defaults to zone 13 North if unparseable."""
    import re
    m = re.match(r"\s*(\d{1,2})\s*([A-Za-z]?)", str(zone_str))
    if not m:
        return 32613
    zone = int(m.group(1))
    band = (m.group(2) or "N").upper()
    north = band >= "N"
    return (32600 if north else 32700) + zone


def _seed_reference(chunk, nav_csv, acc_h, acc_v):
    """Best-effort: give each camera a lat/lon/alt from the nav CSV.

    Matches on the frame filename's trailing YYYYMMDDTHHMMSS timestamp against
    the nav table's timestamp_iso.  Silently does nothing if columns are absent
    — georeferencing is optional and alignment still works without it.
    """
    import csv
    import re
    from datetime import datetime, timezone

    def parse_iso(s):
        s = (s or "").strip().rstrip("Zz")
        for f in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, f)
            except ValueError:
                pass
        return None

    rows = []
    use_utm = False
    epsg = None
    try:
        with open(nav_csv, "r", newline="") as fh:
            rd = csv.DictReader(fh)
            cols = {c.lower(): c for c in (rd.fieldnames or [])}
            tcol = cols.get("timestamp_iso") or cols.get("timestamp")
            eastc = cols.get("easting"); northc = cols.get("northing"); depthc = cols.get("depth")
            zonec = cols.get("utm_zone")
            latc = cols.get("lat") or cols.get("latitude")
            lonc = cols.get("lon") or cols.get("longitude")
            altc = cols.get("alt") or cols.get("altitude") or cols.get("depth")
            headc = cols.get("heading") or cols.get("yaw")
            pitchc = cols.get("pitch")
            rollc = cols.get("roll")
            # Prefer UTM easting/northing/depth (all metres) so the model is
            # georeferenced in the SAME frame as the sensor rasters/trackline and
            # Metashape gets consistent metric references.  Seeding lat/lon
            # (degrees) with a ~2500 m depth as altitude mixes units and both
            # mislocates AND distorts the reconstruction — the bug this fixes.
            use_utm = bool(eastc and northc and depthc)
            if not tcol or not (use_utm or (latc and lonc)):
                log("nav csv lacks usable georef columns — skipping georeference")
                return 0
            has_orient = bool(headc and pitchc and rollc)

            def _num(r, c):
                try:
                    return float(r[c]) if c and r.get(c) not in (None, "") else None
                except (ValueError, TypeError):
                    return None
            for r in rd:
                dt = parse_iso(r.get(tcol))
                if dt is None:
                    continue
                try:
                    t = dt.replace(tzinfo=timezone.utc).timestamp()
                    if use_utm:
                        # x=easting, y=northing, z=-depth (below sea level negative)
                        x = float(r[eastc]); y = float(r[northc]); z = -float(r[depthc])
                        if epsg is None and zonec and r.get(zonec):
                            epsg = _utm_epsg(r[zonec])
                    else:
                        x = float(r[lonc]); y = float(r[latc])
                        z = float(r[altc]) if altc and r.get(altc) not in (None, "") else 0.0
                    rows.append((t, x, y, z, _num(r, headc), _num(r, pitchc), _num(r, rollc)))
                except (ValueError, TypeError):
                    continue
    except OSError as e:
        log("could not read nav csv (%s) — skipping georeference" % e)
        return 0
    if not rows:
        return 0
    rows.sort()
    times = [r[0] for r in rows]

    # Set the chunk coordinate system: UTM (metres) when we have easting/northing,
    # else WGS84.  Camera references + all exported products land in this frame.
    crs_code = "EPSG::%d" % (epsg or 32613) if use_utm else "EPSG::4326"
    try:
        chunk.crs = Metashape.CoordinateSystem(crs_code)
    except Exception:                                           # noqa: BLE001
        pass
    log("georeference frame: %s (%s)" % (crs_code, "UTM metres" if use_utm else "WGS84 lat/lon"))

    import bisect
    frame_rx = re.compile(r"(\d{8}T\d{6})")
    seeded = 0
    oriented = 0
    for cam in chunk.cameras:
        m = list(frame_rx.finditer(cam.label))
        if not m:
            continue
        try:
            dt = datetime.strptime(m[-1].group(1), "%Y%m%dT%H%M%S")
        except ValueError:
            continue
        t = dt.replace(tzinfo=timezone.utc).timestamp()
        i = bisect.bisect_left(times, t)
        i = min(range(max(0, i - 1), min(len(rows), i + 1)),
                key=lambda k: abs(times[k] - t))
        _, x, y, z, heading, pitch, roll = rows[i]
        # x,y,z already in the chunk CRS axis order (UTM: easting/northing/-depth;
        # WGS84: lon/lat/alt) so the same vector serves both frames.
        cam.reference.location = Metashape.Vector([x, y, z])
        cam.reference.accuracy = Metashape.Vector([acc_h, acc_h, acc_v])
        cam.reference.enabled = True
        # Orientation prior (heading/pitch/roll → Metashape yaw/pitch/roll), seeded
        # LOOSELY: a large rotation accuracy makes it a gentle nudge, so if the
        # vehicle→camera convention is off it can't dominate the image-based
        # solution, but when it's right it helps constrain a low-parallax scene.
        if has_orient and None not in (heading, pitch, roll):
            try:
                cam.reference.rotation = Metashape.Vector([heading, pitch, roll])
                cam.reference.rotation_accuracy = Metashape.Vector([30.0, 30.0, 30.0])
                cam.reference.rotation_enabled = True
                oriented += 1
            except Exception:                                   # noqa: BLE001
                pass
        seeded += 1
    if seeded:
        # CRS already set above (UTM or WGS84); just refresh the transform so the
        # solved cameras are placed into that georeferenced frame.
        try:
            chunk.updateTransform()
        except Exception:                                       # noqa: BLE001
            pass
    log("georeference: seeded %d/%d cameras (%d with orientation)"
        % (seeded, len(chunk.cameras), oriented))
    return seeded


# --------------------------------------------------------------------------
# per-chunk pipeline
# --------------------------------------------------------------------------
def process_chunk(chunk, spec, opt):
    import os
    run_dir = spec["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    products = {}
    j = lambda name: os.path.join(run_dir, name)

    log("add %d photos" % len(spec["photos"]))
    chunk.addPhotos(spec["photos"])

    # ---- image quality gate (Metashape's own grading) ----
    # analyzeImages() scores each photo 0-1 on the sharpness of its sharpest
    # region; Agisoft recommends disabling anything below ~0.5.  For underwater
    # footage (motion blur, backscatter, turbidity) this removes garbage frames
    # that otherwise poison feature matching.  Disabled cameras are skipped by
    # matchPhotos/alignCameras.  Threshold 0 disables the gate.
    q_thresh = float(opt.get("quality_threshold", 0.5))
    if q_thresh > 0:
        try:
            analyze = getattr(chunk, "analyzeImages", None) or getattr(chunk, "estimateImageQuality")
            analyze()  # 2.x: analyzeImages(); 1.x: estimateImageQuality()
            scores, dropped = [], 0
            for cam in chunk.cameras:
                try:
                    q = float(cam.meta["Image/Quality"])
                except (KeyError, TypeError, ValueError):
                    continue
                scores.append(q)
                if q < q_thresh:
                    cam.enabled = False
                    dropped += 1
            if scores:
                import statistics
                log("image quality: min=%.2f med=%.2f max=%.2f — dropped %d/%d below %.2f"
                    % (min(scores), statistics.median(scores), max(scores),
                       dropped, len(scores), q_thresh))
        except Exception as e:                                  # noqa: BLE001
            log("image-quality gate skipped: %r" % e)

    # Seed nav reference (camera locations + orientation) BEFORE matching, so
    # REFERENCE PRESELECTION can use it — Metashape then only matches images that
    # are spatially near each other per the nav, which is the primary payoff of
    # importing navigation.  It is ON automatically whenever a nav reference is
    # supplied.  (Georeferencing of the solved cameras is finalised after align.)
    use_ref = bool(spec.get("nav_csv") and opt["use_nav_reference"])
    if use_ref:
        _seed_reference(chunk, spec["nav_csv"], opt["nav_accuracy_h"], opt["nav_accuracy_v"])

    log("matchPhotos + alignCameras (accuracy=%s, reference_preselection=%s)"
        % (opt["align_accuracy"], use_ref))
    match_kw = dict(
        downscale={"Highest": 0, "High": 1, "Medium": 2, "Low": 4,
                   "Lowest": 8}.get(opt["align_accuracy"], 1),
        generic_preselection=opt["generic_preselect"],
        reference_preselection=use_ref,
        keypoint_limit=opt["key_point_limit"],
        tiepoint_limit=opt["tie_point_limit"])
    if use_ref:
        try:
            match_kw["reference_preselection_mode"] = Metashape.ReferencePreselectionSource
        except AttributeError:
            pass  # older API: reference_preselection=True alone uses source coords
    _call(chunk.matchPhotos, **match_kw)
    _call(chunk.alignCameras, adaptive_fitting=opt["adaptive_fitting"])

    aligned = sum(1 for c in chunk.cameras if c.transform is not None)
    log("aligned %d/%d cameras" % (aligned, len(chunk.cameras)))
    if use_ref:
        try:
            chunk.updateTransform()   # georeference the solved cameras
        except Exception:             # noqa: BLE001
            pass

    if aligned >= 2:
        try:
            chunk.exportPointCloud(j("sparse.ply"),
                                   source_data=Metashape.TiePointsData)
            products["sparse_ply"] = j("sparse.ply")
        except Exception as e:                                  # noqa: BLE001
            log("sparse export failed: %r" % e)

    # ---- dense (2.x: depth maps → point cloud) ----
    have_dense = False
    if opt["build_dense"] and aligned >= 2:
        log("buildDepthMaps + buildPointCloud (quality=%s)" % opt["dense_quality"])
        _call(chunk.buildDepthMaps, downscale={"Ultra High": 1, "High": 2, "Medium": 4,
              "Low": 8, "Lowest": 16}.get(opt["dense_quality"], 4),
              filter_mode=_depth_filter(opt["depth_filter"]),
              progress=_progress("buildDepthMaps"))
        _call(chunk.buildPointCloud, progress=_progress("buildPointCloud"))
        have_dense = chunk.point_cloud is not None
        if have_dense and opt["export_dense_ply"]:
            chunk.exportPointCloud(j("dense.ply"), source_data=Metashape.PointCloudData)
            products["dense_ply"] = j("dense.ply")

    # ---- mesh ----
    have_mesh = False
    if opt["build_mesh"] and aligned >= 2:
        src = (Metashape.DepthMapsData if opt["mesh_source"] == "Depth maps"
               else Metashape.PointCloudData)
        if src == Metashape.PointCloudData and not have_dense:
            log("mesh wants dense cloud but none built — building point cloud first")
            _call(chunk.buildDepthMaps, downscale=4, progress=_progress("buildDepthMaps"))
            _call(chunk.buildPointCloud, progress=_progress("buildPointCloud"))
        log("buildModel (surface=%s, faces=%s, source=%s)"
            % (opt["mesh_surface"], opt["mesh_faces"], opt["mesh_source"]))
        _call(chunk.buildModel, surface_type=_surface(opt["mesh_surface"]),
              face_count=_faces(opt["mesh_faces"]), source_data=src,
              vertex_colors=opt["mesh_vertex_colors"], progress=_progress("buildModel"))
        have_mesh = chunk.model is not None
        if have_mesh:
            if opt["build_texture"]:
                _call(chunk.buildUV, progress=_progress("buildUV"))
                _call(chunk.buildTexture, blending_mode=_blending(opt["texture_blending"]),
                      texture_size=opt["texture_size"], fill_holes=opt["texture_fill_holes"],
                      progress=_progress("buildTexture"))
                chunk.exportModel(j("mesh_textured.obj"), save_texture=True)
                products["mesh_textured_obj"] = j("mesh_textured.obj")
            if opt["export_mesh_obj"]:
                chunk.exportModel(j("mesh.obj"), save_texture=False)
                products["mesh_obj"] = j("mesh.obj")

    # ---- DEM + orthomosaic (needs a georeferenced transform) ----
    if (opt["build_orthomosaic"] or opt["build_dem"]) and aligned >= 2:
        try:
            surface = (Metashape.DataSource.PointCloudData if have_dense
                       else Metashape.DataSource.TiePointsData)
            log("buildDem")
            _call(chunk.buildDem, source_data=surface, progress=_progress("buildDem"))
            if chunk.elevation is not None and opt["export_dem"]:
                chunk.exportRaster(j("dem.tif"), source_data=Metashape.ElevationData)
                products["dem_tif"] = j("dem.tif")
            if opt["build_orthomosaic"] and chunk.elevation is not None:
                # Orthomosaic is a true orthorectified projection onto the DEM by
                # default — NOT draped on the mesh (a noisy mesh warps the ortho).
                # Set ortho_surface="Mesh" to override where the mesh is trusted.
                want_mesh_ortho = (opt.get("ortho_surface", "DEM") == "Mesh") and have_mesh
                surf_for_ortho = Metashape.ModelData if want_mesh_ortho else Metashape.ElevationData
                log("buildOrthomosaic (surface=%s)"
                    % ("mesh" if want_mesh_ortho else "DEM"))
                _call(chunk.buildOrthomosaic, surface_data=surf_for_ortho,
                      blending_mode=_blending(opt["texture_blending"]),
                      fill_holes=True, progress=_progress("buildOrthomosaic"))
                if chunk.orthomosaic is not None:
                    chunk.exportRaster(j("orthomosaic.tif"),
                                       source_data=Metashape.OrthomosaicData)
                    products["orthomosaic_tif"] = j("orthomosaic.tif")
        except Exception as e:                                  # noqa: BLE001
            log("DEM/ortho stage failed: %r" % e)

    # ---- report + camera poses ----
    try:
        cams = {c.label: ([list(c.transform.translation())] if c.transform else None)
                for c in chunk.cameras}
        with open(j("cameras.json"), "w") as fh:
            json.dump(cams, fh)
        products["cameras_json"] = j("cameras.json")
    except Exception:                                          # noqa: BLE001
        pass
    if opt["make_report"]:
        try:
            chunk.exportReport(j("report.pdf"))
            products["report_pdf"] = j("report.pdf")
        except Exception as e:                                 # noqa: BLE001
            log("report export failed: %r" % e)

    return {"label": spec["label"], "run_dir": run_dir,
            "cameras_total": len(chunk.cameras), "cameras_aligned": aligned,
            "products": products}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    global _LOG
    if len(sys.argv) < 2:
        print("usage: metashape.exe -r metashape_worker.py <params.json>", flush=True)
        sys.exit(2)
    params = json.load(open(sys.argv[1], "r"))
    result_path = params["result_path"]
    result = {"ok": False, "version": Metashape.app.version, "chunks": []}
    try:
        _LOG = open(params["log_path"], "w")
    except OSError:
        _LOG = None

    try:
        opt = params["options"]
        t0 = time.time()
        log("Metashape %s — %d chunk(s)" % (Metashape.app.version, len(params["chunks"])))

        doc = Metashape.Document()
        doc.save(params["project_psx"])         # create the .psx up front

        for spec in params["chunks"]:
            ch = doc.addChunk()
            ch.label = spec["label"]
            try:
                result["chunks"].append(process_chunk(ch, spec, opt))
            except Exception as e:                             # noqa: BLE001
                log("CHUNK '%s' FAILED: %r" % (spec["label"], e))
                log(traceback.format_exc())
                result["chunks"].append({
                    "label": spec["label"], "run_dir": spec["run_dir"],
                    "cameras_total": 0, "cameras_aligned": 0,
                    "products": {}, "error": repr(e)})
            if opt["save_project"]:
                doc.save()

        result["ok"] = True
        log("done in %.1f s" % (time.time() - t0))
    except Exception as e:                                     # noqa: BLE001
        result["error"] = repr(e)
        log("FATAL: %r" % e)
        log(traceback.format_exc())
    finally:
        with open(result_path, "w") as fh:
            json.dump(result, fh, indent=1)
        if _LOG:
            _LOG.close()


if __name__ == "__main__":
    main()
