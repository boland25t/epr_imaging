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
    try:
        with open(nav_csv, "r", newline="") as fh:
            rd = csv.DictReader(fh)
            cols = {c.lower(): c for c in (rd.fieldnames or [])}
            tcol = cols.get("timestamp_iso") or cols.get("timestamp")
            latc = cols.get("lat") or cols.get("latitude")
            lonc = cols.get("lon") or cols.get("longitude")
            altc = cols.get("alt") or cols.get("altitude") or cols.get("depth")
            if not (tcol and latc and lonc):
                log("nav csv lacks timestamp/lat/lon columns — skipping georeference")
                return 0
            for r in rd:
                dt = parse_iso(r.get(tcol))
                if dt is None:
                    continue
                try:
                    rows.append((dt.replace(tzinfo=timezone.utc).timestamp(),
                                 float(r[latc]), float(r[lonc]),
                                 float(r[altc]) if altc and r.get(altc) not in (None, "") else 0.0))
                except (ValueError, TypeError):
                    continue
    except OSError as e:
        log("could not read nav csv (%s) — skipping georeference" % e)
        return 0
    if not rows:
        return 0
    rows.sort()
    times = [r[0] for r in rows]

    import bisect
    frame_rx = re.compile(r"(\d{8}T\d{6})")
    seeded = 0
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
        _, lat, lon, alt = rows[i]
        cam.reference.location = Metashape.Vector([lon, lat, alt])
        cam.reference.accuracy = Metashape.Vector([acc_h, acc_h, acc_v])
        cam.reference.enabled = True
        seeded += 1
    if seeded:
        chunk.crs = Metashape.CoordinateSystem("EPSG::4326")
        try:
            chunk.updateTransform()
        except Exception:                                       # noqa: BLE001
            pass
    log("georeference: seeded %d/%d cameras" % (seeded, len(chunk.cameras)))
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

    log("matchPhotos + alignCameras (accuracy=%s)" % opt["align_accuracy"])
    _call(chunk.matchPhotos,
          downscale={"Highest": 0, "High": 1, "Medium": 2, "Low": 4,
                     "Lowest": 8}.get(opt["align_accuracy"], 1),
          generic_preselection=opt["generic_preselect"],
          reference_preselection=False,
          keypoint_limit=opt["key_point_limit"],
          tiepoint_limit=opt["tie_point_limit"])
    _call(chunk.alignCameras, adaptive_fitting=opt["adaptive_fitting"])

    aligned = sum(1 for c in chunk.cameras if c.transform is not None)
    log("aligned %d/%d cameras" % (aligned, len(chunk.cameras)))

    if spec.get("nav_csv") and opt["use_nav_reference"]:
        _seed_reference(chunk, spec["nav_csv"], opt["nav_accuracy_h"], opt["nav_accuracy_v"])

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
              filter_mode=_depth_filter(opt["depth_filter"]))
        _call(chunk.buildPointCloud)
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
            _call(chunk.buildDepthMaps, downscale=4)
            _call(chunk.buildPointCloud)
        log("buildModel (surface=%s, faces=%s, source=%s)"
            % (opt["mesh_surface"], opt["mesh_faces"], opt["mesh_source"]))
        _call(chunk.buildModel, surface_type=_surface(opt["mesh_surface"]),
              face_count=_faces(opt["mesh_faces"]), source_data=src,
              vertex_colors=opt["mesh_vertex_colors"])
        have_mesh = chunk.model is not None
        if have_mesh:
            if opt["build_texture"]:
                _call(chunk.buildUV)
                _call(chunk.buildTexture, blending_mode=_blending(opt["texture_blending"]),
                      texture_size=opt["texture_size"], fill_holes=opt["texture_fill_holes"])
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
            _call(chunk.buildDem, source_data=surface)
            if chunk.elevation is not None and opt["export_dem"]:
                chunk.exportRaster(j("dem.tif"), source_data=Metashape.ElevationData)
                products["dem_tif"] = j("dem.tif")
            if opt["build_orthomosaic"] and chunk.elevation is not None:
                surf_for_ortho = (Metashape.ModelData if have_mesh
                                  else Metashape.ElevationData)
                log("buildOrthomosaic (surface=%s)"
                    % ("mesh" if have_mesh else "DEM"))
                _call(chunk.buildOrthomosaic, surface_data=surf_for_ortho,
                      blending_mode=_blending(opt["texture_blending"]),
                      fill_holes=True)
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
