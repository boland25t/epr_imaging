#!/usr/bin/env python3
"""Turn BIIGLE annotation report CSVs back into survey products: a UTM GeoJSON
layer, a gas-joined frame table, and an optional QGIS "Annotations" group.

BIIGLE image-annotation CSV report schema (confirmed 2026-09 against
https://biigle.de/manual/tutorials/reports/reports-schema and the generator
source, biigle/core app/Services/Reports/Volumes/ImageAnnotations/
CsvReportGenerator.php on GitHub):

    annotation_label_id, label_id, label_name, label_hierarchy, user_id,
    firstname, lastname, image_id, filename, image_longitude, image_latitude,
    shape_id, shape_name, points, attributes, annotation_id, created_at

  * "points" is a JSON array of alternating x/y pixel values
    "[x1,y1,x2,y2,...]"; for Circle the third value is the radius "[x,y,r]".
  * "attributes" (JSON object of image metadata, e.g. width/height) is
    OPTIONAL — the user may hide it when requesting the report.
  * Uncertainties: Ellipse points are four axis-endpoint vertices and are
    treated here as a plain polygon; a "confidence" column is not part of the
    stock report but is passed through if present.

Two coordinate contexts, auto-detected per row from "filename":

  FRAME  the filename matches a video frame in the workspace manifests
         (segment_*/interp.csv via frame_color_analysis.load_frame_manifest).
         The annotation becomes a POINT at the frame's nav easting/northing,
         joined with unix_time/alt and the CO2/CH4/O2 gas channels.
         Sub-frame pixel->world mapping is NOT attempted for frames — the
         whole frame is georeferenced to one nav fix.
  ORTHO  the filename matches an orthomosaic tile exported by
         biigle_export_prep (<seg>_<chunk>_ortho.*) with a sidecar
         survey/biigle/orthos/<name>.georef.json — both <stem>.georef.json
         and <full filename>.georef.json are probed (export_prep writes
         the latter, e.g. seg01_chunk_01_ortho.tif.georef.json) — holding
         {"transform": [a,b,c,d,e,f], "crs": "EPSG:32613", ...}.  Every
         pixel vertex is affine-transformed to UTM; circles become 24-gon
         polygons, points stay points, lines/polygons keep their vertices.

Outputs under <workspace>/survey/annotation/:
    biigle_annotations_utm.geojson   EPSG:32613, crs member as anomaly_utm.py
    biigle_frame_annotations.csv     frame-context join table

Qt-free.  build_products(workspace_dir, report_path) -> dict of paths+counts.
CLI: python3 biigle_ingest.py <workspace> <report.csv|zip> [--qgs path]
"""
from __future__ import annotations
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import pandas as pd

CRS = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32613"}}
CIRCLE_VERTS = 24
REQUIRED = ["label_name", "filename", "shape_name", "points"]
OPTIONAL = ["annotation_label_id", "label_id", "label_hierarchy", "user_id",
            "firstname", "lastname", "image_id", "image_longitude",
            "image_latitude", "shape_id", "attributes", "annotation_id",
            "created_at", "confidence"]


def _parse_points(v):
    """'[x1,y1,...]' -> list of floats (json first, bare-split fallback)."""
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    s = str(v).strip()
    try:
        return [float(x) for x in json.loads(s)]
    except (json.JSONDecodeError, TypeError, ValueError):
        return [float(x) for x in s.strip("[]").split(",") if x.strip()]


def load_report(csv_path) -> pd.DataFrame:
    """Parse a BIIGLE image-annotation CSV report (or a .zip containing one).

    The points column is decoded to a float list; missing optional columns
    are added as None so downstream code can rely on the full schema."""
    p = Path(csv_path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise FileNotFoundError(f"no .csv inside {p}")
            df = pd.read_csv(io.BytesIO(z.read(names[0])))
    else:
        df = pd.read_csv(p)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"report missing required columns: {missing}")
    for c in OPTIONAL:
        if c not in df.columns:
            df[c] = None
    df["points"] = df["points"].map(_parse_points)
    return df


def _georef_sidecar(workspace_dir, filename) -> Path | None:
    d = Path(workspace_dir) / "survey" / "biigle" / "orthos"
    for cand in (Path(filename).stem, Path(filename).name):
        sc = d / f"{cand}.georef.json"
        if sc.is_file():
            return sc
    return None


def _px_to_utm(t, pts):
    """Affine [a,b,c,d,e,f] applied to [(x,y),...] pixel points."""
    a, b, c, d, e, f = t
    return [[round(a * x + b * y + c, 3), round(d * x + e * y + f, 3)]
            for x, y in pts]


def _ortho_geometry(shape: str, pts: list, transform: list) -> dict | None:
    """Pixel-space BIIGLE shape -> UTM GeoJSON geometry via the affine."""
    s = (shape or "").lower()
    if s == "circle" and len(pts) >= 3:
        cx, cy, r = pts[0], pts[1], pts[2]
        ring = [(cx + r * math.cos(2 * math.pi * i / CIRCLE_VERTS),
                 cy + r * math.sin(2 * math.pi * i / CIRCLE_VERTS))
                for i in range(CIRCLE_VERTS)]
        ring.append(ring[0])
        return {"type": "Polygon", "coordinates": [_px_to_utm(transform, ring)]}
    xy = list(zip(pts[0::2], pts[1::2]))
    if not xy:
        return None
    if s == "point" or len(xy) == 1:
        return {"type": "Point", "coordinates": _px_to_utm(transform, xy)[0]}
    if s in ("line", "linestring", "polyline"):
        return {"type": "LineString", "coordinates": _px_to_utm(transform, xy)}
    # polygon / rectangle / ellipse (axis endpoints as plain vertices)
    if xy[0] != xy[-1]:
        xy.append(xy[0])
    return {"type": "Polygon", "coordinates": [_px_to_utm(transform, xy)]}


def _base_props(row) -> dict:
    fn, ln = row.get("firstname"), row.get("lastname")
    annotator = " ".join(str(v) for v in (fn, ln) if pd.notna(v) and v) or None
    props = {"label_name": row.get("label_name"),
             "label_hierarchy": row.get("label_hierarchy"),
             "shape_name": row.get("shape_name"),
             "source_image": row.get("filename"),
             "annotator": annotator}
    if pd.notna(row.get("confidence")):
        props["confidence"] = row["confidence"]
    return {k: (None if (v is None or (isinstance(v, float) and pd.isna(v)))
                else v) for k, v in props.items()}


def build_products(workspace_dir, report_path, out_dir=None) -> dict:
    """Ingest a BIIGLE report into <workspace>/survey/annotation/ products."""
    from frame_color_analysis import load_frame_manifest

    ws = Path(workspace_dir)
    out = Path(out_dir) if out_dir else ws / "survey" / "annotation"
    out.mkdir(parents=True, exist_ok=True)

    rep = load_report(report_path)
    mf = load_frame_manifest(str(ws))
    mf = mf.assign(basename=[Path(f).name for f in mf.fn])
    frames = {r.basename: r for r in mf.itertuples(index=False)}

    features, frame_rows, skipped = [], [], []
    georefs: dict[str, dict | None] = {}
    for _, row in rep.iterrows():
        base = Path(str(row["filename"])).name
        if base in frames:                                # FRAME context
            fr = frames[base]
            props = _base_props(row)
            props.update({"context": "frame",
                          "CH4": round(float(fr.CH4), 4),
                          "CO2": round(float(fr.CO2), 4),
                          "O2": round(float(fr.O2), 4),
                          "alt": round(float(fr.alt), 3),
                          "unix_time": float(fr.unix_time)})
            features.append({"type": "Feature",
                             "geometry": {"type": "Point", "coordinates":
                                          [round(float(fr.easting), 3),
                                           round(float(fr.northing), 3)]},
                             "properties": props})
            frame_rows.append({**props, "easting": round(float(fr.easting), 3),
                               "northing": round(float(fr.northing), 3),
                               "seg": fr.seg, "points": json.dumps(row["points"])})
            continue
        if base not in georefs:                           # ORTHO context
            sc = _georef_sidecar(ws, base)
            georefs[base] = json.loads(sc.read_text()) if sc else None
        g = georefs[base]
        geom = (_ortho_geometry(row["shape_name"], row["points"],
                                g["transform"]) if g else None)
        if geom is None:
            skipped.append(base)
            continue
        props = _base_props(row)
        props["context"] = "ortho"
        features.append({"type": "Feature", "geometry": geom,
                         "properties": props})

    gj = out / "biigle_annotations_utm.geojson"
    gj.write_text(json.dumps({"type": "FeatureCollection",
                              "name": "biigle_annotations_utm", "crs": CRS,
                              "features": features}))
    fcsv = out / "biigle_frame_annotations.csv"
    pd.DataFrame(frame_rows).to_csv(fcsv, index=False)

    counts = {}
    for f in features:
        counts[f["properties"]["label_name"]] = \
            counts.get(f["properties"]["label_name"], 0) + 1
    print(f"{len(features)} annotations ingested "
          f"({len(frame_rows)} frame-context, "
          f"{len(features) - len(frame_rows)} ortho-context, "
          f"{len(skipped)} skipped — no manifest frame or georef sidecar)")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {n}")
    return {"geojson": str(gj), "frame_csv": str(fcsv), "counts": counts,
            "n_frame": len(frame_rows),
            "n_ortho": len(features) - len(frame_rows),
            "n_skipped": len(skipped)}


PALETTE = ["#e91e63", "#00bcd4", "#ff9800", "#8bc34a", "#9c27b0",
           "#ffeb3b", "#3f51b5", "#795548"]


def add_qgis_group(qgs_path) -> str:
    """Append an "Annotations" group pointing at the ingested GeoJSON,
    categorized by label_name.  Reuses qgis_brightness helpers; inserted
    after Brightness (else after Anomalies).  Non-point ortho shapes render
    with QGIS defaults — the marker symbology targets the point features."""
    import uuid
    import xml.etree.ElementTree as ET

    from qgis_brightness import _hex_rgb, _sym, _vector_maplayer

    qgs_path = Path(qgs_path)
    ws = qgs_path.parent.parent.parent            # <ws>/survey/qgis/x.qgs
    gj = ws / "survey" / "annotation" / "biigle_annotations_utm.geojson"
    if not gj.is_file():
        raise FileNotFoundError(gj)
    labels = sorted({f["properties"].get("label_name") or ""
                     for f in json.load(open(gj))["features"]})

    tree = ET.parse(qgs_path)
    root = tree.getroot()
    lt_root = root.find("layer-tree-group")
    groups = lt_root.findall("layer-tree-group")
    if any(g.get("name") == "Annotations" for g in groups):
        return str(qgs_path)                      # already present: no-op
    anchor = next((g for g in groups if g.get("name") == "Brightness"),
                  next((g for g in groups if g.get("name") == "Anomalies"),
                       groups[0] if groups else None))

    srs_node = None
    for ml in root.find("projectlayers").findall("maplayer"):
        if ml.get("type") == "vector" and ml.find("srs") is not None:
            srs_node = ml.find("srs")
            break

    lid = f"BIIGLE_Annotations_{uuid.uuid4().hex[:8]}"
    src = "../annotation/biigle_annotations_utm.geojson"
    grp = ET.Element("layer-tree-group", {"name": "Annotations",
                                          "expanded": "0",
                                          "checked": "Qt::Checked"})
    ET.SubElement(grp, "layer-tree-layer",
                  {"id": lid, "name": "BIIGLE Annotations", "expanded": "1",
                   "checked": "Qt::Checked", "providerKey": "ogr",
                   "source": src})
    lt_children = list(lt_root)
    idx = lt_children.index(anchor) + 1 if anchor is not None else 0
    lt_root.insert(idx, grp)

    ml = _vector_maplayer(lid, "BIIGLE Annotations", src, srs_node)
    r2 = ET.SubElement(ml, "renderer-v2", {"type": "categorizedSymbol",
                                           "attr": "label_name"})
    cats = ET.SubElement(r2, "categories")
    syms = ET.SubElement(r2, "symbols")
    for i, lab in enumerate(labels):
        ET.SubElement(cats, "category", {"symbol": str(i), "value": lab,
                                         "render": "true", "label": lab})
        syms.append(_sym(i, _hex_rgb(PALETTE[i % len(PALETTE)]), "2.4"))
    root.find("projectlayers").append(ml)

    tree.write(qgs_path, encoding="utf-8", xml_declaration=True)
    qgz = next(qgs_path.parent.glob("*.qgz"), None)
    if qgz:
        with zipfile.ZipFile(qgz, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(qgs_path, qgs_path.name)
    return str(qgs_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        sys.exit("usage: biigle_ingest.py <workspace> <report.csv|zip> "
                 "[--qgs path]")
    print(build_products(args[0], args[1]))
    if "--qgs" in sys.argv:
        print(add_qgis_group(sys.argv[sys.argv.index("--qgs") + 1]))
