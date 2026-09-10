#!/usr/bin/env python3
"""Add a "Brightness" layer group to a dive's QGIS project: per-frame corrected
scene brightness as graduated cividis dots plus bright-pixel-cover rings, from
the GeoJSON layers exported by brightness_layers.py.

The group is inserted after "Anomalies" in the layer tree so anomaly strokes
draw above the dots.  The matching .qgz alongside the .qgs is re-zipped.

Qt-free (pure XML surgery).  add_brightness_group(qgs_path, n_classes=5) -> str
"""
from __future__ import annotations
import json
import sys
import uuid
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CIVIDIS = ["#00204d", "#31446b", "#666970", "#a69d75", "#ffea46"]


def _sym(idx: int, color: str, size: str, outline: str = "35,56,70,255",
         no_fill: bool = False) -> ET.Element:
    sym = ET.Element("symbol", {"type": "marker", "name": str(idx),
                                "alpha": "1", "clip_to_extent": "1"})
    layer = ET.SubElement(sym, "layer", {"class": "SimpleMarker", "enabled": "1",
                                         "pass": "0", "locked": "0"})
    props = {
        "name": "circle", "size": size, "size_unit": "MM",
        "color": ("0,0,0,0" if no_fill else color),
        "outline_color": (color if no_fill else outline),
        "outline_width": ("0.6" if no_fill else "0.2"),
        "outline_width_unit": "MM", "outline_style": "solid",
    }
    for k, v in props.items():
        ET.SubElement(layer, "Option" if False else "prop", {"k": k, "v": v})
    return sym


def _hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},255"


def _vector_maplayer(lid: str, name: str, source: str, srs_node) -> ET.Element:
    ml = ET.Element("maplayer", {"type": "vector", "geometry": "Point",
                                 "hasScaleBasedVisibilityFlag": "0"})
    ET.SubElement(ml, "id").text = lid
    ET.SubElement(ml, "datasource").text = source
    ET.SubElement(ml, "layername").text = name
    ET.SubElement(ml, "provider", {"encoding": "UTF-8"}).text = "ogr"
    if srs_node is not None:
        import copy
        ml.append(copy.deepcopy(srs_node))
    return ml


def add_brightness_group(qgs_path, n_classes: int = 5) -> str:
    qgs_path = Path(qgs_path)
    ws = qgs_path.parent.parent.parent          # <ws>/survey/qgis/x.qgs
    fb = ws / "survey" / "nav_trackline" / "frame_brightness.geojson"
    bc = ws / "survey" / "nav_trackline" / "bright_cover.geojson"
    if not fb.is_file():
        raise FileNotFoundError(fb)

    vals = sorted(f["properties"]["bright"]
                  for f in json.load(open(fb))["features"])
    breaks = list(np.quantile(vals, np.linspace(0, 1, n_classes + 1)))

    tree = ET.parse(qgs_path)
    root = tree.getroot()
    lt_root = root.find("layer-tree-group")
    groups = lt_root.findall("layer-tree-group")
    anom_i = next((i for i, g in enumerate(groups) if g.get("name") == "Anomalies"), 0)

    # take the CRS node from any existing vector maplayer
    srs_node = None
    for ml in root.find("projectlayers").findall("maplayer"):
        if ml.get("type") == "vector" and ml.find("srs") is not None:
            srs_node = ml.find("srs")
            break

    fb_id = f"Frame_Brightness_{uuid.uuid4().hex[:8]}"
    bc_id = f"Bright_Cover_{uuid.uuid4().hex[:8]}"

    grp = ET.Element("layer-tree-group", {"name": "Brightness", "expanded": "0",
                                          "checked": "Qt::Checked"})
    for lid, nm in ((bc_id, "Bright Cover >2%"), (fb_id, "Frame Brightness")):
        ET.SubElement(grp, "layer-tree-layer",
                      {"id": lid, "name": nm, "expanded": "1",
                       "checked": "Qt::Checked", "providerKey": "ogr",
                       "source": f"../nav_trackline/{'bright_cover' if lid == bc_id else 'frame_brightness'}.geojson"})
    # insert AFTER Anomalies so anomaly strokes render above
    lt_children = list(lt_root)
    idx = lt_children.index(groups[anom_i]) + 1
    lt_root.insert(idx, grp)

    pl = root.find("projectlayers")

    ml = _vector_maplayer(fb_id, "Frame Brightness",
                          "../nav_trackline/frame_brightness.geojson", srs_node)
    r2 = ET.SubElement(ml, "renderer-v2",
                       {"type": "graduatedSymbol", "attr": "bright",
                        "graduatedMethod": "GraduatedColor"})
    rngs = ET.SubElement(r2, "ranges")
    syms = ET.SubElement(r2, "symbols")
    for i in range(n_classes):
        ET.SubElement(rngs, "range",
                      {"symbol": str(i), "lower": f"{breaks[i]:.2f}",
                       "upper": f"{breaks[i+1]:.2f}", "render": "true",
                       "label": f"{breaks[i]:.0f}–{breaks[i+1]:.0f}"})
        syms.append(_sym(i, _hex_rgb(CIVIDIS[i]), "1.7"))
    pl.append(ml)

    ml2 = _vector_maplayer(bc_id, "Bright Cover >2%",
                           "../nav_trackline/bright_cover.geojson", srs_node)
    r2b = ET.SubElement(ml2, "renderer-v2", {"type": "singleSymbol"})
    symsb = ET.SubElement(r2b, "symbols")
    symsb.append(_sym(0, _hex_rgb("#f2c94c"), "3.0", no_fill=True))
    pl.append(ml2)

    tree.write(qgs_path, encoding="utf-8", xml_declaration=True)

    qgz = next(qgs_path.parent.glob("*.qgz"), None)
    if qgz:
        with zipfile.ZipFile(qgz, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(qgs_path, qgs_path.name)
    return str(qgs_path)


if __name__ == "__main__":
    print(add_brightness_group(sys.argv[1]))
