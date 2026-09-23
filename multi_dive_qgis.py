#!/usr/bin/env python3
"""Combined QGIS project overlaying every dive's trackline, anomaly windows
and ranked sites in one UTM frame — the cross-dive companion to
multi_dive_map.py, built on qgis_project's own layer writer.

One layer-tree group per dive (trackline + tier-categorized windows + sites),
all in EPSG:32613 so the dives register without reprojection.

build_cross_dive_project(dives, out_path=None) -> .qgs path (matching .qgz
written alongside).  Qt-free.
"""
from __future__ import annotations
import sys
import zipfile
from pathlib import Path

from qgis_project import _Layer, _write_qgs_xml, _nonempty

ROOT = Path("/mnt/f/EPR_2026_PROCESSED")


def build_cross_dive_project(dives, out_path=None, log=print) -> str:
    layers = []
    for dive in dives:
        ws = ROOT / f"{dive}_down.eprproj" / "survey"
        trk = ws / "nav_trackline" / "trackline.geojson"
        seg = ws / "anomaly" / "anomaly_segments_utm.geojson"
        sites = ws / "anomaly" / "anomalous_sites_utm.geojson"
        added = 0
        if _nonempty(trk):
            layers.append(_Layer("vector", f"{dive} Trackline", trk, dive,
                                 1, None, epsg=32613, style="trackline",
                                 geometry="Line"))
            added += 1
        if _nonempty(seg):
            layers.append(_Layer("vector", f"{dive} Anomaly Windows", seg, dive,
                                 1, None, epsg=32613, style="anomaly_segments",
                                 geometry="Line"))
            added += 1
        if _nonempty(sites):
            layers.append(_Layer("vector", f"{dive} Sites", sites, dive,
                                 1, None, epsg=32613, style="anomaly_sites",
                                 geometry="Point"))
            added += 1
        log(f"{dive}: {added} layers")
    if not layers:
        raise RuntimeError("no layers found for any dive")

    out = Path(out_path or ROOT / "cross_dive_qgis" / "EPR_all_dives_anomalies.qgs")
    out.parent.mkdir(parents=True, exist_ok=True)
    xml = _write_qgs_xml(layers, out, "EPR 9°N — all-dive gas anomaly overlay")
    out.write_text(xml, encoding="utf-8")
    qgz = out.with_name(out.stem + "_QGIS_project.qgz")
    with zipfile.ZipFile(qgz, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out, out.name)
    log(f"wrote {out} (+{qgz.name}) with {len(layers)} layers")
    return str(out)


if __name__ == "__main__":
    dives = sys.argv[1:] or ["J1754", "J1755", "J1756", "J1758", "J1759", "J1760", "J1761"]
    build_cross_dive_project(dives)
