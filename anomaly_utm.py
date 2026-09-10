#!/usr/bin/env python3
"""Reproject the anomaly catalog's lat/lon GeoJSON layers into the survey's
UTM frame (EPSG:32613), producing the files every map/figure consumer expects:

  survey/anomaly/anomalous_sites_utm.geojson   (Points, catalog properties kept)
  survey/anomaly/anomaly_segments_utm.geojson  (LineStrings, per-window)

Sources: survey/anomaly/anomalous_sites.geojson and the catalog's qgis/
*_anomaly_segments.geojson (both EPSG:4326).

Qt-free.  build_utm_layers(workspace_dir) -> (sites_path, segments_path)
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

from pyproj import Transformer

CRS = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32613"}}
_T = Transformer.from_crs("EPSG:4326", "EPSG:32613", always_xy=True)


def _tx_geom(geom: dict) -> dict:
    if geom["type"] == "Point":
        x, y = _T.transform(*geom["coordinates"][:2])
        return {"type": "Point", "coordinates": [round(x, 3), round(y, 3)]}
    if geom["type"] == "LineString":
        return {"type": "LineString", "coordinates": [
            [round(v, 3) for v in _T.transform(c[0], c[1])]
            for c in geom["coordinates"]]}
    raise ValueError(f"unsupported geometry: {geom['type']}")


def _convert(src: Path, dst: Path, name: str) -> str:
    d = json.loads(src.read_text())
    out = {"type": "FeatureCollection", "name": name, "crs": CRS,
           "features": [{"type": "Feature", "geometry": _tx_geom(f["geometry"]),
                         "properties": f.get("properties", {})}
                        for f in d["features"]]}
    dst.write_text(json.dumps(out))
    return str(dst)


def build_utm_layers(workspace_dir):
    A = Path(workspace_dir) / "survey" / "anomaly"
    sites = _convert(A / "anomalous_sites.geojson",
                     A / "anomalous_sites_utm.geojson", "anomalous_sites_utm")
    seg_srcs = sorted(glob.glob(str(A / "qgis" / "*_anomaly_segments.geojson")))
    if not seg_srcs:
        raise FileNotFoundError("no *_anomaly_segments.geojson under survey/anomaly/qgis")
    segs = _convert(Path(seg_srcs[0]),
                    A / "anomaly_segments_utm.geojson", "anomaly_segments_utm")
    return sites, segs


if __name__ == "__main__":
    print(build_utm_layers(sys.argv[1] if len(sys.argv) > 1 else "."))
