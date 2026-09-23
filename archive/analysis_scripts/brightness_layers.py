#!/usr/bin/env python3
"""Export the per-frame corrected-brightness record as GIS point layers (UTM
32613) so the imagery measurement can be fused with the anomaly layers in QGIS:

  survey/nav_trackline/frame_brightness.geojson  — every analysed transit frame
      (props: bright, white, CH4, CO2, O2, alt, seg, utc)
  survey/nav_trackline/bright_cover.geojson      — frames with bright-pixel
      cover > 2 %, for ring highlighting

Qt-free.  export_layers(workspace_dir) -> (frames_path, cover_path)
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CRS = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32613"}}


def export_layers(workspace_dir):
    B = str(workspace_dir)
    d = pd.read_csv(f"{B}/survey/frame_color_metrics.csv")

    def feat(r):
        return {"type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(r.E, 3), round(r.N, 3)]},
                "properties": {
                    "bright": round(float(r.bright), 1),
                    "white_pct": round(float(r.white) * 100, 1),
                    "CH4": round(float(r.CH4), 2), "CO2": round(float(r.CO2), 1),
                    "O2": round(float(r.O2), 2), "alt_m": round(float(r.alt), 1),
                    "seg": r.seg,
                    "utc": datetime.fromtimestamp(r.t, timezone.utc).strftime("%H:%M:%S")}}
    frames = {"type": "FeatureCollection", "name": "frame_brightness", "crs": CRS,
              "features": [feat(r) for r in d.itertuples(index=False)]}
    cover = {"type": "FeatureCollection", "name": "bright_cover", "crs": CRS,
             "features": [feat(r) for r in d.itertuples(index=False) if r.white > 0.02]}
    out1 = Path(B) / "survey" / "nav_trackline" / "frame_brightness.geojson"
    out2 = Path(B) / "survey" / "nav_trackline" / "bright_cover.geojson"
    out1.write_text(json.dumps(frames))
    out2.write_text(json.dumps(cover))
    return str(out1), str(out2)


if __name__ == "__main__":
    print(export_layers(sys.argv[1] if len(sys.argv) > 1 else "."))
