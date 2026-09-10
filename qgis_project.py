"""
qgis_project.py — assemble a QGIS project (.qgs + zipped .qgz) for an EPR dive.

Given an ``.eprproj`` workspace (or a legacy workspace directory), this collects
every georeferenced raster product produced by the pipeline — orthomosaics, DEMs,
the navigation depth raster and the interpolated 2-D sensor rasters — plus a
trackline vector built from the master interpolation, and writes a portable QGIS
3.x project the user can open and export from directly.

Everything is UTM zone 13N (EPSG:32613).  Datasources are stored as paths relative
to the ``.qgs`` file so the project stays valid when the ``survey/`` tree is shipped
alongside the ``.qgz`` archive.

Design notes
------------
* Qt-free and importable.  The public entry point is
  ``build_qgis_project(workspace_dir, out_path=None) -> str`` and there is a
  ``__main__`` CLI.
* If the QGIS Python API (``qgis.core``) is importable *and* can run headless it is
  used to build and ``write()`` the project.  On virtually every machine here it is
  not available, so the project XML is written directly — the ``.qgs`` format is a
  documented QGIS 3.x XML document.
* The existing ``build_anomaly_site_catalog.py`` does *not* generate a ``.qgs``
  project of its own; it emits GeoJSON layers plus a ``.qml`` style meant to be
  loaded into a pre-existing project.  So there is no project-XML template to reuse,
  but the relative-path / GeoJSON-LineString / EPSG conventions here match it.

Only files created: ``<workspace>/survey/nav_trackline/trackline.geojson`` and the
QGIS outputs under ``<workspace>/survey/qgis/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
from xml.sax.saxutils import escape, quoteattr

try:
    import rasterio  # extent + band count per raster
except Exception:  # pragma: no cover - rasterio is expected to be present
    rasterio = None

from workspace_paths import PathResolver

# --------------------------------------------------------------------------- #
# CRS constants — the whole dive is WGS 84 / UTM zone 13N.
# --------------------------------------------------------------------------- #
EPSG_CODE = 32613
EPSG_AUTHID = "EPSG:32613"
CRS_DESCRIPTION = "WGS 84 / UTM zone 13N"
CRS_PROJ4 = "+proj=utm +zone=13 +datum=WGS84 +units=m +no_defs"
CRS_WKT = (
    'PROJCRS["WGS 84 / UTM zone 13N",BASEGEOGCRS["WGS 84",'
    'DATUM["World Geodetic System 1984",'
    'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],'
    'ID["EPSG",4326]],CONVERSION["UTM zone 13N",'
    'METHOD["Transverse Mercator",ID["EPSG",9807]],'
    'PARAMETER["Latitude of natural origin",0],'
    'PARAMETER["Longitude of natural origin",-105],'
    'PARAMETER["Scale factor at natural origin",0.9996],'
    'PARAMETER["False easting",500000],'
    'PARAMETER["False northing",0]],'
    'CS[Cartesian,2],AXIS["easting",east,ORDER[1]],'
    'AXIS["northing",north,ORDER[2]],LENGTHUNIT["metre",1],ID["EPSG",32613]]'
)
# QGIS internal srsid for EPSG:32613 (stable across 3.x); QGIS re-resolves via authid
# if it disagrees, so an exact match is not required.
CRS_SRSID = 3126


def _spatialrefsys(indent: str = "      ") -> str:
    """A ``<spatialrefsys>`` block QGIS resolves to EPSG:32613."""
    return (
        f"{indent}<spatialrefsys nativeFormat=\"Wkt\">\n"
        f"{indent}  <wkt>{escape(CRS_WKT)}</wkt>\n"
        f"{indent}  <proj4>{escape(CRS_PROJ4)}</proj4>\n"
        f"{indent}  <srsid>{CRS_SRSID}</srsid>\n"
        f"{indent}  <srid>{EPSG_CODE}</srid>\n"
        f"{indent}  <authid>{EPSG_AUTHID}</authid>\n"
        f"{indent}  <description>{escape(CRS_DESCRIPTION)}</description>\n"
        f"{indent}  <projectionacronym>utm</projectionacronym>\n"
        f"{indent}  <ellipsoidacronym>EPSG:7030</ellipsoidacronym>\n"
        f"{indent}  <geographicflag>false</geographicflag>\n"
        f"{indent}</spatialrefsys>\n"
    )


# WGS 84 (EPSG:4326) block for the anomaly GeoJSON layers, which are stored in
# lat/lon and let QGIS reproject on-the-fly into the project's UTM 13N canvas.
CRS4326_WKT = (
    'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",'
    'MEMBER["World Geodetic System 1984 (Transit)"],'
    'MEMBER["World Geodetic System 1984 (G730)"],'
    'MEMBER["World Geodetic System 1984 (G873)"],'
    'MEMBER["World Geodetic System 1984 (G1150)"],'
    'MEMBER["World Geodetic System 1984 (G1674)"],'
    'MEMBER["World Geodetic System 1984 (G1762)"],'
    'MEMBER["World Geodetic System 1984 (G2139)"],'
    'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],'
    'ENSEMBLEACCURACY[2.0]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],'
    'CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,ORDER[1],'
    'ANGLEUNIT["degree",0.0174532925199433]],'
    'AXIS["geodetic longitude (Lon)",east,ORDER[2],'
    'ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]]'
)
CRS4326_PROJ4 = "+proj=longlat +datum=WGS84 +no_defs"


def _spatialrefsys_4326(indent: str = "      ") -> str:
    """A ``<spatialrefsys>`` block QGIS resolves to EPSG:4326 (WGS 84 lat/lon)."""
    return (
        f"{indent}<spatialrefsys nativeFormat=\"Wkt\">\n"
        f"{indent}  <wkt>{escape(CRS4326_WKT)}</wkt>\n"
        f"{indent}  <proj4>{escape(CRS4326_PROJ4)}</proj4>\n"
        f"{indent}  <srsid>3452</srsid>\n"
        f"{indent}  <srid>4326</srid>\n"
        f"{indent}  <authid>EPSG:4326</authid>\n"
        f"{indent}  <description>WGS 84</description>\n"
        f"{indent}  <projectionacronym>longlat</projectionacronym>\n"
        f"{indent}  <ellipsoidacronym>EPSG:7030</ellipsoidacronym>\n"
        f"{indent}  <geographicflag>true</geographicflag>\n"
        f"{indent}</spatialrefsys>\n"
    )


def _srs_block(epsg: int, indent: str) -> str:
    """Dispatch to the right ``<spatialrefsys>`` for a layer's native EPSG."""
    return _spatialrefsys_4326(indent) if epsg == 4326 else _spatialrefsys(indent)


# --------------------------------------------------------------------------- #
# Layer collection
# --------------------------------------------------------------------------- #
_RUN_RE = re.compile(r"run_(\d+)")


class _Layer:
    """One resolved layer bound for the project."""

    __slots__ = ("kind", "name", "path", "group", "bands", "extent", "layer_id",
                 "epsg", "style", "geometry")

    def __init__(self, kind: str, name: str, path: Path, group: str,
                 bands: int, extent: Optional[Tuple[float, float, float, float]],
                 epsg: int = EPSG_CODE, style: Optional[str] = None,
                 geometry: str = "Line"):
        self.kind = kind          # "raster" | "vector"
        self.name = name
        self.path = Path(path)
        self.group = group
        self.bands = bands
        self.extent = extent      # (xmin, ymin, xmax, ymax) or None
        self.epsg = epsg          # layer-native EPSG (32613 default; 4326 for anomalies)
        self.style = style        # None | "trackline" | "anomaly_segments" | "anomaly_sites"
        self.geometry = geometry  # vector geometry: "Line" | "Point"
        # Unique, stable id: slug + short hash of the absolute path.
        slug = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "layer"
        h = hashlib.md5(str(self.path).encode("utf-8")).hexdigest()[:13]
        self.layer_id = f"{slug}_{h}"


def _run_number(path: Path) -> int:
    """Highest ``run_NNN`` component in a path, or -1 if none."""
    best = -1
    for part in path.parts:
        m = _RUN_RE.fullmatch(part)
        if m:
            best = max(best, int(m.group(1)))
    return best


def _pick_latest(paths: List[Path]) -> Optional[Path]:
    """Choose the newest of several candidate product files.

    Prefers the highest ``run_NNN``; tie-broken by larger size then newer mtime.
    Products get re-run in place, leaving several ``run_*`` copies (and, from an
    older migration, doubled ``<product>/<product>/`` subtrees) — we want one.
    """
    if not paths:
        return None

    def key(p: Path):
        try:
            st = p.stat()
            size, mtime = st.st_size, st.st_mtime
        except OSError:
            size, mtime = 0, 0.0
        # Prefer shallower path (the non-doubled subtree) on further ties.
        return (_run_number(p), size, mtime, -len(p.parts))

    return max(paths, key=key)


def _raster_meta(path: Path) -> Tuple[int, Optional[Tuple[float, float, float, float]]]:
    """(band_count, extent) for a raster, guarded — (1, None) if unreadable."""
    if rasterio is None:
        return 1, None
    try:
        with rasterio.open(path) as ds:
            b = ds.bounds
            return int(ds.count), (float(b.left), float(b.bottom),
                                   float(b.right), float(b.top))
    except Exception:
        return 1, None


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _prefer_lite(tif: Path) -> Tuple[Path, bool]:
    """Return a downsampled ``*_lite.tif`` sibling if present, else the original.

    The lite orthomosaics are 3-band RGB with an in-file nodata=0; the lite DEMs
    keep their own nodata.  Both are lighter to render than the full products, so
    the project prefers them when they exist next to the canonical file.
    """
    lite = tif.with_name(tif.stem + "_lite" + tif.suffix)
    if _nonempty(lite):
        return lite, True
    return tif, False


def collect_layers(resolver: PathResolver, notes: List[str]) -> List[_Layer]:
    """Resolve every raster/vector layer for the project (guarded, order preserved)."""
    survey = resolver.survey_products()
    layers: List[_Layer] = []

    # -- Orthomosaics + DEMs -------------------------------------------------- #
    photo_root = resolver.photogrammetry_root(survey)
    ortho_n = dem_n = 0
    ortho_lite = dem_lite = 0
    if photo_root.is_dir():
        for tif in sorted(photo_root.rglob("orthomosaic.tif")):
            tif, used_lite = _prefer_lite(tif)
            if not _nonempty(tif):
                notes.append(f"skip empty orthomosaic: {tif}")
                continue
            seg_chunk = _seg_chunk_label(tif, photo_root)
            bands, ext = _raster_meta(tif)
            layers.append(_Layer("raster", f"Ortho {seg_chunk}", tif,
                                 "Orthomosaics", bands, ext))
            ortho_n += 1
            ortho_lite += 1 if used_lite else 0
        for tif in sorted(photo_root.rglob("dem.tif")):
            tif, used_lite = _prefer_lite(tif)
            if not _nonempty(tif):
                notes.append(f"skip empty dem: {tif}")
                continue
            seg_chunk = _seg_chunk_label(tif, photo_root)
            bands, ext = _raster_meta(tif)
            layers.append(_Layer("raster", f"DEM {seg_chunk}", tif,
                                 "DEMs", bands, ext))
            dem_n += 1
            dem_lite += 1 if used_lite else 0
    else:
        notes.append(f"no photogrammetry dir: {photo_root}")
    notes.append(f"orthomosaics: {ortho_n} ({ortho_lite} lite)")
    notes.append(f"dems: {dem_n} ({dem_lite} lite)")

    # -- Navigation depth raster (latest run) -------------------------------- #
    nav_root = survey / "nav_depth"
    nav_tifs = [p for p in nav_root.rglob("nav_depth.tif") if _nonempty(p)] \
        if nav_root.is_dir() else []
    nav_best = _pick_latest(nav_tifs)
    if nav_best is not None:
        bands, ext = _raster_meta(nav_best)
        layers.append(_Layer("raster", "Nav Depth", nav_best,
                             "Navigation", bands, ext))
        notes.append(f"nav_depth: 1 ({nav_best.relative_to(survey)})")
    else:
        notes.append("nav_depth: 0 (missing)")

    # -- Sensor 2-D rasters (latest run per channel) ------------------------- #
    sensor_root = survey / "sensor_2d"
    sensor_n = 0
    if sensor_root.is_dir():
        by_channel: dict[str, List[Path]] = {}
        for tif in sensor_root.rglob("*_2d.tif"):
            if not _nonempty(tif):
                continue
            channel = _sensor_channel(tif)
            by_channel.setdefault(channel, []).append(tif)
        for channel in sorted(by_channel):
            best = _pick_latest(by_channel[channel])
            if best is None:
                continue
            bands, ext = _raster_meta(best)
            layers.append(_Layer("raster", channel, best,
                                 "Sensors", bands, ext))
            sensor_n += 1
    else:
        notes.append(f"no sensor_2d dir: {sensor_root}")
    notes.append(f"sensors: {sensor_n}")

    return layers


def _detect_geojson_epsg(path: Path, notes: List[str]) -> int:
    """Best-effort EPSG for a GeoJSON: honour a ``crs`` member, else sniff coords.

    GeoJSON defaults to WGS 84 lon/lat when no ``crs`` member is present.  We
    also sanity-check the first coordinate magnitude — |x|<=180 and |y|<=90 is
    lon/lat (EPSG:4326); large positive values are projected UTM (EPSG:32613).
    """
    try:
        doc = json.loads(path.read_text())
    except Exception as exc:
        notes.append(f"anomaly: cannot parse {path.name} ({exc}); assuming EPSG:4326")
        return 4326

    crs = (doc.get("crs") or {}).get("properties", {}).get("name", "")
    if isinstance(crs, str):
        m = re.search(r"(?:EPSG:{1,2}|/)(\d{4,6})\b", crs)
        if m:
            code = int(m.group(1))
            if code in (4326, 32613):
                return code

    # Fall back to coordinate magnitude of the first feature.
    try:
        c = doc["features"][0]["geometry"]["coordinates"]
        while isinstance(c[0], list):
            c = c[0]
        x, y = float(c[0]), float(c[1])
        if abs(x) <= 180.0 and abs(y) <= 90.0:
            return 4326
        return 32613
    except Exception:
        return 4326


def _feature_count(path: Path) -> int:
    try:
        return len(json.loads(path.read_text()).get("features", []))
    except Exception:
        return 0


def collect_anomaly_layers(resolver: PathResolver, notes: List[str]) -> List[_Layer]:
    """Resolve the anomaly-window and anomalous-site vector layers (guarded)."""
    survey = resolver.survey_products()
    anomaly_root = survey / "anomaly"
    out: List[_Layer] = []
    if not anomaly_root.is_dir():
        notes.append(f"no anomaly dir: {anomaly_root}")
        return out

    segments = anomaly_root / "qgis" / "J1754_anomaly_segments.geojson"
    if _nonempty(segments):
        epsg = _detect_geojson_epsg(segments, notes)
        n = _feature_count(segments)
        out.append(_Layer("vector", "Anomaly Windows", segments, "Anomalies",
                          1, None, epsg=epsg, style="anomaly_segments",
                          geometry="Line"))
        notes.append(f"anomaly windows: {n} features (EPSG:{epsg})")
    else:
        notes.append(f"skip anomaly windows (missing): {segments}")

    sites = anomaly_root / "anomalous_sites.geojson"
    if _nonempty(sites):
        epsg = _detect_geojson_epsg(sites, notes)
        n = _feature_count(sites)
        out.append(_Layer("vector", "Anomalous Sites", sites, "Anomalies",
                          1, None, epsg=epsg, style="anomaly_sites",
                          geometry="Point"))
        notes.append(f"anomalous sites: {n} features (EPSG:{epsg})")
    else:
        notes.append(f"skip anomalous sites (missing): {sites}")

    return out


def _seg_chunk_label(tif: Path, photo_root: Path) -> str:
    """Human label 'segNN/chunk_CC' from an ortho/dem path."""
    try:
        rel = tif.relative_to(photo_root)
        parts = [p for p in rel.parts[:-1]]  # drop filename
        return "/".join(parts) if parts else tif.stem
    except ValueError:
        return tif.parent.name


def _sensor_channel(tif: Path) -> str:
    """Channel name for a sensor 2-D raster.

    Path is ``.../sensor_2d/<channel>/run_NNN/<channel>_2d.tif``; prefer stripping
    the ``_2d`` suffix off the filename, which preserves the channel's real name
    (including spaces, e.g. 'CH4 Concentration').
    """
    stem = tif.stem
    if stem.endswith("_2d"):
        return stem[:-3]
    return tif.parent.parent.name


# --------------------------------------------------------------------------- #
# Trackline vector from the master interpolation
# --------------------------------------------------------------------------- #
def build_trackline_geojson(resolver: PathResolver, notes: List[str]) -> Optional[Path]:
    """Write a UTM LineString GeoJSON of the dive track; return its path or None."""
    interp = resolver.interp_full()
    if not _nonempty(interp):
        notes.append(f"no interp_full.csv: {interp}")
        return None

    coords: List[List[float]] = []
    try:
        import pandas as pd
        df = pd.read_csv(interp, usecols=lambda c: c in ("easting", "northing"))
        if "easting" not in df.columns or "northing" not in df.columns:
            notes.append("interp_full.csv lacks easting/northing columns")
            return None
        for e, n in zip(df["easting"], df["northing"]):
            try:
                fe, fn = float(e), float(n)
            except (TypeError, ValueError):
                continue
            if fe == fe and fn == fn:  # not NaN
                coords.append([round(fe, 3), round(fn, 3)])
    except Exception as exc:  # pragma: no cover - defensive
        notes.append(f"failed reading interp_full.csv: {exc}")
        return None

    if len(coords) < 2:
        notes.append("interp_full.csv has < 2 valid easting/northing points")
        return None

    out_dir = resolver.survey_products() / "nav_trackline"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        notes.append(f"cannot create trackline dir: {exc}")
        return None
    out_path = out_dir / "trackline.geojson"

    feature_collection = {
        "type": "FeatureCollection",
        "name": "nav_trackline",
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:EPSG::{EPSG_CODE}"},
        },
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"name": "Nav Trackline", "point_count": len(coords)},
            }
        ],
    }
    try:
        out_path.write_text(json.dumps(feature_collection, separators=(",", ":")))
    except OSError as exc:
        notes.append(f"cannot write trackline.geojson: {exc}")
        return None
    notes.append(f"trackline: {len(coords)} vertices -> {out_path.name}")
    return out_path


# --------------------------------------------------------------------------- #
# .qgs XML assembly (hand-written; no Qt / no qgis.core required)
# --------------------------------------------------------------------------- #
def _rel(datasource: Path, qgs_path: Path) -> str:
    """POSIX relative path from the .qgs file's directory to a datasource."""
    rel = os.path.relpath(str(datasource), str(qgs_path.parent))
    rel = rel.replace(os.sep, "/")
    if not rel.startswith((".", "/")):
        rel = "./" + rel
    return rel


def _raster_maplayer(layer: _Layer, qgs_path: Path) -> str:
    ds = _rel(layer.path, qgs_path)
    ext = _extent_xml(layer.extent, "    ")
    if layer.bands >= 3:
        renderer = (
            '        <rasterrenderer type="multibandcolor" opacity="1" '
            'redBand="1" greenBand="2" blueBand="3" alphaBand="-1">\n'
            '          <redContrastEnhancement><algorithm>StretchToMinimumMaximum</algorithm></redContrastEnhancement>\n'
            '          <greenContrastEnhancement><algorithm>StretchToMinimumMaximum</algorithm></greenContrastEnhancement>\n'
            '          <blueContrastEnhancement><algorithm>StretchToMinimumMaximum</algorithm></blueContrastEnhancement>\n'
            '        </rasterrenderer>\n'
        )
    else:
        renderer = (
            '        <rasterrenderer type="singlebandgray" opacity="1" '
            'grayBand="1" gradient="BlackToWhite">\n'
            '          <contrastEnhancement><algorithm>StretchToMinimumMaximum</algorithm></contrastEnhancement>\n'
            '        </rasterrenderer>\n'
        )
    # Honour the source nodata value so nodata pixels render transparent
    # (the lite orthomosaics carry an in-file nodata=0 over their RGB bands).
    band_count = max(1, layer.bands)
    nodata = "      <noData>\n" + "".join(
        f'        <noDataList bandNo="{b}" useSrcNoData="1"/>\n'
        for b in range(1, band_count + 1)
    ) + "      </noData>\n"
    return (
        '    <maplayer type="raster" hasScaleBasedVisibilityFlag="0" '
        'autoRefreshEnabled="0" refreshOnNotifyEnabled="0">\n'
        f"      <id>{escape(layer.layer_id)}</id>\n"
        f"      <datasource>{escape(ds)}</datasource>\n"
        f"      <layername>{escape(layer.name)}</layername>\n"
        f"{ext}"
        "      <srs>\n" + _srs_block(layer.epsg, "        ") + "      </srs>\n"
        "      <provider>gdal</provider>\n"
        f"{nodata}"
        "      <pipe>\n"
        f"{renderer}"
        '        <brightnesscontrast brightness="0" contrast="0"/>\n'
        '        <huesaturation saturation="0" grayscaleMode="0"/>\n'
        '        <rasterresampler maxOversampling="2"/>\n'
        "      </pipe>\n"
        "      <blendMode>0</blendMode>\n"
        "    </maplayer>\n"
    )


def _vector_maplayer(layer: _Layer, qgs_path: Path) -> str:
    ds = _rel(layer.path, qgs_path)
    ext = _extent_xml(layer.extent, "    ")
    return (
        '    <maplayer type="vector" geometry="Line" '
        'hasScaleBasedVisibilityFlag="0" wkbType="LineString" '
        'autoRefreshEnabled="0" refreshOnNotifyEnabled="0">\n'
        f"      <id>{escape(layer.layer_id)}</id>\n"
        f"      <datasource>{escape(ds)}</datasource>\n"
        f"      <layername>{escape(layer.name)}</layername>\n"
        f"{ext}"
        "      <srs>\n" + _srs_block(layer.epsg, "        ") + "      </srs>\n"
        "      <provider encoding=\"UTF-8\">ogr</provider>\n"
        '      <renderer-v2 type="singleSymbol" forceraster="0" symbollevels="0">\n'
        "        <symbols>\n"
        '          <symbol type="line" name="0" alpha="1" clip_to_extent="1">\n'
        '            <layer class="SimpleLine" enabled="1">\n'
        '              <Option type="Map">\n'
        '                <Option name="line_color" type="QString" value="228,26,28,255"/>\n'
        '                <Option name="line_width" type="QString" value="0.3"/>\n'
        '                <Option name="line_width_unit" type="QString" value="MM"/>\n'
        '                <Option name="capstyle" type="QString" value="round"/>\n'
        '                <Option name="joinstyle" type="QString" value="round"/>\n'
        "              </Option>\n"
        "            </layer>\n"
        "          </symbol>\n"
        "        </symbols>\n"
        "      </renderer-v2>\n"
        "    </maplayer>\n"
    )


# --------------------------------------------------------------------------- #
# Anomaly layers — categorized line windows + prominent site points, both with
# labels.  The GeoJSONs are WGS 84 lat/lon (EPSG:4326); QGIS reprojects them
# on the fly into the project's UTM 13N canvas.
# --------------------------------------------------------------------------- #

# Confidence tier -> line colour (RGBA).  Matches the tier palette requested for
# the project (HIGH red / MODERATE orange / SCREEN yellow).
_ANOMALY_TIER_COLORS = {
    "HIGH": "215,25,28,255",
    "MODERATE": "253,174,97,255",
    "SCREEN": "255,255,150,255",
}


def _text_style_xml(field: str, indent: str) -> str:
    """A simple ~8pt black label with a white buffer over ``field``."""
    return (
        f'{indent}<text-style fontFamily="Sans Serif" fontSize="8" '
        'fontSizeUnit="Point" textColor="0,0,0,255" textOpacity="1" '
        f'fieldName="{escape(field)}" isExpression="0">\n'
        f'{indent}  <text-buffer bufferDraw="1" bufferSize="1" '
        'bufferSizeUnits="MM" bufferColor="255,255,255,255" '
        'bufferOpacity="1" bufferNoFill="0"/>\n'
        f"{indent}</text-style>\n"
    )


def _anomaly_segments_maplayer(layer: _Layer, qgs_path: Path) -> str:
    """Line layer categorized on ``confidence`` with ``window_id`` labels."""
    ds = _rel(layer.path, qgs_path)
    ext = _extent_xml(layer.extent, "    ")
    tiers = ["HIGH", "MODERATE", "SCREEN"]

    categories = "".join(
        f'          <category value="{t}" label="{t}" symbol="{i}" '
        'render="true"/>\n'
        for i, t in enumerate(tiers)
    )
    symbols = "".join(
        (
            f'          <symbol type="line" name="{i}" alpha="1" clip_to_extent="1">\n'
            '            <layer class="SimpleLine" enabled="1">\n'
            '              <Option type="Map">\n'
            f'                <Option name="line_color" type="QString" value="{_ANOMALY_TIER_COLORS[t]}"/>\n'
            '                <Option name="line_width" type="QString" value="1.2"/>\n'
            '                <Option name="line_width_unit" type="QString" value="MM"/>\n'
            '                <Option name="capstyle" type="QString" value="round"/>\n'
            '                <Option name="joinstyle" type="QString" value="round"/>\n'
            "              </Option>\n"
            "            </layer>\n"
            "          </symbol>\n"
        )
        for i, t in enumerate(tiers)
    )
    return (
        '    <maplayer type="vector" geometry="Line" '
        'hasScaleBasedVisibilityFlag="0" wkbType="LineString" '
        'autoRefreshEnabled="0" refreshOnNotifyEnabled="0">\n'
        f"      <id>{escape(layer.layer_id)}</id>\n"
        f"      <datasource>{escape(ds)}</datasource>\n"
        f"      <layername>{escape(layer.name)}</layername>\n"
        f"{ext}"
        "      <srs>\n" + _srs_block(layer.epsg, "        ") + "      </srs>\n"
        "      <provider encoding=\"UTF-8\">ogr</provider>\n"
        '      <renderer-v2 type="categorizedSymbol" attr="confidence" '
        'forceraster="0" symbollevels="0" enableorderby="0">\n'
        "        <categories>\n"
        f"{categories}"
        "        </categories>\n"
        "        <symbols>\n"
        f"{symbols}"
        "        </symbols>\n"
        "      </renderer-v2>\n"
        '      <labeling type="simple">\n'
        "        <settings>\n"
        + _text_style_xml("window_id", "          ")
        + '          <placement placement="2" repeatDistance="0"/>\n'
        '          <rendering scaleVisibility="0" drawLabels="1"/>\n'
        "        </settings>\n"
        "      </labeling>\n"
        "    </maplayer>\n"
    )


def _anomaly_sites_maplayer(layer: _Layer, qgs_path: Path) -> str:
    """Point layer with a prominent red-outlined star and ``site_id`` labels."""
    ds = _rel(layer.path, qgs_path)
    ext = _extent_xml(layer.extent, "    ")
    return (
        '    <maplayer type="vector" geometry="Point" '
        'hasScaleBasedVisibilityFlag="0" wkbType="Point" '
        'autoRefreshEnabled="0" refreshOnNotifyEnabled="0">\n'
        f"      <id>{escape(layer.layer_id)}</id>\n"
        f"      <datasource>{escape(ds)}</datasource>\n"
        f"      <layername>{escape(layer.name)}</layername>\n"
        f"{ext}"
        "      <srs>\n" + _srs_block(layer.epsg, "        ") + "      </srs>\n"
        "      <provider encoding=\"UTF-8\">ogr</provider>\n"
        '      <renderer-v2 type="singleSymbol" forceraster="0" symbollevels="0">\n'
        "        <symbols>\n"
        '          <symbol type="marker" name="0" alpha="1" clip_to_extent="1">\n'
        '            <layer class="SimpleMarker" enabled="1">\n'
        '              <Option type="Map">\n'
        '                <Option name="name" type="QString" value="star"/>\n'
        '                <Option name="color" type="QString" value="255,215,0,255"/>\n'
        '                <Option name="outline_color" type="QString" value="215,25,28,255"/>\n'
        '                <Option name="outline_width" type="QString" value="0.6"/>\n'
        '                <Option name="outline_width_unit" type="QString" value="MM"/>\n'
        '                <Option name="size" type="QString" value="4"/>\n'
        '                <Option name="size_unit" type="QString" value="MM"/>\n'
        '                <Option name="horizontal_anchor_point" type="QString" value="1"/>\n'
        '                <Option name="vertical_anchor_point" type="QString" value="1"/>\n'
        "              </Option>\n"
        "            </layer>\n"
        "          </symbol>\n"
        "        </symbols>\n"
        "      </renderer-v2>\n"
        '      <labeling type="simple">\n'
        "        <settings>\n"
        + _text_style_xml("site_id", "          ")
        + '          <placement placement="1" repeatDistance="0" '
        'dist="2" distUnits="MM"/>\n'
        '          <rendering scaleVisibility="0" drawLabels="1"/>\n'
        "        </settings>\n"
        "      </labeling>\n"
        "    </maplayer>\n"
    )


def _extent_xml(extent: Optional[Tuple[float, float, float, float]], indent: str) -> str:
    if not extent:
        return ""
    xmin, ymin, xmax, ymax = extent
    return (
        f"{indent}<extent>\n"
        f"{indent}  <xmin>{xmin!r}</xmin>\n"
        f"{indent}  <ymin>{ymin!r}</ymin>\n"
        f"{indent}  <xmax>{xmax!r}</xmax>\n"
        f"{indent}  <ymax>{ymax!r}</ymax>\n"
        f"{indent}</extent>\n"
    )


def _union_extent(layers: List[_Layer]) -> Tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for lyr in layers:
        if lyr.extent:
            xs0.append(lyr.extent[0]); ys0.append(lyr.extent[1])
            xs1.append(lyr.extent[2]); ys1.append(lyr.extent[3])
    if not xs0:
        # Fall back to a small box around the dive's UTM neighbourhood.
        return (578000.0, 1081000.0, 579000.0, 1083000.0)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


# Order in which groups appear.  In a QGIS layer tree the FIRST child is drawn
# ON TOP, so "Anomalies" leads (above everything), then the trackline, sensors,
# DEMs and finally the orthomosaics at the bottom of the stack.
_GROUP_ORDER = ["Anomalies", "Navigation", "Sensors", "DEMs", "Orthomosaics"]


def _layer_tree(layers: List[_Layer], qgs_path: Path) -> str:
    out = ['  <layer-tree-group>\n']
    groups = {g: [] for g in _GROUP_ORDER}
    for lyr in layers:
        groups.setdefault(lyr.group, []).append(lyr)
    for gname in list(_GROUP_ORDER) + [g for g in groups if g not in _GROUP_ORDER]:
        glayers = groups.get(gname)
        if not glayers:
            continue
        out.append(
            f'    <layer-tree-group name={quoteattr(gname)} '
            'checked="Qt::Checked" expanded="1">\n'
        )
        for lyr in glayers:
            provider = "gdal" if lyr.kind == "raster" else "ogr"
            src = _rel(lyr.path, qgs_path)
            out.append(
                f'      <layer-tree-layer id={quoteattr(lyr.layer_id)} '
                f'name={quoteattr(lyr.name)} source={quoteattr(src)} '
                f'providerKey="{provider}" checked="Qt::Checked" expanded="0"/>\n'
            )
        out.append("    </layer-tree-group>\n")
    out.append("    <custom-order enabled=\"0\">\n")
    for lyr in layers:
        out.append(f"      <item>{escape(lyr.layer_id)}</item>\n")
    out.append("    </custom-order>\n")
    out.append("  </layer-tree-group>\n")
    return "".join(out)


def _write_qgs_xml(layers: List[_Layer], qgs_path: Path, project_title: str) -> str:
    xmin, ymin, xmax, ymax = _union_extent(layers)
    parts: List[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>\n')
    parts.append(
        f'<qgis projectname={quoteattr(project_title)} version="3.34.15-Prizren">\n'
    )
    parts.append("  <homePath path=\"\"/>\n")
    parts.append(f"  <title>{escape(project_title)}</title>\n")

    # Layer tree
    parts.append(_layer_tree(layers, qgs_path))

    # Map canvas / default view
    parts.append('  <mapcanvas name="theMapCanvas" annotationsVisible="1">\n')
    parts.append("    <units>meters</units>\n")
    parts.append(_extent_xml((xmin, ymin, xmax, ymax), "    "))
    parts.append("    <destinationsrs>\n")
    parts.append(_spatialrefsys("      "))
    parts.append("    </destinationsrs>\n")
    parts.append("  </mapcanvas>\n")

    # Project CRS
    parts.append("  <projectCrs>\n")
    parts.append(_spatialrefsys("    "))
    parts.append("  </projectCrs>\n")

    # Layer definitions
    parts.append("  <projectlayers>\n")
    for lyr in layers:
        if lyr.kind == "raster":
            parts.append(_raster_maplayer(lyr, qgs_path))
        elif lyr.style == "anomaly_segments":
            parts.append(_anomaly_segments_maplayer(lyr, qgs_path))
        elif lyr.style == "anomaly_sites":
            parts.append(_anomaly_sites_maplayer(lyr, qgs_path))
        else:
            parts.append(_vector_maplayer(lyr, qgs_path))
    parts.append("  </projectlayers>\n")

    # Layer order (bottom-to-top draw order = reverse of tree top-to-bottom)
    parts.append('  <layerorder>\n')
    for lyr in reversed(layers):
        parts.append(f"    <layer id={quoteattr(lyr.layer_id)}/>\n")
    parts.append("  </layerorder>\n")

    # Store datasource paths as relative so the project is portable.
    parts.append("  <properties>\n")
    parts.append("    <Paths>\n")
    parts.append('      <Absolute type="bool">false</Absolute>\n')
    parts.append("    </Paths>\n")
    parts.append("  </properties>\n")

    parts.append("</qgis>\n")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Optional headless qgis.core path
# --------------------------------------------------------------------------- #
def _qgis_api_available() -> bool:
    try:
        from qgis.core import (  # noqa: F401
            QgsProject, QgsApplication, QgsRasterLayer, QgsVectorLayer,
        )
    except Exception:
        return False
    return True


def _build_with_qgis_api(layers: List[_Layer], trackline: Optional[Path],
                         qgs_path: Path, project_title: str,
                         notes: List[str]) -> bool:
    """Build the .qgs via the real QGIS API when it is importable headless."""
    try:
        from qgis.core import (
            QgsProject, QgsApplication, QgsRasterLayer, QgsVectorLayer,
            QgsCoordinateReferenceSystem, QgsLayerTreeGroup,
        )
    except Exception:
        return False
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QgsApplication([], False)
        app.initQgis()
        project = QgsProject.instance()
        project.setCrs(QgsCoordinateReferenceSystem(EPSG_AUTHID))
        project.setTitle(project_title)
        root = project.layerTreeRoot()
        group_nodes: dict[str, QgsLayerTreeGroup] = {}
        for gname in _GROUP_ORDER:
            group_nodes[gname] = root.addGroup(gname)
        for lyr in layers:
            grp = group_nodes.get(lyr.group) or root.addGroup(lyr.group)
            group_nodes.setdefault(lyr.group, grp)
            if lyr.kind == "raster":
                ml = QgsRasterLayer(str(lyr.path), lyr.name, "gdal")
            else:
                ml = QgsVectorLayer(str(lyr.path), lyr.name, "ogr")
            if not ml.isValid():
                notes.append(f"qgis-api: invalid layer {lyr.name}")
                continue
            project.addMapLayer(ml, False)
            grp.addLayer(ml)
        qgs_path.parent.mkdir(parents=True, exist_ok=True)
        ok = project.write(str(qgs_path))
        app.exitQgis()
        if ok:
            notes.append("built with qgis.core API")
        return bool(ok)
    except Exception as exc:  # pragma: no cover - API path rarely reached here
        notes.append(f"qgis.core API path failed ({exc}); using hand-written XML")
        return False


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_qgis_project(workspace_dir: str, out_path: Optional[str] = None) -> str:
    """Assemble a QGIS project for an EPR dive workspace.

    Parameters
    ----------
    workspace_dir : str
        Path to the ``.eprproj`` bundle (or legacy workspace) to assemble.
    out_path : str, optional
        Where to write the ``.qgs``.  Defaults to
        ``<workspace>/survey/qgis/<name>_project.qgs``.  The matching ``.qgz``
        archive is written alongside with a ``_QGIS_project.qgz`` suffix.

    Returns
    -------
    str
        Absolute path to the written ``.qgs`` file.
    """
    resolver = PathResolver(workspace_dir)
    notes: List[str] = []

    ws = Path(workspace_dir)
    # Dive/base name for output filenames, e.g. "J1756_down".
    base_name = ws.stem if ws.suffix else ws.name
    base_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", base_name).strip("_") or "dive"

    if out_path:
        qgs_path = Path(out_path)
    else:
        qgs_path = resolver.survey_products() / "qgis" / f"{base_name}_project.qgs"
    qgs_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect layers.
    layers = collect_layers(resolver, notes)

    # Build and add the trackline vector.
    trackline = build_trackline_geojson(resolver, notes)
    if trackline is not None:
        layers.append(_Layer("vector", "Nav Trackline", trackline,
                             "Navigation", 1, None))

    # Anomaly windows + sites (drawn on top, in their own group).
    layers.extend(collect_anomaly_layers(resolver, notes))

    if not layers:
        notes.append("WARNING: no layers resolved; writing an empty project")

    project_title = f"{base_name} — EPR dive"

    # Prefer the real QGIS API only if it is importable AND runs headless.
    used_api = False
    if _qgis_api_available():
        used_api = _build_with_qgis_api(layers, trackline, qgs_path,
                                        project_title, notes)
    if not used_api:
        xml = _write_qgs_xml(layers, qgs_path, project_title)
        qgs_path.write_text(xml, encoding="utf-8")
        notes.append("built with hand-written .qgs XML")

    # Zip the .qgs into a portable .qgz (a .qgz is just a zip with the .qgs inside).
    qgz_path = qgs_path.with_name(f"{base_name}_QGIS_project.qgz")
    try:
        with zipfile.ZipFile(qgz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(qgs_path, arcname=qgs_path.name)
        notes.append(f"wrote qgz: {qgz_path}")
    except OSError as exc:
        notes.append(f"failed writing qgz: {exc}")

    # Emit notes to stderr for visibility without polluting the return value.
    for line in notes:
        print(f"[qgis_project] {line}", file=sys.stderr)

    return str(qgs_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Assemble a QGIS project (.qgs + .qgz) for an EPR dive workspace.",
    )
    parser.add_argument("workspace", help="Path to the .eprproj bundle (or workspace).")
    parser.add_argument("-o", "--out", default=None,
                        help="Output .qgs path (default: <ws>/survey/qgis/<name>_project.qgs).")
    args = parser.parse_args(argv)

    qgs = build_qgis_project(args.workspace, args.out)
    print(qgs)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
