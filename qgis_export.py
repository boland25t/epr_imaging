"""
qgis_export.py — QGIS 3.x project file generator

Scans an EPR outputs directory tree for GeoTIFF files, then writes a .qgs
project XML file that QGIS 3.x can open directly — no additional imports or
layer-by-layer setup needed.

Layer organisation written to the project:
  ├── Navigation
  │     └── Nav Depth (nav_depth.tif)
  └── Sensors
        └── {channel}
              ├── {channel} 2D           (sensor_2d/*.tif)
              └── Depth Slices {channel} (sensor_slices/*.tif, sorted by depth)

Every sensor layer gets a viridis pseudocolor renderer.  The nav layer gets
a blues-to-reds depth renderer.  Min/max are read from each GeoTIFF so the
colormap stretches to the actual data range.

The project CRS is read from the first GeoTIFF found; it must be a projected
(UTM) CRS so that spatial extents are meaningful in metres.

Usage:
    from qgis_export import generate_qgis_project
    path = generate_qgis_project(output_dir="/path/to/outputs",
                                  project_name="EPR Survey")
"""

from __future__ import annotations

import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Viridis and depth colormap stops (QGIS INTERPOLATED format)
# ---------------------------------------------------------------------------

# Each entry: (fraction 0-1, hex_color)
_VIRIDIS_STOPS = [
    (0.000, "#440154"),
    (0.125, "#472d7b"),
    (0.250, "#3b528b"),
    (0.375, "#2c728e"),
    (0.500, "#21918c"),
    (0.625, "#28ae80"),
    (0.750, "#5ec962"),
    (0.875, "#addc30"),
    (1.000, "#fde725"),
]

_DEPTH_STOPS = [
    (0.000, "#0d0887"),
    (0.333, "#7201a8"),
    (0.667, "#cb4679"),
    (1.000, "#f0f921"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_qgis_project(
    output_dir: str,
    project_name: str = "EPR Survey",
) -> str:
    """Scan output_dir for GeoTIFFs and write a .qgs project file.

    Returns the absolute path of the written .qgs file.

    Raises FileNotFoundError if no GeoTIFF files are found.
    Raises RuntimeError if rasterio is not available (needed for min/max/CRS).
    """
    try:
        import rasterio
    except ImportError:
        raise RuntimeError("rasterio is required for QGIS project generation.")

    out_root = Path(output_dir)

    # ── Collect GeoTIFF files ─────────────────────────────────────────────
    nav_tifs     = _collect_tifs(out_root / "nav_depth")
    sensor_2d    = _collect_sensor_2d(out_root / "sensor_2d")
    sensor_slices = _collect_sensor_slices(out_root / "sensor_slices")

    all_tifs = (
        nav_tifs
        + [p for paths in sensor_2d.values() for p in paths]
        + [p for paths in sensor_slices.values() for p in paths]
    )
    if not all_tifs:
        raise FileNotFoundError(
            f"No GeoTIFF files found under {output_dir}. "
            "Generate nav or sensor outputs first."
        )

    # ── Determine project CRS from first available GeoTIFF ────────────────
    epsg = _read_epsg(all_tifs[0])

    # ── Build XML tree ────────────────────────────────────────────────────
    root = ET.Element(
        "qgis",
        attrib={
            "projectname": project_name,
            "version": "3.28.0",
            "saveNumber": "1",
        },
    )
    ET.SubElement(root, "title").text = project_name
    ET.SubElement(root, "autotransaction", attrib={"active": "0"})
    ET.SubElement(root, "evaluateDefaultValues", attrib={"active": "0"})
    ET.SubElement(root, "trust", attrib={"active": "0"})

    _add_project_crs(root, epsg)

    # Layer tree + projectlayers are built together; we collect layer elements
    # and add them at the end (projectlayers must be a flat list).
    project_layers_el = ET.SubElement(root, "projectlayers")
    tree_root         = ET.SubElement(root, "layer-tree-group")
    ET.SubElement(tree_root, "custom-order", attrib={"enabled": "0"})

    # ── Navigation group ─────────────────────────────────────────────────
    if nav_tifs:
        nav_group = _tree_group(tree_root, "Navigation")
        for tif in nav_tifs:
            lid   = _new_id()
            _add_tree_layer(nav_group, Path(tif).stem.replace("_", " ").title(), lid)
            ml    = _make_raster_layer(tif, "Nav Depth", lid, epsg)
            _add_pseudocolor_pipe(ml, tif, _DEPTH_STOPS)
            project_layers_el.append(ml)

    # ── Sensors group ─────────────────────────────────────────────────────
    if sensor_2d or sensor_slices:
        sensors_group = _tree_group(tree_root, "Sensors")
        all_channels  = sorted(set(list(sensor_2d) + list(sensor_slices)))

        for channel in all_channels:
            ch_group = _tree_group(sensors_group, channel)

            # 2D GeoTIFF(s) for this channel
            for tif in sensor_2d.get(channel, []):
                lid   = _new_id()
                label = f"{channel} 2D"
                _add_tree_layer(ch_group, label, lid)
                ml    = _make_raster_layer(tif, label, lid, epsg)
                _add_pseudocolor_pipe(ml, tif, _VIRIDIS_STOPS)
                project_layers_el.append(ml)

            # Depth slices for this channel (sub-group)
            slices = sensor_slices.get(channel, [])
            if slices:
                slices_group = _tree_group(ch_group, f"Depth Slices — {channel}")
                for tif in sorted(slices):
                    lid   = _new_id()
                    z_str = _depth_label(Path(tif).stem)
                    label = f"{channel}  {z_str}"
                    _add_tree_layer(slices_group, label, lid)
                    ml    = _make_raster_layer(tif, label, lid, epsg)
                    _add_pseudocolor_pipe(ml, tif, _VIRIDIS_STOPS)
                    project_layers_el.append(ml)

    # ── Write file ────────────────────────────────────────────────────────
    _indent_xml(root)
    tree     = ET.ElementTree(root)
    qgs_path = out_root / f"{project_name.replace(' ', '_')}.qgs"
    with open(qgs_path, "wb") as f:
        f.write(b'<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)

    return str(qgs_path)


# ---------------------------------------------------------------------------
# File collection helpers
# ---------------------------------------------------------------------------

def _collect_tifs(directory: Path) -> list[str]:
    """Return all .tif files under directory, newest run first."""
    if not directory.exists():
        return []
    tifs = sorted(directory.rglob("*.tif"), key=lambda p: p.parent.name, reverse=True)
    return [str(t) for t in tifs]


def _collect_sensor_2d(directory: Path) -> dict[str, list[str]]:
    """Return {channel: [tif_path, ...]} for sensor_2d outputs."""
    result: dict[str, list[str]] = {}
    if not directory.exists():
        return result
    for ch_dir in sorted(directory.iterdir()):
        if ch_dir.is_dir():
            tifs = sorted(ch_dir.rglob("*.tif"), key=lambda p: p.parent.name, reverse=True)
            if tifs:
                result[ch_dir.name] = [str(t) for t in tifs]
    return result


def _collect_sensor_slices(directory: Path) -> dict[str, list[str]]:
    """Return {channel: [tif_path, ...]} for sensor_slices outputs."""
    result: dict[str, list[str]] = {}
    if not directory.exists():
        return result
    for ch_dir in sorted(directory.iterdir()):
        if ch_dir.is_dir():
            tifs = list(ch_dir.rglob("*.tif"))
            if tifs:
                # Sort by depth value embedded in filename: z+00xx.xxm
                def _z_key(p: Path) -> float:
                    m = re.search(r"z([+-]\d+\.\d+)m", p.name)
                    return float(m.group(1)) if m else 0.0
                result[ch_dir.name] = [str(t) for t in sorted(tifs, key=_z_key)]
    return result


def _depth_label(stem: str) -> str:
    """Convert 'ch4_z+005.00m' stem to 'z = +5.0 m'."""
    m = re.search(r"z([+-]\d+\.\d+)m", stem)
    if m:
        return f"z = {float(m.group(1)):+.1f} m"
    return stem


# ---------------------------------------------------------------------------
# Raster stats
# ---------------------------------------------------------------------------

def _read_epsg(tif_path: str) -> int:
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            if src.crs:
                epsg = src.crs.to_epsg()
                if epsg:
                    return int(epsg)
    except Exception:
        pass
    return 32633  # fallback: UTM 33N


def _raster_min_max(tif_path: str) -> tuple[float, float]:
    """Return (min, max) of valid data in the first band."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            data = src.read(1, masked=True)
            valid = data.compressed()
            if len(valid):
                return float(valid.min()), float(valid.max())
    except Exception:
        pass
    return 0.0, 1.0


# ---------------------------------------------------------------------------
# XML construction helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    return uuid.uuid4().hex


def _srs_el(epsg: int) -> ET.Element:
    """Build a minimal <spatialrefsys> element from an EPSG code."""
    el = ET.Element("spatialrefsys")
    ET.SubElement(el, "wkt")
    ET.SubElement(el, "proj4")
    ET.SubElement(el, "srsid").text   = str(epsg)
    ET.SubElement(el, "srid").text    = str(epsg)
    ET.SubElement(el, "authid").text  = f"EPSG:{epsg}"
    ET.SubElement(el, "description")
    ET.SubElement(el, "projectionacronym").text  = "utm"
    ET.SubElement(el, "ellipsoidacronym").text   = "EPSG:7030"
    ET.SubElement(el, "geographicflag").text     = "false"
    return el


def _add_project_crs(parent: ET.Element, epsg: int) -> None:
    crs_el = ET.SubElement(parent, "projectCrs")
    crs_el.append(_srs_el(epsg))


def _tree_group(parent: ET.Element, name: str) -> ET.Element:
    g = ET.SubElement(
        parent, "layer-tree-group",
        attrib={"name": name, "expanded": "1", "checked": "Qt::Checked"},
    )
    ET.SubElement(g, "custom-order", attrib={"enabled": "0"})
    return g


def _add_tree_layer(group: ET.Element, name: str, layer_id: str) -> None:
    ET.SubElement(
        group, "layer-tree-layer",
        attrib={
            "name":      name,
            "layerid":   layer_id,
            "expanded":  "1",
            "checked":   "Qt::Checked",
            "source":    "",  # QGIS resolves from maplayer
        },
    )


def _make_raster_layer(
    tif_path: str,
    name: str,
    layer_id: str,
    epsg: int,
) -> ET.Element:
    ml = ET.Element(
        "maplayer",
        attrib={
            "type":                    "raster",
            "autoRefreshEnabled":      "0",
            "refreshOnNotifyEnabled":  "0",
        },
    )
    ET.SubElement(ml, "id").text         = layer_id
    ET.SubElement(ml, "datasource").text = tif_path
    ET.SubElement(ml, "layername").text  = name
    ET.SubElement(ml, "provider", attrib={"encoding": "System"}).text = "gdal"

    srs = ET.SubElement(ml, "srs")
    srs.append(_srs_el(epsg))
    return ml


def _add_pseudocolor_pipe(
    ml: ET.Element,
    tif_path: str,
    stops: list[tuple[float, str]],
) -> None:
    """Append a singlebandpseudocolor pipe element with the given color stops."""
    v_min, v_max = _raster_min_max(tif_path)
    v_range = v_max - v_min if v_max > v_min else 1.0

    pipe = ET.SubElement(ml, "pipe")
    renderer = ET.SubElement(
        pipe, "rasterrenderer",
        attrib={
            "type":              "singlebandpseudocolor",
            "band":              "1",
            "alphaBand":         "-1",
            "opacity":           "1",
            "nodataColor":       "",
            "classificationMax": str(v_max),
            "classificationMin": str(v_min),
        },
    )
    ET.SubElement(renderer, "rasterTransparency")

    shader = ET.SubElement(
        renderer, "colorrampshader",
        attrib={
            "colorRampType":    "INTERPOLATED",
            "classificationMode": "1",
            "clip":             "0",
            "minimumValue":     str(v_min),
            "maximumValue":     str(v_max),
        },
    )
    for frac, color in stops:
        val = v_min + frac * v_range
        ET.SubElement(
            shader, "item",
            attrib={
                "value": str(val),
                "color": color,
                "label": f"{val:.4g}",
                "alpha": "255",
            },
        )


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """In-place pretty-print indentation for Python < 3.9 compatibility."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent
