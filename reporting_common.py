#!/usr/bin/env python3
"""Shared helpers for the product and reporting modules: provenance, units, chunk QC.

Qt-free and light: stdlib only at import time.  Used by catalog_builder,
survey_report, merge_products, fauna_timeseries, fauna_occurrences,
fathomnet_detect and photogrammetry_service.

  code_version()             git describe of this checkout ("unknown" if no git)
  file_info(path, sha=False) {path, size_bytes, mtime_utc[, sha256]}
  provenance(inputs, **kw)   the block every product sidecar carries
  sensor_units(ws)           {display_name: units} from workspace.json
  channel_label(name, units) "pCH4 (uatm)" / "O2 (uM)" / "Temperature (degC)"
  chunk_status(chunk_dir)    per-chunk photogrammetry QC (aligned/total, ok|failed)
"""
from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, Optional

REPO_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------- provenance --
@functools.lru_cache(maxsize=1)
def code_version() -> str:
    """``git describe --always --dirty --tags`` of this checkout.

    A dirty tree is reported as such ("<hash>-dirty"): a product made from
    uncommitted code cannot be traced to a commit, and the sidecar says so.
    """
    try:
        res = subprocess.run(
            ["git", "-C", str(REPO_DIR), "describe", "--always", "--dirty",
             "--tags", "--abbrev=12"],
            capture_output=True, text=True, timeout=10)
        out = res.stdout.strip()
        if res.returncode == 0 and out:
            return out
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown (no git checkout)"


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc(ts: float) -> str:
    try:
        return _dt.datetime.fromtimestamp(float(ts), _dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path, sha: bool = False) -> dict:
    """Path + size + mtime (UTC) of one input; sha256 only when asked (big
    inputs on /mnt/f cost seconds to hash)."""
    p = Path(str(path))
    info: dict = {"path": str(p)}
    try:
        st = p.stat()
    except OSError:
        info["missing"] = True
        return info
    info["size_bytes"] = int(st.st_size)
    info["mtime_utc"] = _utc(st.st_mtime)
    if sha and p.is_file():
        try:
            info["sha256"] = sha256_file(p)
        except OSError:
            pass
    return info


def provenance(inputs: Iterable = (), *, hashed: Iterable = (), **extra) -> dict:
    """The provenance block written into every product sidecar.

    ``inputs`` are recorded by path/size/mtime; ``hashed`` also get a sha256
    (use it for small or identity-critical files such as model weights).
    """
    ins = [file_info(p) for p in inputs if p]
    ins += [file_info(p, sha=True) for p in hashed if p]
    block = {"generated_utc": utc_now(),
             "code_version": code_version(),
             "inputs": ins}
    block.update({k: v for k, v in extra.items() if v is not None})
    return block


def write_json(path, data: dict) -> Optional[str]:
    """Best-effort sidecar write (a sidecar failure must not cost the product)."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2, default=str),
                              encoding="utf-8")
        return str(path)
    except OSError:
        return None


# --------------------------------------------------------------------- units --
_UNIT_DISPLAY = {"uatm": "µatm", "um": "µM", "umol/l": "µmol/L", "degc": "°C",
                 "psu": "PSU", "m": "m"}


def sensor_units(workspace_dir) -> dict:
    """{channel display_name: units} from workspace.json (empty when unknown)."""
    out: dict = {}
    try:
        ws = json.loads((Path(str(workspace_dir)) / "workspace.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for sf in ws.get("sensor_files") or []:
        for ch in sf.get("channels") or []:
            name = ch.get("display_name") or ch.get("source_column")
            if name:
                out[str(name)] = str(ch.get("units") or "").strip()
    return out


def _gas(name: str) -> Optional[str]:
    m = re.match(r"\s*(CO2|CH4)\b", str(name), re.I)
    return m.group(1).upper() if m else None


def channel_label(name: str, units: str = "", ascii_only: bool = True) -> str:
    """Unit-bearing label for a sensor channel.

    CO2/CH4 recorded in uatm are PARTIAL PRESSURES, not concentrations, so
    "CH4 Concentration" [uatm] becomes "pCH4 (uatm)".  Other channels drop the
    word "Concentration" only when units are known (the units say what it is).
    ``ascii_only=False`` gives display units (µatm, °C) for figures.
    """
    name = str(name)
    u = str(units or "").strip()
    shown = u if ascii_only else _UNIT_DISPLAY.get(u.lower(), u)
    gas = _gas(name)
    if gas and u.lower() == "uatm":
        base = f"p{gas}"
    elif u:
        base = re.sub(r"\s*Concentration\s*$", "", name, flags=re.I) or name
    else:
        return name
    return f"{base} ({shown})" if shown else base


def label_for(workspace_dir, name: str, ascii_only: bool = True) -> str:
    return channel_label(name, sensor_units(workspace_dir).get(name, ""),
                         ascii_only=ascii_only)


# ----------------------------------------------------------- chunk status ----
ORTHO_MIN_BYTES = 100_000


def chunk_status(chunk_dir) -> dict:
    """QC for one photogrammetry chunk directory, read from what is on disk.

    aligned/total come from cameras.json (null pose = not aligned).  A chunk is
    "ok" when it produced an orthomosaic or DEM of real size; otherwise it is
    "failed" with a reason.  0-byte files (Metashape writes an empty report for
    a chunk with nothing aligned) are never counted as products.
    """
    d = Path(str(chunk_dir))
    st: dict = {"chunk": d.name, "run": d.parent.name, "path": str(d)}
    total = aligned = None
    cams = d / "cameras.json"
    if cams.is_file():
        try:
            data = json.loads(cams.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                total = len(data)
                aligned = sum(1 for v in data.values() if v)
        except (OSError, ValueError):
            pass
    st["cameras_total"], st["cameras_aligned"] = total, aligned
    products, empty = [], []
    for name in ("orthomosaic.tif", "dem.tif", "mesh.obj", "dense.ply",
                 "sparse.ply", "report.pdf"):
        p = d / name
        try:
            size = p.stat().st_size
        except OSError:
            continue
        (products if size > 0 else empty).append(name)
    st["products"], st["empty_files"] = products, empty

    def _big(name, floor):
        try:
            return (d / name).stat().st_size > floor
        except OSError:
            return False
    ok = _big("orthomosaic.tif", ORTHO_MIN_BYTES) or _big("dem.tif", ORTHO_MIN_BYTES)
    st["status"] = "ok" if ok else "failed"
    if not ok:
        if aligned is not None and total:
            st["reason"] = (f"{aligned}/{total} cameras aligned"
                            + ("" if aligned >= 2 else " — nothing to reconstruct"))
        elif total is None:
            st["reason"] = "no cameras.json (run crashed or never started)"
        else:
            st["reason"] = "no orthomosaic/DEM written"
    return st


def run_status(run_dir) -> dict:
    """``run_status.json`` written by photogrammetry_service (empty if absent)."""
    try:
        return json.loads((Path(str(run_dir)) / "run_status.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return {}
