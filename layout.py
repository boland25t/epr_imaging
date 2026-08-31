"""
layout.py — the single owner of every path inside a workspace.

Today path strings are built ad hoc across output_service, plan_service,
photogrammetry_service, stack_runner and main_window, which produced the mess
this redesign is fixing: spaces in directory names ("CO2 Concentration",
"Down Data w Orientation"), version-suffix cruft (job_..._v3/_v5/_v6), the
overloaded "job_" prefix, and per-product run_NNN counters with no per-execution
correlation.  Every path in the new layout is minted HERE instead, so the rules
live in one place and the tree stays consistent.

Design (Metashape-inspired — a small manifest at the root, heavy data addressed
relative to it):

    <workspace>.eprproj/
    ├── project.json                 the manifest (all stored paths are relative)
    ├── inputs/    interp_full.csv, navigation.csv
    ├── jobs/
    │   └── job_007/                 stable ID in the path; human name in job.json
    │       ├── job.json  filtered_interp.csv
    │       ├── frames/run_003/segments/…
    │       └── products/
    │           ├── tracklines/run_002/
    │           ├── rasters/<slug>/run_NNN/
    │           ├── photogrammetry/run_002/{project.psx, chunks/chunk_000/…}
    │           ├── netcdf/<slug>/run_NNN/
    │           ├── qc/run_NNN/   anomaly/run_NNN/
    ├── survey/                      full-dataset products (same shape as a job's products/)
    ├── cache/  logs/  archive/

Two rules the whole system relies on:

  * SLUGIFY EVERYTHING — no spaces ever reach the filesystem.  Human names
    ("J1756 Job 2", "CO2 Concentration") live only in JSON metadata.
  * A RUN IS ONE EXECUTION STEP, not one product — every run dir gets a
    run.json (written by manifest.py) and an ``execution_id`` shared across all
    runs of one stack invocation, so "what did the 11:10 run produce?" is
    answerable.

This module is Qt-free and does pure path arithmetic (plus mkdir on request); it
never reads or writes product data.  manifest.py builds on it for provenance.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

WORKSPACE_SUFFIX = ".eprproj"

# Canonical product-type folder names under a scope's products/ directory.
# Maps the app's task/product identifiers → clean folder names.
PRODUCT_FOLDER = {
    "nav_3d": "tracklines",
    "nav_2d": "rasters",
    "sensor_2d": "rasters",
    "depth_slice_geotiffs": "rasters",
    "sensor_netcdf": "netcdf",
    "photogrammetry": "photogrammetry",
    "qc_report": "qc",
    "anomaly_detect": "anomaly",
    "frame_stats": "frame_stats",
    # archived, but keep a home so migration of old outputs has somewhere to land
    "sensor_3d": "rasters",
    "sensor_slices": "rasters",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_RUN_RE = re.compile(r"^run_(\d+)$")


def slugify(name: str, default: str = "item") -> str:
    """Filesystem-safe slug: lowercase, non-alphanumerics → single '_', trimmed.

        "CO2 Concentration"        -> "co2_concentration"
        "Down Data w Orientation"  -> "down_data_w_orientation"
        "J1756 Job 2 (v5)"         -> "j1756_job_2_v5"

    Empty / all-symbol input falls back to `default` so a path component is
    never blank.
    """
    s = _SLUG_RE.sub("_", str(name).strip().lower()).strip("_")
    return s or default


def workspace_name(display_name: str) -> str:
    """Slugified bundle directory name for a workspace (without the suffix)."""
    return slugify(display_name, default="workspace")


class WorkspaceLayout:
    """All paths for one workspace, minted from its root directory.

    `root` is the workspace bundle directory (…/<name>.eprproj).  Nothing here
    creates data; call sites that need a directory to exist pass ``create=True``
    to the accessor, which mkdirs it (parents included) and returns the path.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- top level ------------------------------------------------------------
    @property
    def project_json(self) -> Path:
        return self.root / "project.json"

    def inputs_dir(self, create: bool = False) -> Path:
        return self._dir(self.root / "inputs", create)

    @property
    def interp_full(self) -> Path:
        return self.root / "inputs" / "interp_full.csv"

    def logs_dir(self, create: bool = False) -> Path:
        return self._dir(self.root / "logs", create)

    def cache_dir(self, create: bool = False) -> Path:
        return self._dir(self.root / "cache", create)

    def archive_dir(self, create: bool = False) -> Path:
        return self._dir(self.root / "archive", create)

    def runs_dir(self, create: bool = False) -> Path:
        """Workspace-level provenance store (registry + input-hash cache)."""
        return self._dir(self.root / "runs", create)

    @property
    def registry_json(self) -> Path:
        return self.root / "runs" / "registry.json"

    # -- jobs -----------------------------------------------------------------
    def job_dir(self, job_id: int, create: bool = False) -> Path:
        return self._dir(self.root / "jobs" / f"job_{int(job_id):03d}", create)

    def job_json(self, job_id: int) -> Path:
        return self.job_dir(job_id) / "job.json"

    def job_filtered_interp(self, job_id: int) -> Path:
        return self.job_dir(job_id) / "filtered_interp.csv"

    def frames_dir(self, job_id: int, create: bool = False) -> Path:
        return self._dir(self.job_dir(job_id) / "frames", create)

    def frames_run_dir(self, job_id: int, run_id: int, create: bool = False) -> Path:
        return self._dir(self.frames_dir(job_id) / _run(run_id) / "segments", create)

    # -- products (job scope or the survey/ full-dataset scope) ---------------
    def products_dir(self, scope: str | int, create: bool = False) -> Path:
        """`scope` is a job_id (int) or the string 'survey' (full dataset)."""
        if scope == "survey" or scope == "full":
            base = self.root / "survey"
        else:
            base = self.job_dir(int(scope)) / "products"
        return self._dir(base, create)

    def product_type_dir(self, scope: str | int, product_type: str,
                         channel: Optional[str] = None, create: bool = False) -> Path:
        """Folder holding all runs of one product type (per channel where relevant)."""
        folder = PRODUCT_FOLDER.get(product_type, slugify(product_type))
        base = self.products_dir(scope) / folder
        if channel:
            base = base / slugify(channel)
        return self._dir(base, create)

    def product_run_dir(self, scope: str | int, product_type: str, run_id: int,
                       channel: Optional[str] = None, create: bool = False) -> Path:
        """One run's output directory for a product type."""
        return self._dir(
            self.product_type_dir(scope, product_type, channel) / _run(run_id), create)

    # -- photogrammetry (per-chunk, Metashape-native) -------------------------
    def photogrammetry_run_dir(self, scope: str | int, run_id: int,
                              create: bool = False) -> Path:
        return self._dir(
            self.products_dir(scope) / "photogrammetry" / _run(run_id), create)

    def photogrammetry_project(self, scope: str | int, run_id: int) -> Path:
        return self.photogrammetry_run_dir(scope, run_id) / "project.psx"

    def chunk_dir(self, scope: str | int, run_id: int, chunk_index: int,
                  create: bool = False) -> Path:
        return self._dir(
            self.photogrammetry_run_dir(scope, run_id) / "chunks"
            / f"chunk_{int(chunk_index):03d}", create)

    # -- run allocation -------------------------------------------------------
    @staticmethod
    def next_run_id(parent: Path) -> int:
        """Next monotonic run number under `parent` (existing run_NNN dirs)."""
        parent = Path(parent)
        if not parent.is_dir():
            return 1
        used = [int(m.group(1)) for p in parent.iterdir()
                if p.is_dir() and (m := _RUN_RE.match(p.name))]
        return (max(used) + 1) if used else 1

    @staticmethod
    def latest_run_id(parent: Path) -> Optional[int]:
        """Highest existing run number under `parent`, or None."""
        parent = Path(parent)
        if not parent.is_dir():
            return None
        used = [int(m.group(1)) for p in parent.iterdir()
                if p.is_dir() and (m := _RUN_RE.match(p.name))]
        return max(used) if used else None

    # -- relative-path portability (renaming the workspace stays valid) -------
    def relative(self, path: str | Path) -> str:
        """Workspace-relative POSIX string for storing in project.json/manifests."""
        p = Path(path)
        try:
            return p.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return p.as_posix()          # outside the workspace: store as-is

    def resolve(self, rel: str) -> Path:
        """Turn a stored workspace-relative path back into an absolute Path."""
        rp = Path(rel)
        return rp if rp.is_absolute() else (self.root / rp)

    # -- internal -------------------------------------------------------------
    @staticmethod
    def _dir(path: Path, create: bool) -> Path:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path


def _run(run_id: int) -> str:
    return f"run_{int(run_id):03d}"
