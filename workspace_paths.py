"""
workspace_paths.py — one resolver the whole app asks "where does this go?".

The app has two on-disk layouts:

  * LEGACY — a plain workspace directory, flat: ``<ws>/interp_full.csv``,
    ``<ws>/outputs/…``, ``<ws>/job_003_name/outputs/…``, ``<ws>/sampling_…/…``.
    This is what every existing workspace (including real survey data) uses.

  * BUNDLE — the new ``<name>.eprproj`` tree owned by layout.py: inputs/, jobs/
    job_003/products/…, survey/, runs/.  Clean, ID-addressed, no spaces.

``PathResolver`` is the single seam between the writers (main_window, plan_service,
stack_runner, photogrammetry) and those two layouts.  In LEGACY mode every method
reproduces today's exact path *byte for byte*, so pointing an existing workspace
through the resolver changes nothing.  In BUNDLE mode the same call yields the new
layout.py path.  A workspace is a bundle iff its directory ends in ``.eprproj`` or
contains a ``project.json`` marker.

Because output_service already builds ``<base>/<subfolder>/run_NNN`` itself, the
resolver only has to hand it the right *base* directory for a scope; the product
subfolders and run numbering keep working unchanged in both layouts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from layout import WorkspaceLayout, WORKSPACE_SUFFIX, workspace_name

# Parse the integer job id out of a legacy "job_003_name" directory / a Job.
_JOB_ID_RE = re.compile(r"job_(\d+)")


def is_bundle(workspace_dir: str | Path) -> bool:
    """True if a workspace directory is a new .eprproj bundle."""
    ws = Path(workspace_dir)
    return ws.suffix == WORKSPACE_SUFFIX or (ws / "project.json").is_file()


def _legacy_job_dirname(job) -> str:
    """Reproduce MainWindow._job_output_dirname exactly: job_NNN[_name]."""
    base = f"job_{int(job.job_id):03d}"
    name = getattr(job, "name", None)
    if name:
        safe = re.sub(r"[^\w\-]", "_", str(name)).strip("_")
        return f"{base}_{safe}" if safe else base
    return base


class PathResolver:
    """Answers every "where does X live?" question for one workspace."""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.ws = Path(workspace_dir)
        self.bundle = is_bundle(self.ws)
        self.layout: Optional[WorkspaceLayout] = (
            WorkspaceLayout(self.ws) if self.bundle else None)

    # -- master inputs --------------------------------------------------------
    def interp_full(self) -> Path:
        if self.bundle:
            return self.layout.interp_full
        return self.ws / "interp_full.csv"

    def inputs_dir(self, create: bool = False) -> Path:
        if self.bundle:
            return self.layout.inputs_dir(create=create)
        if create:
            self.ws.mkdir(parents=True, exist_ok=True)
        return self.ws

    # -- product bases (output_service appends <subfolder>/run_NNN) -----------
    def survey_products(self, create: bool = False) -> Path:
        """Base dir for full-dataset ('survey') products."""
        if self.bundle:
            return self.layout.products_dir("survey", create=create)
        base = self.ws / "outputs"
        if create:
            base.mkdir(parents=True, exist_ok=True)
        return base

    def job_products(self, job, create: bool = False) -> Path:
        """Base dir for a job's products."""
        if self.bundle:
            return self.layout.products_dir(int(job.job_id), create=create)
        base = self.ws / _legacy_job_dirname(job) / "outputs"
        if create:
            base.mkdir(parents=True, exist_ok=True)
        return base

    def products_for(self, job, create: bool = False) -> Path:
        """Product base for a scope: `job` is a Job, or None for the survey scope."""
        return self.survey_products(create) if job is None else self.job_products(job, create)

    # -- per-job metadata / interp -------------------------------------------
    def job_dir(self, job, create: bool = False) -> Path:
        if self.bundle:
            return self.layout.job_dir(int(job.job_id), create=create)
        base = self.ws / _legacy_job_dirname(job)
        if create:
            base.mkdir(parents=True, exist_ok=True)
        return base

    def job_filtered_interp(self, job) -> Path:
        if self.bundle:
            return self.layout.job_filtered_interp(int(job.job_id))
        return self.ws / _legacy_job_dirname(job) / "filtered_interp.csv"

    # -- frame extraction / sampling -----------------------------------------
    def sampling_dir(self, task_id, job) -> Path:
        """Directory the sampling pipeline extracts frames/segments into."""
        if self.bundle:
            # frames live under the job (or survey) scope; a run number groups one
            # extraction's segments.  run_001 is fine — segments carry their own ids.
            if job is None:
                base = self.layout.products_dir("survey") / "frames"
            else:
                base = self.layout.frames_dir(int(job.job_id))
            run = self.layout.next_run_id(base)
            return base / f"run_{run:03d}" / "segments"
        suffix = _legacy_job_dirname(job) if job is not None else "full"
        return self.ws / f"sampling_{task_id}_{suffix}"

    # -- photogrammetry -------------------------------------------------------
    def photogrammetry_root(self, products_base: str | Path) -> Path:
        """Root passed to prepare_run_dir(); legacy nests one extra job_NNN, bundle
        keeps it flat under the scope's photogrammetry/ dir."""
        return Path(products_base) / "photogrammetry"

    # -- anomaly detection ----------------------------------------------------
    def anomaly_dir(self) -> Path:
        if self.bundle:
            return self.layout.products_dir("survey") / "anomaly"
        return self.ws / "anomaly_site_catalog"

    # -- provenance -----------------------------------------------------------
    def registry_json(self) -> Path:
        return self.ws / "runs" / "registry.json"


def default_bundle_dir(display_name: str, parent: str | Path) -> Path:
    """The .eprproj bundle path for a new workspace with the given display name."""
    return Path(parent) / f"{workspace_name(display_name)}{WORKSPACE_SUFFIX}"
