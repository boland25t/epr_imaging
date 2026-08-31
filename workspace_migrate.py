"""
workspace_migrate.py — opt-in migration of an OLD-layout workspace into a new
``.eprproj`` bundle.

The pre-redesign workspace was a plain directory whose paths were built ad hoc
all over the app (see layout.py's module docstring for the mess this replaces).
This module reads such a directory and produces — then, on user confirmation,
executes — a plan that COPIES its contents into the clean bundle layout owned by
``layout.WorkspaceLayout``.

Design rules (matching the rest of the redesign backbone):

  * Qt-free — pure functions plus one filesystem executor.  No PySide imports.
  * NON-DESTRUCTIVE — the executor only copies (``shutil.copy2`` /
    ``shutil.copytree``); it never moves or deletes the source.  If anything
    goes wrong the original workspace is left exactly as it was.
  * DESTINATIONS ARE MINTED, NEVER SPELLED — every target path comes from a
    ``WorkspaceLayout`` accessor, so the migrated tree obeys the same slug/ID/run
    rules as freshly produced output.
  * All timestamps are naive-UTC (the whole app is single-timezone).

Typical use by a UI::

    plan   = build_migration_plan(old_ws, default_bundle_path(old_ws))
    print(summarize_plan(plan))          # show the user, ask to confirm
    pj     = build_project_json(old_ws)
    result = execute_plan(plan, pj, progress=on_progress)

Nothing here imports Qt; the confirmation dialog lives in the UI layer.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import layout
from layout import WorkspaceLayout, workspace_name, WORKSPACE_SUFFIX

# --------------------------------------------------------------------------
# old-layout directory-name grammar
# --------------------------------------------------------------------------
# A job directory:  job_<NNN>[_<sanitized name>]  — the integer id is what we
# key on; the trailing human name is discarded (it lives in job.json now).
_JOB_DIR_RE = re.compile(r"^job_(\d+)(?:_.*)?$")
# Per-job extracted frames:  sampling_<taskid>_job_<NNN>[_<name>]
_SAMPLING_JOB_RE = re.compile(r"^sampling_.+?_job_(\d+)(?:_.*)?$")
# Full-dataset extracted frames:  sampling_<taskid>_full
_SAMPLING_FULL_RE = re.compile(r"^sampling_.+?_full$")
# A jobs/ destination segment — job_NNN or the de-collided job_NNN_<slug> — for
# reading the id back out of a destination path in the summary.
_JOB_SEG_RE = re.compile(r"^job_(\d+)(?:_.*)?$")

# Top-level bundle directories, used to recover the bundle root from a plan's
# destination paths (execute_plan only receives the plan, not the root).
_BUNDLE_TOP = {"inputs", "survey", "jobs", "logs", "cache", "archive", "runs"}

# Sources handled specially rather than copied as a plain file/tree.
_WORKSPACE_JSON = "workspace.json"
_INTERP_FULL = "interp_full.csv"
_FILTERED_INTERP = "filtered_interp.csv"
_FILTERED_META = "filtered_interp.csv.meta.json"


# --------------------------------------------------------------------------
# plan model
# --------------------------------------------------------------------------
@dataclass
class MoveOp:
    """One copy operation in a migration plan.

    ``kind`` is ``"file"`` (copied with ``shutil.copy2``) or ``"tree"`` (copied
    with ``shutil.copytree(dirs_exist_ok=True)``).  Despite the name nothing is
    ever moved — it is always a copy — but "MoveOp" reads naturally in a UI that
    presents the migration as relocating the workspace into a bundle.
    """
    src: str
    dst: str
    kind: str  # "file" | "tree"


# --------------------------------------------------------------------------
# plan building (pure — reads the old workspace, writes nothing)
# --------------------------------------------------------------------------
def _job_dest_segments(old: Path) -> dict[str, str]:
    """Map each legacy ``job_<NNN>[_name]`` directory NAME → its destination
    segment under ``jobs/``.

    A numeric id used by exactly one legacy dir migrates to the clean
    ``job_NNN``.  When the OLD naming reused an id across several dirs (the
    ``_v3/_v5/_v6`` cruft — e.g. three ``job_023_*`` folders), they would all
    collapse onto ``jobs/job_023`` and overwrite each other; so colliding ids are
    disambiguated by appending the folder's name slug (``job_023_fulltest1`` …),
    keeping every job's data separate.
    """
    by_id: dict[int, list[str]] = {}
    for entry in old.iterdir():
        m = _JOB_DIR_RE.match(entry.name)
        if m and entry.is_dir():
            by_id.setdefault(int(m.group(1)), []).append(entry.name)
    segments: dict[str, str] = {}
    for job_id, names in by_id.items():
        collide = len(names) > 1
        for name in names:
            if collide:
                segments[name] = f"job_{job_id:03d}_{layout.slugify(name)}"
            else:
                segments[name] = f"job_{job_id:03d}"
    return segments


def build_migration_plan(old_ws: str, new_bundle: str,
                         skip_frames: bool = True) -> list[MoveOp]:
    """Enumerate every copy needed to migrate ``old_ws`` into ``new_bundle``.

    Pure: it only READS ``old_ws`` to list its contents and returns a
    deterministically ordered list of :class:`MoveOp`.  It performs no writes and
    creates no directories — ``execute_plan`` does that.

    ``skip_frames`` (default True) omits the extracted-frame stores — the
    ``sampling_*`` directories and any bare ``segment_*`` dirs inside a legacy job
    folder.  Those hold the bulk of a workspace's bytes (tens of GB of JPGs) and
    are fully regenerable from the source video, so copying them into the bundle
    is rarely wanted.  Products, interp, filtered interp, meshes and rasters are
    always migrated.

    Mapping (product/interp destinations minted via ``WorkspaceLayout``; colliding
    legacy job ids are de-collided, see :func:`_job_dest_segments`):

      * ``interp_full.csv``            → ``inputs/interp_full.csv``
      * ``outputs/<sub>``              → ``survey/<sub>``
      * ``job_NNN*/outputs/<sub>``     → ``jobs/<job seg>/products/<sub>``
      * ``job_NNN*/filtered_interp.csv`` → ``jobs/<job seg>/filtered_interp.csv``
      * ``sampling_*``                 → skipped (frames) unless skip_frames=False
      * ``anomaly_site_catalog/<sub>`` → ``survey/anomaly/<sub>``
      * ``logs/<sub>``                 → ``logs/<sub>``
      * ``workspace.json``             → consumed into ``project.json``
      * anything else                  → ``archive/imported/<original rel path>``
    """
    old = Path(old_ws)
    lo = WorkspaceLayout(new_bundle)
    ops: list[MoveOp] = []

    if not old.is_dir():
        return ops

    job_segments = _job_dest_segments(old)

    for entry in sorted(old.iterdir(), key=lambda p: p.name):
        name = entry.name

        if name == _WORKSPACE_JSON:
            # Folded into project.json, not copied as-is.
            continue

        if name == _INTERP_FULL and entry.is_file():
            ops.append(MoveOp(str(entry), str(lo.interp_full), "file"))

        elif name == "outputs" and entry.is_dir():
            ops += _map_children(entry, lo.products_dir("survey"))

        elif name == "anomaly_site_catalog" and entry.is_dir():
            ops += _map_children(entry, lo.products_dir("survey") / "anomaly")

        elif name == "logs" and entry.is_dir():
            ops += _map_children(entry, lo.logs_dir())

        elif _SAMPLING_FULL_RE.match(name) and entry.is_dir():
            if skip_frames:
                continue
            dst = lo.products_dir("survey") / "frames" / "run_001" / "segments"
            ops += _map_children(entry, dst)

        elif (m := _SAMPLING_JOB_RE.match(name)) and entry.is_dir():
            if skip_frames:
                continue
            job_id = int(m.group(1))
            ops += _map_children(entry, lo.frames_run_dir(job_id, run_id=1))

        elif _JOB_DIR_RE.match(name) and entry.is_dir():
            seg = job_segments.get(name, name)
            ops += _map_job_dir(entry, old, lo, seg, skip_frames)

        else:
            # Unrecognized — never dropped; parked under archive/imported/
            # preserving its original relative path.
            ops += _map_unrecognized(entry, old, lo)

    return ops


def _map_children(src_dir: Path, dst_dir: Path) -> list[MoveOp]:
    """Map each immediate child of ``src_dir`` into ``dst_dir`` (sorted).

    Working at the immediate-child granularity keeps ``run_NNN`` / channel /
    segment subtrees intact under one ``copytree`` while still giving the plan
    (and its summary) a meaningful per-item count.
    """
    ops: list[MoveOp] = []
    for child in sorted(src_dir.iterdir(), key=lambda p: p.name):
        kind = "tree" if child.is_dir() else "file"
        ops.append(MoveOp(str(child), str(dst_dir / child.name), kind))
    return ops


# Bare frame-store dir names that can appear directly inside a legacy job dir.
_FRAME_DIR_RE = re.compile(r"^(segment_\d+|frames|frames_annotated|frames_clahe|sensors)")


def _map_job_dir(job_dir: Path, old_ws: Path, lo: WorkspaceLayout,
                 seg: str, skip_frames: bool) -> list[MoveOp]:
    """Map one ``job_NNN*`` directory into ``jobs/<seg>/`` (seg de-collided)."""
    job_base = lo.root / "jobs" / seg
    ops: list[MoveOp] = []
    for child in sorted(job_dir.iterdir(), key=lambda p: p.name):
        name = child.name
        if name == "outputs" and child.is_dir():
            ops += _map_children(child, job_base / "products")
        elif name == _FILTERED_INTERP and child.is_file():
            ops.append(MoveOp(str(child), str(job_base / _FILTERED_INTERP), "file"))
        elif name == _FILTERED_META and child.is_file():
            ops.append(MoveOp(str(child), str(job_base / _FILTERED_META), "file"))
        elif skip_frames and child.is_dir() and _FRAME_DIR_RE.match(name):
            continue                          # regenerable frames — don't copy
        else:
            ops += _map_unrecognized(child, old_ws, lo)
    return ops


def _map_unrecognized(entry: Path, old_ws: Path, lo: WorkspaceLayout) -> list[MoveOp]:
    """Park an unrecognized path under ``archive/imported/<original rel path>``."""
    try:
        rel = entry.relative_to(old_ws)
    except ValueError:
        rel = Path(entry.name)
    kind = "tree" if entry.is_dir() else "file"
    return [MoveOp(str(entry), str(lo.archive_dir() / "imported" / rel), kind)]


# --------------------------------------------------------------------------
# project.json (folds in the old workspace.json)
# --------------------------------------------------------------------------
def _now_iso() -> str:
    """Naive-UTC ISO timestamp (no tzinfo — the app is single-timezone)."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def build_project_json(old_ws: str) -> dict:
    """Build the bundle's ``project.json`` from the old ``workspace.json``.

    Copies whatever fields the old manifest had, then stamps the migration
    provenance.  Missing / unreadable ``workspace.json`` is not an error — the
    stamped fields alone are returned.
    """
    old = Path(old_ws)
    data: dict = {}
    wj = old / _WORKSPACE_JSON
    if wj.is_file():
        try:
            loaded = json.loads(wj.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (json.JSONDecodeError, OSError):
            data = {}
    data["schema"] = "eprproj"
    data["migrated_from"] = str(old.resolve())
    data["migrated_at"] = _now_iso()
    return data


# --------------------------------------------------------------------------
# summary (what a UI shows before the user confirms)
# --------------------------------------------------------------------------
def _path_bytes(path: Path) -> int:
    """Total bytes of a file, or of every file under a directory."""
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            total = 0
            for root, _dirs, files in os.walk(path):
                for f in files:
                    try:
                        total += (Path(root) / f).stat().st_size
                    except OSError:
                        pass
            return total
    except OSError:
        return 0
    return 0


def _op_bytes(op: MoveOp) -> int:
    return _path_bytes(Path(op.src))


def _human_bytes(n: int) -> str:
    step = 1024.0
    val = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < step or unit == "TB":
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= step
    return f"{val:.1f} TB"


def summarize_plan(plan: list[MoveOp]) -> str:
    """Short human-readable summary grouped by destination area.

    Example::

        Migration plan: 7 operations, 1.2 MB total
        Inputs: interp_full.csv
        Survey products: 1 item
        Survey anomaly: 1 item
        Job 002: frames + products (2 items)
        Logs: 1 item
    """
    inputs: list[str] = []
    survey_products = survey_frames = survey_anomaly = 0
    logs = archive = 0
    jobs: dict[int, dict] = {}
    total_bytes = 0

    for op in plan:
        total_bytes += _op_bytes(op)
        parts = Path(op.dst).parts
        name = Path(op.dst).name

        if "inputs" in parts:
            inputs.append(name)
        elif "jobs" in parts:
            i = parts.index("jobs")
            seg = parts[i + 1] if i + 1 < len(parts) else ""
            m = _JOB_SEG_RE.match(seg)
            job_id = int(m.group(1)) if m else -1
            info = jobs.setdefault(job_id, {"areas": set(), "items": 0})
            info["items"] += 1
            tail = parts[i + 2:]
            if "frames" in tail:
                info["areas"].add("frames")
            elif "products" in tail:
                info["areas"].add("products")
            elif name == _FILTERED_INTERP or name == _FILTERED_META:
                info["areas"].add("filtered interp")
            else:
                info["areas"].add("other")
        elif "survey" in parts:
            tail = parts[parts.index("survey") + 1:]
            head = tail[0] if tail else ""
            if head == "frames":
                survey_frames += 1
            elif head == "anomaly":
                survey_anomaly += 1
            else:
                survey_products += 1
        elif "logs" in parts:
            logs += 1
        elif "archive" in parts:
            archive += 1

    def _items(n: int) -> str:
        return f"{n} item" if n == 1 else f"{n} items"

    lines = [f"Migration plan: {_items(len(plan))}, {_human_bytes(total_bytes)} total"]
    if inputs:
        lines.append("Inputs: " + ", ".join(inputs))
    if survey_products:
        lines.append(f"Survey products: {_items(survey_products)}")
    if survey_frames:
        lines.append(f"Survey frames: {_items(survey_frames)}")
    if survey_anomaly:
        lines.append(f"Survey anomaly: {_items(survey_anomaly)}")
    for job_id in sorted(jobs):
        info = jobs[job_id]
        # Present areas in a stable, readable order.
        order = ["frames", "products", "filtered interp", "other"]
        areas = [a for a in order if a in info["areas"]]
        label = " + ".join(areas) if areas else "items"
        lines.append(f"Job {job_id:03d}: {label} ({_items(info['items'])})")
    if logs:
        lines.append(f"Logs: {_items(logs)}")
    if archive:
        lines.append(f"Archive (unrecognized): {_items(archive)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# execution (the only part that touches the filesystem — copies only)
# --------------------------------------------------------------------------
def _bundle_root_from_plan(plan: list[MoveOp]) -> Optional[Path]:
    """Recover the bundle root from a plan's destination paths.

    Every destination is ``<root>/<top>/...`` where ``<top>`` is one of the
    known bundle directories; the prefix before the first such segment is the
    root.  Returns None for an empty / unrecognizable plan.
    """
    for op in plan:
        parts = Path(op.dst).parts
        for i, part in enumerate(parts):
            if part in _BUNDLE_TOP and i > 0:
                return Path(*parts[:i])
    return None


def execute_plan(plan: list[MoveOp], project_json: Optional[dict] = None,
                 progress: Optional[Callable[[int, int, MoveOp], None]] = None) -> dict:
    """Execute a migration plan by COPYING each op; never moves or deletes.

    Files are copied with ``shutil.copy2`` (metadata preserved); trees with
    ``shutil.copytree(dirs_exist_ok=True)``.  A single failed copy never aborts
    the migration — it is recorded in ``errors`` and the rest proceed, so a
    partial copy is always the worst case and the source is always intact.

    ``project_json``, when given, is written to ``<bundle>/project.json`` after
    the copies (``schema`` and ``migrated_at`` filled in if absent).

    ``progress`` is called as ``progress(done, total, op)`` after each op.

    Returns ``{"copied", "skipped", "errors", "bytes"}`` where ``errors`` is a
    list of ``{"src", "dst", "error"}`` dicts.
    """
    result: dict = {"copied": 0, "skipped": 0, "errors": [], "bytes": 0}
    total = len(plan)

    for i, op in enumerate(plan):
        try:
            src = Path(op.src)
            dst = Path(op.dst)
            if not src.exists():
                result["skipped"] += 1
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if op.kind == "tree":
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
                result["copied"] += 1
                result["bytes"] += _path_bytes(dst)
        except Exception as exc:  # never abort the whole migration on one copy
            result["errors"].append(
                {"src": op.src, "dst": op.dst, "error": str(exc)})
        if progress is not None:
            try:
                progress(i + 1, total, op)
            except Exception:
                pass

    if project_json is not None:
        root = _bundle_root_from_plan(plan)
        if root is not None:
            pj = dict(project_json)
            pj.setdefault("schema", "eprproj")
            pj.setdefault("migrated_at", _now_iso())
            try:
                root.mkdir(parents=True, exist_ok=True)
                (root / "project.json").write_text(
                    json.dumps(pj, indent=1), encoding="utf-8")
            except OSError as exc:
                result["errors"].append(
                    {"src": _WORKSPACE_JSON, "dst": str(root / "project.json"),
                     "error": str(exc)})

    return result


# --------------------------------------------------------------------------
# default destination
# --------------------------------------------------------------------------
def default_bundle_path(old_ws: str) -> str:
    """Sibling ``<parent>/<slugified basename>.eprproj`` for the old workspace."""
    old = Path(old_ws)
    name = workspace_name(old.name) + WORKSPACE_SUFFIX
    return str(old.parent / name)
