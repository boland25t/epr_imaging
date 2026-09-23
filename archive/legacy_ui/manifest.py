"""
manifest.py — run provenance and deduplication.

Two artifacts, both Qt-free and built on layout.py:

  * a per-run ``run.json`` written next to each run's outputs, recording exactly
    what was run (task, scope, resolved settings), against what inputs (hashed
    by CONTENT, not path), with which engine, when, how long, and what it
    produced; and
  * a workspace ``runs/registry.json`` indexing every run, with a
    ``by_signature`` map that answers "has this exact run already been done?" in
    O(1).

The dedup key is a RUN SIGNATURE:

    signature = sha256( task_type + scope + settings + input-data-hash + engine )

Hashing the input DATA (not its path) is the crucial part — it catches the case
where interp_full.csv was edited in place, and it generalises the existing
intervals-fingerprint trick used for the filtered-interp cache.  Volatile,
location-dependent settings (output dirs, absolute input paths, run globs) are
stripped before hashing so the same intent produces the same signature across
machines and workspace moves.

This module computes and records; the decision to skip / prompt / run on a
duplicate is made by the caller (stack_runner / main_window) using ``lookup``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SIGNATURE_VERSION = 1
REGISTRY_SCHEMA = 1

# Settings keys that describe WHERE things go or WHICH interpreter/machine ran,
# not WHAT was computed.  Excluded from the signature so identical intent dedups
# across workspace location and machine.
_VOLATILE_SETTING_KEYS = {
    "output_dir", "output_root", "interp_path", "colmap_nav_csv", "nav_csv",
    "frame_dir", "workspace_dir", "_run_glob", "_scale_source_glob",
    "job_id", "log_fn", "file_log_fn", "log_callback", "status_callback",
}


# --------------------------------------------------------------------------
# hashing primitives
# --------------------------------------------------------------------------
def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, non-JSON → str."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _short(sig: str) -> str:
    """Last 8 hex chars of a 'sha256:...' string, for run-id readability."""
    return sig.split(":")[-1][:8]


# --------------------------------------------------------------------------
# input fingerprints
# --------------------------------------------------------------------------
def file_content_hash_cached(path: str | Path, cache_path: str | Path) -> str:
    """SHA-256 of a file, memoised in a JSON cache keyed by (size, mtime).

    interp_full.csv is large and hashed for every step; this recomputes only
    when the file actually changes — the same (size, mtime) guard the
    filtered-interp cache already relies on.
    """
    path = Path(path)
    st = path.stat()
    key = str(path.resolve())
    cache_path = Path(cache_path)
    cache: dict = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}
    hit = cache.get(key)
    if hit and hit.get("size") == st.st_size and hit.get("mtime") == st.st_mtime:
        return hit["sha256"]
    digest = sha256_file(path)
    cache[key] = {"size": st.st_size, "mtime": st.st_mtime, "sha256": digest}
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(canonical_json(cache), encoding="utf-8")
    except OSError:
        pass
    return digest


def frame_set_hash(frame_paths: list[str]) -> str:
    """Stable hash of a frame set: sorted (basename, size) pairs.

    Uses names+sizes rather than pixel bytes — cheap, and sufficient to detect
    that the SAME frames are being reprocessed.
    """
    items = []
    for p in sorted(frame_paths):
        try:
            items.append((Path(p).name, Path(p).stat().st_size))
        except OSError:
            items.append((Path(p).name, -1))
    return sha256_text(canonical_json(items))


def intervals_fingerprint(intervals) -> str:
    """Hash of a job's (start_unix, end_unix) interval pairs.

    Generalises MainWindow._job_intervals_fingerprint (which used sha1) to the
    sha256 used everywhere here.  `intervals` is any iterable of objects with
    .start_time / .end_time datetimes, or of (start, end) tuples.
    """
    pairs = []
    for iv in intervals:
        if hasattr(iv, "start_time"):
            a, b = iv.start_time, iv.end_time
            a = a.timestamp() if hasattr(a, "timestamp") else a
            b = b.timestamp() if hasattr(b, "timestamp") else b
        else:
            a, b = iv
        pairs.append((float(a), float(b)))
    pairs.sort()
    return sha256_text(canonical_json(pairs))


# --------------------------------------------------------------------------
# signature
# --------------------------------------------------------------------------
def settings_for_signature(settings: dict) -> dict:
    """Intent-bearing settings only: drop volatile/location/callable keys."""
    out = {}
    for k, v in (settings or {}).items():
        if k in _VOLATILE_SETTING_KEYS or callable(v):
            continue
        out[k] = v
    return out


def run_signature(task_type: str, scope_id: str, settings: dict,
                  input_hash: str, engine_key: str = "", channel: Optional[str] = None) -> str:
    """The dedup key. Identical (task, scope, channel, intent, data, engine) →
    identical signature."""
    payload = {
        "v": SIGNATURE_VERSION,
        "task_type": task_type,
        "scope_id": scope_id,
        "channel": channel or "",
        "settings": settings_for_signature(settings),
        "input": input_hash,
        "engine": engine_key,
    }
    return sha256_text(canonical_json(payload))


# --------------------------------------------------------------------------
# per-run manifest
# --------------------------------------------------------------------------
def new_run_id(signature: str) -> str:
    """Sortable, unique run id carrying the signature prefix:
    r_<UTC-timestamp>_<8 hex of signature>."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"r_{ts}_{_short(signature)}"


def new_execution_id() -> str:
    """One id shared by every run of a single stack invocation."""
    return "x_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


@dataclass
class RunRecord:
    """The contents of a run.json (and the source of a registry entry)."""
    run_id: str
    signature: str
    task_type: str
    scope_id: str
    status: str = "completed"            # completed | failed | partial | skipped
    channel: Optional[str] = None
    execution_id: Optional[str] = None
    target: dict = field(default_factory=dict)
    engine: str = ""
    inputs: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_s: float = 0.0
    error: Optional[str] = None
    outputs: list = field(default_factory=list)     # [{path, size, sha256?}]
    log: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": REGISTRY_SCHEMA,
            "run_id": self.run_id,
            "signature": self.signature,
            "task_type": self.task_type,
            "scope_id": self.scope_id,
            "channel": self.channel,
            "execution_id": self.execution_id,
            "target": self.target,
            "engine": self.engine,
            "status": self.status,
            "inputs": self.inputs,
            "settings": self.settings,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "error": self.error,
            "outputs": self.outputs,
            "log": self.log,
        }


# Products smaller than this get an output content hash; larger ones record
# size only, so writing a manifest never stalls on a multi-GB dense cloud/ortho.
_OUTPUT_HASH_MAX_BYTES = 256 * 1024 * 1024


def describe_outputs(paths: list[str], hash_max_bytes: int = _OUTPUT_HASH_MAX_BYTES) -> list:
    """[{path, size, sha256?}] for each existing output."""
    out = []
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            continue
        size = pp.stat().st_size if pp.is_file() else None
        entry = {"path": str(p), "size": size}
        if pp.is_file() and size is not None and size <= hash_max_bytes:
            try:
                entry["sha256"] = sha256_file(pp)
            except OSError:
                pass
        out.append(entry)
    return out


def write_manifest(run_dir: str | Path, record: RunRecord) -> Path:
    """Write run.json into a run directory. Returns its path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    path.write_text(json.dumps(record.to_dict(), indent=1), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
class Registry:
    """Workspace-level index of all runs, for dedup and history.

    Derived and rebuildable: if runs/registry.json is lost it can be
    reconstructed by scanning run.json files.  ``by_signature`` maps a signature
    to the latest COMPLETED run only, so a failed run never suppresses a retry.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.runs: list[dict] = []
        self.by_signature: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self.runs = data.get("runs", [])
        self.by_signature = data.get("by_signature", {})

    def lookup(self, signature: str) -> Optional[dict]:
        """The latest completed run entry for a signature, or None."""
        run_id = self.by_signature.get(signature)
        if not run_id:
            return None
        return next((r for r in self.runs if r["run_id"] == run_id), None)

    def group_latest(self, task_type: str, scope_id: str,
                     channel: Optional[str]) -> Optional[dict]:
        """Latest completed run of the same (task_type, scope, channel) group,
        regardless of signature — used to set the supersession chain."""
        matches = [r for r in self.runs
                   if r.get("task_type") == task_type and r.get("scope_id") == scope_id
                   and (r.get("channel") or None) == (channel or None)
                   and r.get("status") == "completed" and not r.get("superseded_by")]
        return matches[-1] if matches else None

    def record(self, rec: RunRecord, manifest_path: str | Path,
               output_dir: str | Path) -> dict:
        """Append a run and update the dedup/supersession indices."""
        entry = {
            "run_id": rec.run_id,
            "signature": rec.signature,
            "task_type": rec.task_type,
            "channel": rec.channel,
            "scope_id": rec.scope_id,
            "status": rec.status,
            "execution_id": rec.execution_id,
            "finished_at": rec.finished_at,
            "duration_s": rec.duration_s,
            "engine": rec.engine,
            "manifest_path": str(manifest_path),
            "output_dir": str(output_dir),
            "n_outputs": len(rec.outputs),
            "superseded_by": None,
        }
        if rec.status == "completed":
            # A genuinely new (changed) run supersedes the prior latest of its group.
            prior = self.group_latest(rec.task_type, rec.scope_id, rec.channel)
            if prior and prior["signature"] != rec.signature:
                prior["superseded_by"] = rec.run_id
            self.by_signature[rec.signature] = rec.run_id
        self.runs.append(entry)
        return entry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": REGISTRY_SCHEMA,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runs": self.runs,
            "by_signature": self.by_signature,
        }
        self.path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def outputs_exist(self, entry: dict) -> bool:
        """True if the run's manifest still lists outputs that are all present.

        A duplicate is only a real skip candidate if its products are still on
        disk; if the user deleted them, the run must happen again.
        """
        mp = Path(entry.get("manifest_path", ""))
        if not mp.is_file():
            return False
        try:
            man = json.loads(mp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        outs = man.get("outputs", [])
        return bool(outs) and all(Path(o["path"]).exists() for o in outs)
