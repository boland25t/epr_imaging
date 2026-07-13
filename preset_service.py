"""
preset_service.py — Persistent task & stack templates.

A **task template** is a reusable, target-free snapshot of one task:
    {"name": str, "task_type": str, "settings": dict, "channels": [str]}

A **stack template** is a named, ordered list of task templates (inline, so it
is self-contained):
    {"name": str, "tasks": [ {"task_type", "settings", "channels"}, ... ]}

Templates deliberately do NOT capture a target or task_id — the target is
chosen when the template is applied, which is what makes a template reusable
across surveys/datasets.

Storage is a single JSON file in the user's config directory (the same place
last_session.json lives), so templates persist across workspaces and reboots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import Task


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def presets_path() -> Path:
    """Path to presets.json in the user config dir (mirrors last_session.json)."""
    cfg_dir = Path.home() / ".config" / "epr_imaging"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "presets.json"


def load_presets() -> dict[str, list]:
    """Load all presets; returns {'task_templates': [...], 'stack_templates': [...]}."""
    path = presets_path()
    if not path.exists():
        return {"task_templates": [], "stack_templates": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"task_templates": [], "stack_templates": []}
    data.setdefault("task_templates", [])
    data.setdefault("stack_templates", [])
    return data


def save_presets(data: dict) -> None:
    presets_path().write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def task_to_template(task: Task) -> dict[str, Any]:
    """Snapshot one Task as a (target-free) template body."""
    return {
        "task_type": task.task_type,
        "settings":  dict(task.settings),
        "channels":  list(task.channels),
    }


def template_to_task(body: dict, task_id: int) -> Task:
    """Instantiate a Task from a template body with a fresh id + default target."""
    return Task(
        task_id=task_id,
        task_type=body["task_type"],
        target={"kind": "full"},      # target chosen at apply time
        settings=dict(body.get("settings", {})),
        channels=list(body.get("channels", [])),
        depends_on=None,              # links are stack-specific; reset on apply
    )


# ---------------------------------------------------------------------------
# Task templates
# ---------------------------------------------------------------------------

def list_task_templates() -> list[dict]:
    return load_presets().get("task_templates", [])


def add_task_template(name: str, task: Task) -> None:
    """Save (or overwrite by name) a task template."""
    data = load_presets()
    body = task_to_template(task)
    body["name"] = name
    data["task_templates"] = [t for t in data["task_templates"] if t.get("name") != name]
    data["task_templates"].append(body)
    save_presets(data)


def delete_task_template(name: str) -> None:
    data = load_presets()
    data["task_templates"] = [t for t in data["task_templates"] if t.get("name") != name]
    save_presets(data)


# ---------------------------------------------------------------------------
# Stack templates
# ---------------------------------------------------------------------------

def list_stack_templates() -> list[dict]:
    return load_presets().get("stack_templates", [])


def add_stack_template(name: str, tasks: list[Task]) -> None:
    """Save (or overwrite by name) an ordered stack of task templates.

    Intra-stack dependencies (e.g. photogrammetry → sampling) are preserved as a
    positional `depends_on_index` so they survive save/load even though task_ids
    are reassigned on apply.
    """
    data = load_presets()
    id_to_index = {t.task_id: i for i, t in enumerate(tasks)}
    bodies = []
    for t in tasks:
        b = task_to_template(t)
        if t.depends_on is not None and t.depends_on in id_to_index:
            b["depends_on_index"] = id_to_index[t.depends_on]
        bodies.append(b)
    body = {"name": name, "tasks": bodies}
    data["stack_templates"] = [s for s in data["stack_templates"] if s.get("name") != name]
    data["stack_templates"].append(body)
    save_presets(data)


def delete_stack_template(name: str) -> None:
    data = load_presets()
    data["stack_templates"] = [s for s in data["stack_templates"] if s.get("name") != name]
    save_presets(data)
