"""Tests for the Qt-free helpers of workbench.py.

Covers the pure logic the three-pane workbench builds its navigator from:
scope-list assembly (whole-trackline default first, injected + registry scopes
appended, de-duped), the product-node outline grouping, the spatial-node
classification, and the outline badge glyph.  All Qt-free: workbench imports
PySide6 but these helpers construct no widgets, so they run headless without a
QApplication.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import product_graph as pg
import workbench as wb
from product_tree_widget import SURVEY_SCOPE


class _FakeRegistry:
    """Minimal stand-in exposing `.runs`, like manifest.Registry."""

    def __init__(self, runs):
        self.runs = runs


# --------------------------------------------------------------------------
# assemble_scopes
# --------------------------------------------------------------------------
def test_assemble_scopes_default_is_whole_trackline():
    scopes = wb.assemble_scopes(None)
    assert scopes[0] == (SURVEY_SCOPE, wb.WHOLE_TRACKLINE_LABEL)
    assert len(scopes) == 1


def test_assemble_scopes_appends_injected_then_registry():
    reg = _FakeRegistry([
        {"scope_id": "job_2", "status": "completed", "task_type": "sampling"},
        {"scope_id": "survey", "status": "completed", "task_type": "nav_3d"},
    ])
    scopes = wb.assemble_scopes(reg, injected=[("int_a", "Interval A")])
    ids = [sid for sid, _ in scopes]
    # survey first, then the injected interval, then the registry scope.
    assert ids[0] == SURVEY_SCOPE
    assert ids[1] == "int_a"
    assert "job_2" in ids
    assert ids.index("int_a") < ids.index("job_2")


def test_assemble_scopes_dedupes_and_folds_survey_aliases():
    reg = _FakeRegistry([{"scope_id": "int_a", "status": "completed",
                          "task_type": "sampling"}])
    # An injected "full" alias must NOT create a second survey row, and an injected
    # scope already discovered from the registry must not duplicate.
    scopes = wb.assemble_scopes(
        reg, injected=[("full", "dupe"), ("int_a", "Interval A")])
    ids = [sid for sid, _ in scopes]
    assert ids.count(SURVEY_SCOPE) == 1
    assert "full" not in ids
    assert ids.count("int_a") == 1


def test_assemble_scopes_uses_injected_label():
    scopes = wb.assemble_scopes(None, injected=[("int_a", "My Interval")])
    assert ("int_a", "My Interval") in scopes


# --------------------------------------------------------------------------
# outline_groups
# --------------------------------------------------------------------------
def test_outline_groups_cover_every_node_once():
    placed = [nid for _label, ids in wb.outline_groups() for nid in ids]
    assert sorted(placed) == sorted(n.id for n in pg.all_nodes())
    assert len(placed) == len(set(placed))


def test_outline_groups_order_and_membership():
    groups = wb.outline_groups()
    labels = [label for label, _ in groups]
    assert labels[0] == "Video / Photogrammetry"
    assert labels[1] == "Navigation / Sensors"
    by_label = {label: ids for label, ids in groups}
    assert "video" in by_label["Video / Photogrammetry"]
    assert "sampling" in by_label["Video / Photogrammetry"]
    assert "nav" in by_label["Navigation / Sensors"]
    assert "interp" in by_label["Navigation / Sensors"]
    assert "report" in by_label["Aggregate"]


def test_outline_groups_keep_declaration_order_within_group():
    decl = {n.id: i for i, n in enumerate(pg.all_nodes())}
    for _label, ids in wb.outline_groups():
        keys = [decl[n] for n in ids]
        assert keys == sorted(keys)


# --------------------------------------------------------------------------
# is_spatial_node
# --------------------------------------------------------------------------
def test_is_spatial_node():
    assert wb.is_spatial_node("orthomosaic")
    assert wb.is_spatial_node("trackline")
    assert not wb.is_spatial_node("sampling")
    assert not wb.is_spatial_node("report")


# --------------------------------------------------------------------------
# node_badge
# --------------------------------------------------------------------------
def test_node_badge_states():
    produced = {"trackline"}
    available = {"trackline", "sensor_raster"}
    imported = {"nav", "sensors"}
    assert wb.node_badge("nav", produced, available, imported) == "✓"     # root imported
    assert wb.node_badge("video", produced, available, imported) == "✗"   # root missing
    assert wb.node_badge("trackline", produced, available, imported) == "✓"  # produced
    assert wb.node_badge("sensor_raster", produced, available, imported) == "○"  # available
    assert wb.node_badge("mesh", produced, available, imported) == "✗"    # locked
