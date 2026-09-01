"""Tests for product_graph — the declarative product-tree dependency backbone.

Pure and Qt-free (like the module under test): no tmp files, no QApplication,
no registry — just the graph math.

Covered:
  * the graph is acyclic and every requires/needs_data reference resolves
  * availability gating by imported data roots + the greyed-out reason text
  * expand_targets: ancestor pull-in, satisfied-branch pruning, shared ancestors
    emitted once, dependencies before dependents, roots never run
  * an intermediate can be selected on its own
  * the archived dense node is present but flagged
  * node_status badge precedence
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import product_graph as pg


# --------------------------------------------------------------------------
# graph integrity
# --------------------------------------------------------------------------
def test_validate_graph_passes():
    # Import already ran it; calling again must not raise.
    pg.validate_graph()


def test_all_references_resolve():
    for node in pg.all_nodes():
        for parent in node.requires:
            assert parent in pg.PRODUCT_GRAPH
        for root in node.needs_data:
            assert root in pg.DATA_ROOTS


def test_graph_is_acyclic():
    # descendants(x) must never include x (no node depends on itself).
    for node in pg.all_nodes():
        assert node.id not in pg.descendants(node.id)


def test_expected_ids_present():
    expected = {
        "video", "nav", "sensors",
        "sampling", "alignment", "dense", "mesh", "orthomosaic", "dem",
        "interp", "trackline", "sensor_raster", "anomaly", "netcdf", "report",
    }
    assert expected <= set(pg.PRODUCT_GRAPH)


def test_get_node_and_edges():
    assert pg.get_node("sampling").requires == ("video",)
    assert pg.get_node("alignment").requires == ("sampling",)
    # mesh/dem/ortho/dense hang off the SHARED alignment, not each other.
    for nid in ("mesh", "orthomosaic", "dem", "dense"):
        assert pg.get_node(nid).requires == ("alignment",)
    assert set(pg.get_node("interp").requires) == {"nav", "sensors"}
    # report aggregates the principal products.
    assert "trackline" in pg.get_node("report").requires
    assert "orthomosaic" in pg.get_node("report").requires


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------
def test_available_with_nav_and_sensors_only():
    avail = pg.available_nodes({"nav", "sensors"})
    # interp branch is fully available without video.
    for nid in ("interp", "trackline", "sensor_raster", "anomaly", "netcdf"):
        assert nid in avail, nid
    # video branch is NOT.
    for nid in ("sampling", "alignment", "mesh", "orthomosaic", "dem"):
        assert nid not in avail, nid
    # report needs orthomosaic/dem (video), so it is unavailable too.
    assert "report" not in avail


def test_available_with_everything():
    avail = pg.available_nodes({"video", "nav", "sensors"})
    for node in pg.all_nodes():
        if node.is_archived:
            continue
        assert node.id in avail, node.id


def test_available_roots_gated_by_import():
    assert "video" in pg.available_nodes({"video"})
    assert "video" not in pg.available_nodes({"nav"})


def test_unavailable_reason_mentions_missing_root():
    reason = pg.unavailable_reason("sampling", {"nav", "sensors"})
    assert reason is not None
    assert "video" in reason
    # available node -> no reason
    assert pg.unavailable_reason("interp", {"nav", "sensors"}) is None
    # interp with nothing imported names both roots
    reason2 = pg.unavailable_reason("interp", set())
    assert "nav" in reason2 and "sensors" in reason2


# --------------------------------------------------------------------------
# expand_targets
# --------------------------------------------------------------------------
def test_expand_mesh_from_scratch():
    assert pg.expand_targets(["mesh"]) == ["sampling", "alignment", "mesh"]


def test_expand_mesh_with_alignment_satisfied():
    # Rebuild mesh alone: alignment already solved, so it is NOT re-run.
    got = pg.expand_targets(["mesh"], satisfied={"sampling", "alignment"})
    assert got == ["mesh"]


def test_expand_shares_alignment_once():
    got = pg.expand_targets(["orthomosaic", "dem"])
    # sampling + alignment appear once each, before their dependents.
    assert got.count("sampling") == 1
    assert got.count("alignment") == 1
    assert got.index("sampling") < got.index("alignment")
    assert got.index("alignment") < got.index("orthomosaic")
    assert got.index("alignment") < got.index("dem")
    assert set(got) == {"sampling", "alignment", "orthomosaic", "dem"}


def test_expand_independent_products_skip_alignment_when_satisfied():
    # ortho + dem with alignment already produced: each is independent and
    # alignment is never repeated.
    got = pg.expand_targets(["orthomosaic", "dem"],
                            satisfied={"sampling", "alignment"})
    assert got == ["orthomosaic", "dem"]
    # And selecting mesh then dem separately never re-runs alignment.
    assert pg.expand_targets(["mesh"], satisfied={"sampling", "alignment"}) == ["mesh"]
    assert pg.expand_targets(["dem"], satisfied={"sampling", "alignment"}) == ["dem"]


def test_expand_intermediate_alone():
    assert pg.expand_targets(["sampling"]) == ["sampling"]
    assert pg.expand_targets(["interp"]) == ["interp"]


def test_expand_excludes_data_roots():
    got = pg.expand_targets(["interp", "trackline"])
    assert "nav" not in got and "sensors" not in got
    assert got == ["interp", "trackline"]


def test_expand_report_pulls_full_tree_in_order():
    got = pg.expand_targets(["report"])
    # report itself is last (depends on everything else selected).
    assert got[-1] == "report"
    # every dependency precedes its dependents.
    assert got.index("sampling") < got.index("alignment")
    assert got.index("alignment") < got.index("orthomosaic")
    assert got.index("interp") < got.index("trackline")
    # no data roots.
    assert not (pg.DATA_ROOTS & set(got))


# --------------------------------------------------------------------------
# archived node
# --------------------------------------------------------------------------
def test_dense_node_present_but_archived():
    dense = pg.get_node("dense")
    assert dense.is_archived is True
    assert dense.requires == ("alignment",)
    # only dense is archived among the current graph.
    archived = {n.id for n in pg.all_nodes() if n.is_archived}
    assert archived == {"dense"}


def test_intermediates_flagged():
    for nid in ("sampling", "alignment", "interp"):
        assert pg.get_node(nid).is_intermediate is True
    # the photogrammetry products are independent, NOT intermediates.
    for nid in ("mesh", "orthomosaic", "dem", "dense"):
        assert pg.get_node(nid).is_intermediate is False


# --------------------------------------------------------------------------
# node_status
# --------------------------------------------------------------------------
def test_node_status_badges():
    available = pg.available_nodes({"nav", "sensors"})
    produced = {"interp"}
    # produced beats available
    assert pg.node_status("interp", produced, available) == "produced"
    # available but not produced
    assert pg.node_status("trackline", produced, available) == "available"
    # not available (needs video)
    assert pg.node_status("sampling", produced, available) == "unavailable"
