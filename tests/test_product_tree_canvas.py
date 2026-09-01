"""Tests for the Qt-free layout core of product_tree_canvas.

Covers the tier/row math the tech-tree canvas lays nodes out with: dependency
depth (data roots leftmost, each product one tier past the deepest thing it
requires), the branch grouping used for deterministic vertical ordering, and the
invariant that every graph node lands in exactly one tier exactly once.

Pure and Qt-free: product_tree_canvas imports PySide6 but compute_tiers /
tier_rows / branch_of construct no widgets, so these run headless without a
QApplication.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import product_graph as pg
import product_tree_canvas as ptc


# --------------------------------------------------------------------------
# compute_tiers — dependency depth
# --------------------------------------------------------------------------
def test_data_roots_are_tier_zero():
    tiers = ptc.compute_tiers()
    for root in pg.data_root_nodes():
        assert tiers[root.id] == 0


def test_photogrammetry_chain_depths():
    tiers = ptc.compute_tiers()
    # video(0) -> sampling(1) -> alignment(2) -> mesh/ortho/dem/dense(3)
    assert tiers["sampling"] == 1
    assert tiers["alignment"] == 2
    for leaf in ("mesh", "orthomosaic", "dem", "dense"):
        assert tiers[leaf] == 3


def test_nav_sensor_chain_depths():
    tiers = ptc.compute_tiers()
    # nav/sensors(0) -> interp(1) -> trackline/sensor_raster/anomaly/netcdf(2)
    assert tiers["interp"] == 1
    for leaf in ("trackline", "sensor_raster", "anomaly", "netcdf"):
        assert tiers[leaf] == 2


def test_report_is_one_past_its_deepest_requirement():
    tiers = ptc.compute_tiers()
    # report requires the tier-3 photogrammetry leaves, so it sits at tier 4.
    assert tiers["report"] == 1 + max(tiers[r] for r in pg.get_node("report").requires)
    assert tiers["report"] == 4


def test_tier_is_one_past_max_requirement_for_every_node():
    tiers = ptc.compute_tiers()
    for node in pg.all_nodes():
        if node.requires:
            assert tiers[node.id] == 1 + max(tiers[r] for r in node.requires)
        else:
            assert tiers[node.id] == 0


# --------------------------------------------------------------------------
# branch_of — vertical grouping
# --------------------------------------------------------------------------
def test_branch_assignment():
    assert ptc.branch_of("sampling") == ptc.BRANCH_VIDEO
    assert ptc.branch_of("mesh") == ptc.BRANCH_VIDEO
    assert ptc.branch_of("interp") == ptc.BRANCH_NAV
    assert ptc.branch_of("sensor_raster") == ptc.BRANCH_NAV
    # report pulls from both chains -> aggregate.
    assert ptc.branch_of("report") == ptc.BRANCH_AGGREGATE


# --------------------------------------------------------------------------
# tier_rows — layout helper
# --------------------------------------------------------------------------
def test_tier_rows_cover_every_node_exactly_once():
    rows = ptc.tier_rows()
    placed = [nid for ids in rows.values() for nid in ids]
    assert sorted(placed) == sorted(n.id for n in pg.all_nodes())
    assert len(placed) == len(set(placed))       # no duplicates


def test_tier_rows_group_video_before_nav_within_a_tier():
    rows = ptc.tier_rows()
    # Tier 1 holds sampling (video branch) and interp (nav branch); video first.
    tier1 = rows[1]
    assert tier1.index("sampling") < tier1.index("interp")


def test_tier_rows_are_ordered_by_branch_then_declaration():
    rows = ptc.tier_rows()
    decl = {n.id: i for i, n in enumerate(pg.all_nodes())}
    for ids in rows.values():
        keys = [(ptc._BRANCH_RANK.get(ptc.branch_of(n), 9), decl[n]) for n in ids]
        assert keys == sorted(keys)
