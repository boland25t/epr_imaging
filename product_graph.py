"""
product_graph.py — the declarative dependency backbone for the Product Tree UI.

This module is the single source of truth for *what products the app can make*
and *what each one depends on*.  It is deliberately Qt-free and side-effect-free
(pure data + pure functions), exactly like models.py / plan_service.py: the UI
layer reads this graph to draw a checkable "product tree", and the planner turns
a user's selection into a concrete Task stack.

The mental model is a directed acyclic graph (DAG) of PRODUCT NODES plus three
DATA-ROOT nodes ("video", "nav", "sensors").  An edge ``A requires B`` means "B
must be produced (or imported) before A can be".  Example chains:

    video ──▶ sampling ──▶ alignment ──▶ mesh / orthomosaic / dem / dense
    nav, sensors ──▶ interp ──▶ trackline / sensor_raster / anomaly / netcdf
    (everything) ──▶ report            (aggregate; no task of its own)

The photogrammetry branch is deliberately two-layered.  ``alignment`` is the
expensive, PERSISTED intermediate — Metashape's sparse cloud + solved camera
poses saved in the .psx project — and you never want to repeat it.  ``mesh``,
``orthomosaic``, ``dem`` and ``dense`` each consume that saved alignment and are
INDEPENDENT products: rebuilding the mesh, or adding a DEM later, must not re-run
alignment when it is already produced.  That is exactly what expand_targets()
delivers via the `satisfied` set (see its docstring / the tests).

Two things the UI does with this graph:

  * grey out nodes whose required data roots have not been imported yet
    (``available_nodes`` / ``unavailable_reason``); and
  * when the user checks a set of targets, compose the ordered list of nodes to
    run — each target plus every not-yet-produced ancestor, dependencies first
    (``expand_targets``).  "Already produced" comes from the manifest Registry;
    this module just takes the ``satisfied`` set and does the graph math.

Task-type mapping.  Each product node names the models.py task type it becomes
in a TaskStack (see TASK_INFO there).  Several nodes share the "photogrammetry"
task type because Metashape produces the sparse alignment and then mesh / DEM /
orthomosaic / dense-cloud from that same project, each configured by a build flag
(build_model, build_dem, …) rather than from separate task types — the node
distinction is a UI/product distinction, resolved to run settings later.  Pure
data roots and the aggregate "report" node have no task type (``task_type=None``).

Param schemas here are REPRESENTATIVE, not exhaustive — just enough for the UI to
render a sensible default form.  The authoritative per-task settings still live
in plan_service.py's step builders and are refined there; keep these in loose
sync but do not treat them as the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data roots
# ---------------------------------------------------------------------------
# The three kinds of imported input a product can transitively depend on.  These
# are modelled as nodes too (below) so availability logic is uniform, but the set
# is also exposed directly for callers that just want "what can be imported".
DATA_ROOTS: frozenset[str] = frozenset({"video", "nav", "sensors"})


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    """One vertex of the product DAG.

    id             stable slug, unique within the graph (the primary key).
    label          human-readable name for the UI.
    requires       parent node ids this node depends on (the DAG edges).  May
                   reference product nodes AND data-root nodes.
    needs_data     data roots this node *directly* consumes (subset of
                   DATA_ROOTS).  Roots inherited through a parent node are NOT
                   repeated here — use transitive_needs()/available_nodes() to
                   get the full picture.  A data-root node lists itself.
    task_type      the models.py task type this node maps to for the TaskStack,
                   or None for pure data roots and aggregate nodes.
    is_root        True for the data-root nodes (video/nav/sensors); these are
                   imported, never "run", and are excluded from expand_targets.
    is_intermediate  True for products the user usually gets on the way to
                   something else but can still select on their own (sampling,
                   photogrammetry, interp).
    is_archived    True for products retired from the UI but kept in the graph so
                   it stays complete and old references still resolve (dense
                   point cloud; mirrors models.ARCHIVED_TASK_TYPES in spirit).
    params         representative run-parameter schema: name -> {type, default,
                   label}.  Not the authoritative settings contract (see module
                   docstring).
    """

    id: str
    label: str
    requires: tuple[str, ...] = ()
    needs_data: frozenset[str] = frozenset()
    task_type: str | None = None
    is_root: bool = False
    is_intermediate: bool = False
    is_archived: bool = False
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PRODUCT_GRAPH — the single source of truth
# ---------------------------------------------------------------------------
# Ordered roughly dependencies-first for readability; expand_targets() does the
# real topological ordering so the source order here is not load-bearing.
_NODES: list[Node] = [
    # -- data roots ---------------------------------------------------------
    Node(
        id="video", label="Video", is_root=True,
        needs_data=frozenset({"video"}),
    ),
    Node(
        id="nav", label="Navigation", is_root=True,
        needs_data=frozenset({"nav"}),
    ),
    Node(
        id="sensors", label="Sensors", is_root=True,
        needs_data=frozenset({"sensors"}),
    ),

    # -- video / photogrammetry branch --------------------------------------
    Node(
        id="sampling", label="Sampling (frame extraction)",
        requires=("video",), needs_data=frozenset({"video"}),
        task_type="sampling", is_intermediate=True,
        params={
            # Real schema is refined in plan_service.build_sampling_config.
            "frames_per_chunk": {"type": "int",   "default": 300,  "label": "Frames per chunk"},
            "frame_rate":       {"type": "float", "default": 1.0,  "label": "Frame rate (Hz)"},
            "mode":             {"type": "choice", "default": "fixed",
                                 "label": "Sampling mode", "choices": ["fixed", "dynamic"]},
        },
    ),
    Node(
        # The expensive, PERSISTED intermediate: sparse cloud + solved camera
        # poses saved in the .psx project.  Every downstream photogrammetry
        # product consumes this without re-solving it — never repeat this step.
        id="alignment", label="Alignment (sparse + camera poses)",
        requires=("sampling",),
        task_type="photogrammetry", is_intermediate=True,
        params={
            "engine":         {"type": "choice", "default": "Metashape",
                               "label": "Engine", "choices": ["Metashape", "COLMAP"]},
            "align_accuracy": {"type": "choice", "default": "High",
                               "label": "Alignment accuracy",
                               "choices": ["Low", "Medium", "High", "Highest"]},
        },
    ),
    # dense/mesh/orthomosaic/dem each require ONLY the saved alignment, so any of
    # them can be (re)built independently without redoing alignment.  All map to
    # the "photogrammetry" task type but carry their own build flag.
    Node(
        id="dense", label="Dense point cloud",
        requires=("alignment",),
        task_type="photogrammetry", is_archived=True,
        params={
            "build_dense":   {"type": "bool", "default": True, "label": "Build dense cloud"},
            "dense_quality": {"type": "choice", "default": "Medium",
                              "label": "Dense quality",
                              "choices": ["Lowest", "Low", "Medium", "High", "Highest"]},
        },
    ),
    Node(
        id="mesh", label="Textured mesh",
        requires=("alignment",),
        task_type="photogrammetry",
        params={
            "build_model":   {"type": "bool", "default": True, "label": "Build model (mesh)"},
            "mesh_faces":    {"type": "choice", "default": "Medium",
                              "label": "Face count", "choices": ["Low", "Medium", "High"]},
            "build_texture": {"type": "bool", "default": True, "label": "Build texture"},
        },
    ),
    Node(
        id="orthomosaic", label="Orthomosaic",
        requires=("alignment",),
        task_type="photogrammetry",
        params={
            "build_orthomosaic": {"type": "bool", "default": True, "label": "Build orthomosaic"},
        },
    ),
    Node(
        id="dem", label="Digital elevation model",
        requires=("alignment",),
        task_type="photogrammetry",
        params={
            "build_dem":  {"type": "bool", "default": True,  "label": "Build DEM"},
            "export_dem": {"type": "bool", "default": True,  "label": "Export DEM GeoTIFF"},
        },
    ),

    # -- nav / sensor branch -------------------------------------------------
    Node(
        id="interp", label="Interpolated table (interp_full.csv)",
        requires=("nav", "sensors"), needs_data=frozenset({"nav", "sensors"}),
        task_type="build_interp", is_intermediate=True,
        params={
            "sample_hz": {"type": "float", "default": 1.0, "label": "Resample rate (Hz)"},
        },
    ),
    Node(
        id="trackline", label="Nav trackline PLY",
        requires=("interp",),
        task_type="nav_3d",
        params={
            "cell_size": {"type": "float", "default": 1.0, "label": "Cell size (m)"},
        },
    ),
    Node(
        id="sensor_raster", label="Sensor raster (GeoTIFF)",
        requires=("interp",),
        task_type="sensor_2d",
        params={
            "cell_size": {"type": "float", "default": 5.0, "label": "Cell size (m)"},
            "crs":       {"type": "choice", "default": "UTM",
                          "label": "CRS", "choices": ["UTM", "WGS84"]},
            "fill":      {"type": "choice", "default": "IDW fill",
                          "label": "Fill method",
                          "choices": ["IDW fill", "Kriging fill", "RBF fill", "No fill"]},
        },
    ),
    Node(
        id="anomaly", label="Anomaly detection + catalog",
        requires=("interp",),
        task_type="anomaly_detect",
        params={
            "run_detector": {"type": "bool", "default": True, "label": "Run detector"},
            "run_catalog":  {"type": "bool", "default": True, "label": "Build catalog"},
        },
    ),
    Node(
        id="netcdf", label="Sensor NetCDF (CF)",
        requires=("interp",),
        task_type="sensor_netcdf",
        params={
            "cell_size":   {"type": "float",  "default": 1.0, "label": "Cell size (m)"},
            "aggregation": {"type": "choice", "default": "mean",
                            "label": "Aggregation", "choices": ["mean", "median", "min", "max"]},
        },
    ),

    # -- aggregate -----------------------------------------------------------
    Node(
        id="report", label="Survey report",
        # Aggregates the survey's principal products.  No task type of its own:
        # selecting it pulls in its dependencies, which are the real work.
        requires=("trackline", "sensor_raster", "anomaly", "orthomosaic", "dem"),
        task_type=None,
        params={},
    ),
]

# id -> Node, the lookup callers use.  This dict IS the graph.
PRODUCT_GRAPH: dict[str, Node] = {n.id: n for n in _NODES}


# ---------------------------------------------------------------------------
# Basic accessors
# ---------------------------------------------------------------------------
def all_nodes() -> list[Node]:
    """Every node in declaration order (data roots first, then products)."""
    return list(_NODES)


def get_node(node_id: str) -> Node:
    """The node with this id, or KeyError if it does not exist."""
    return PRODUCT_GRAPH[node_id]


def data_root_nodes() -> list[Node]:
    """The three importable data-root nodes."""
    return [n for n in _NODES if n.is_root]


# ---------------------------------------------------------------------------
# Graph traversal helpers
# ---------------------------------------------------------------------------
def ancestors(node_id: str) -> set[str]:
    """All node ids this node transitively `requires` (its dependencies).

    Excludes the node itself.  Includes any data-root ancestors.
    """
    seen: set[str] = set()

    def walk(nid: str) -> None:
        for parent in PRODUCT_GRAPH[nid].requires:
            if parent not in seen:
                seen.add(parent)
                walk(parent)

    walk(node_id)
    return seen


def descendants(node_id: str) -> set[str]:
    """All node ids that transitively `require` this node (its dependents)."""
    seen: set[str] = set()

    def walk(nid: str) -> None:
        for other in PRODUCT_GRAPH.values():
            if nid in other.requires and other.id not in seen:
                seen.add(other.id)
                walk(other.id)

    walk(node_id)
    return seen


def transitive_needs(node_id: str) -> set[str]:
    """Every data root this node needs, directly or through any ancestor.

    A node is runnable/available exactly when this set is a subset of the
    imported roots.  Data-root nodes report themselves.
    """
    needs: set[str] = set(PRODUCT_GRAPH[node_id].needs_data)
    for anc in ancestors(node_id):
        needs |= PRODUCT_GRAPH[anc].needs_data
    return needs


# ---------------------------------------------------------------------------
# Availability (data-import gating)
# ---------------------------------------------------------------------------
def available_nodes(imported: set[str]) -> set[str]:
    """Ids of every node whose transitive data requirements are all imported.

    `imported` is a subset of DATA_ROOTS.  A product node is available iff each
    data root it transitively needs is present; a data-root node is available
    iff it is itself imported.
    """
    imported = set(imported)
    return {
        nid for nid in PRODUCT_GRAPH
        if transitive_needs(nid) <= imported
    }


def unavailable_reason(node_id: str, imported: set[str]) -> str | None:
    """Tooltip text for a greyed-out node, or None if the node IS available.

    e.g. ``"needs video"`` or ``"needs nav, sensors"``.
    """
    missing = transitive_needs(node_id) - set(imported)
    if not missing:
        return None
    return "needs " + ", ".join(sorted(missing))


# ---------------------------------------------------------------------------
# Target expansion (the "stack composes itself" logic)
# ---------------------------------------------------------------------------
def expand_targets(selected: list[str],
                   satisfied: set[str] = frozenset()) -> list[str]:
    """Topologically ordered node ids to RUN to produce every selected target.

    For each selected node we emit it plus all of its transitive `requires`
    ancestors that are NOT already ``satisfied`` (i.e. not already produced per
    the registry), dependencies BEFORE dependents.  A satisfied node prunes the
    whole branch above it — if photogrammetry is already produced we neither run
    it nor re-run sampling.  Data-root nodes are never run and never appear.

    The order is deterministic: a post-order walk over `requires`, in `selected`
    order, deduped so each node appears exactly once (its first, dependency-
    respecting position).
    """
    satisfied = set(satisfied)
    ordered: list[str] = []
    emitted: set[str] = set()

    def visit(nid: str) -> None:
        node = PRODUCT_GRAPH[nid]
        # Roots are imported, not run; a satisfied/already-produced node prunes
        # itself and everything it needed to be produced.
        if node.is_root or nid in satisfied or nid in emitted:
            return
        for parent in node.requires:
            visit(parent)
        emitted.add(nid)
        ordered.append(nid)

    for target in selected:
        visit(target)
    return ordered


# ---------------------------------------------------------------------------
# Status badges
# ---------------------------------------------------------------------------
def node_status(node_id: str, produced_ids: set[str],
                available_ids: set[str]) -> str:
    """Badge for a node: "produced" | "available" | "unavailable".

    Takes PRECOMPUTED sets so this stays pure and decoupled from the registry:
    the UI passes `produced_ids` (from the manifest Registry) and `available_ids`
    (from ``available_nodes(imported)``).  Precedence: produced beats available
    beats unavailable.
    """
    if node_id in produced_ids:
        return "produced"
    if node_id in available_ids:
        return "available"
    return "unavailable"


# ---------------------------------------------------------------------------
# Integrity check (run at import time)
# ---------------------------------------------------------------------------
def validate_graph() -> None:
    """Assert the graph is well-formed: every reference resolves and it is a DAG.

    Called at import time so a malformed edit to PRODUCT_GRAPH fails loudly and
    immediately rather than producing subtly wrong stacks later.
    """
    # 1. Unique ids (list -> dict already collapses dupes; catch it explicitly).
    ids = [n.id for n in _NODES]
    assert len(ids) == len(set(ids)), f"duplicate node id(s): {ids}"

    # 2. Every requires / needs_data reference resolves.
    for node in _NODES:
        for parent in node.requires:
            assert parent in PRODUCT_GRAPH, (
                f"node {node.id!r} requires unknown node {parent!r}"
            )
        for root in node.needs_data:
            assert root in DATA_ROOTS, (
                f"node {node.id!r} needs unknown data root {root!r}"
            )
        # Only data-root nodes may declare themselves an importable root.
        if node.needs_data and not node.is_root:
            # non-root nodes may still name direct data roots (e.g. sampling);
            # that is fine — the assertion above already validated them.
            pass

    # 3. Acyclic: a DFS with a recursion stack detects any back edge.
    visiting: set[str] = set()
    done: set[str] = set()

    def check(nid: str, path: tuple[str, ...]) -> None:
        assert nid not in visiting, (
            f"cycle in product graph: {' -> '.join(path + (nid,))}"
        )
        if nid in done:
            return
        visiting.add(nid)
        for parent in PRODUCT_GRAPH[nid].requires:
            check(parent, path + (nid,))
        visiting.discard(nid)
        done.add(nid)

    for node in _NODES:
        check(node.id, ())


# Fail fast on a malformed graph the moment this module is imported.
validate_graph()
