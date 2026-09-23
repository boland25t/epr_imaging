"""
product_browser.py — the single-click "see everything this dive produced" window.

Two layers, cleanly split:

  * ``discover_products(workspace_dir)`` — Qt-free, tested.  It reads the
    provenance registry written by stack_runner (runs/registry.json → each run's
    run.json → its outputs) and, so that pre-registry and hand-made outputs are
    never invisible, also scans the workspace's product folders directly.
    Registry metadata (task, engine, when, superseded) enriches a file when it
    is present; a bare file on disk still shows up without it.

  * ``ProductBrowserDialog`` — a simple viewer: a grouped, filterable tree on
    the left, a preview + provenance panel on the right, and one-click actions
    (open externally, reveal in folder, open a mesh in the 3D viewer).  This is
    the "simple window" companion to the heavier 3D overlay viewer.

The browser never mutates a workspace; it only reads.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Legacy per-job directory: job_<NNN>_<name> (name optional).
_LEGACY_JOB_RE = re.compile(r"^job_(\d+)(?:_.*)?$")

# ---------------------------------------------------------------------------
# What counts as a product, and how we bucket it
# ---------------------------------------------------------------------------
# Extension → coarse kind.  Kind drives the icon, the grouping, and which
# preview/action the right-hand panel offers.
_EXT_KIND = {
    ".tif": "raster", ".tiff": "raster",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".obj": "mesh", ".ply": "mesh", ".stl": "mesh", ".glb": "mesh",
    ".csv": "table",
    ".nc": "dataset",
    ".pdf": "report", ".html": "report", ".htm": "report",
    ".psx": "project", ".psz": "project",
    ".json": "metadata",
}

# Product file extensions the filesystem fallback will surface.  Deliberately
# excludes raster IMAGE formats (.jpg = extracted frames; .png = archived depth
# slices — thousands of each, and not products in the redesigned world where
# rasters are GeoTIFF).  A PNG a run explicitly records still shows: registry
# products are not filtered by extension, only this bare-file scan is.
_SCAN_EXTS = {".tif", ".tiff", ".obj", ".ply", ".stl", ".glb",
              ".csv", ".nc", ".pdf", ".html", ".psx"}

# Directory names never descended into during the fallback scan: frame stores,
# raw inputs, caches, and Metashape's internal per-chunk scratch.  These are
# extraction/working data, not products.
_SCAN_PRUNE_DIRS = {"frames", "frames_annotated", "frames_clahe", "segments",
                    "sensors", "chunks", "inputs", "cache", "logs", "archive",
                    ".git", "__pycache__", ".claude"}

# Directory NAME PREFIXES pruned too — the legacy frame-extraction stores
# (sampling_<taskid>_… and segment_<NNN>_…) hold thousands of frame images that
# must never be surfaced as products.
_SCAN_PRUNE_PREFIXES = ("sampling_", "segment_")

# Directory NAME SUFFIXES pruned — a Metashape project keeps its internals in
# "<name>.files/" (depth maps, the tiled ortho/DEM pyramid: tile-*.tif). Those
# are engine-internal storage, not products — only the EXPORTED orthomosaic.tif /
# dem.tif / mesh.obj (which live outside .files/) are real products.
_SCAN_PRUNE_SUFFIXES = (".files",)

# Roots under a workspace where products actually live.  Scanning is confined to
# these so a workspace that sits next to a 114 GB data folder is never traversed.
_PRODUCT_ROOTS = ("outputs", "products", "jobs", "survey", "photogrammetry")

_MAX_SCAN_FILES = 5000        # hard ceiling; a real dive is far below this


def _kind_for(path: str, task_type: Optional[str]) -> str:
    ext = Path(path).suffix.lower()
    kind = _EXT_KIND.get(ext, "other")
    # A GeoTIFF's task_type disambiguates raster vs DEM vs orthomosaic.
    if kind == "raster" and task_type:
        if "orthomosaic" in task_type or task_type == "orthomosaic":
            return "orthomosaic"
        if "dem" in task_type:
            return "dem"
    return kind


@dataclass
class ProductItem:
    """One product file, with whatever provenance we could attach to it."""
    path: str
    kind: str = "other"
    task_type: Optional[str] = None
    scope_id: Optional[str] = None
    channel: Optional[str] = None
    run_id: Optional[str] = None
    finished_at: Optional[str] = None
    engine: Optional[str] = None
    duration_s: Optional[float] = None
    status: Optional[str] = None
    superseded: bool = False
    size: Optional[int] = None
    source: str = "scan"            # "registry" | "scan"

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def exists(self) -> bool:
        return Path(self.path).exists()

    @property
    def scope_label(self) -> str:
        s = self.scope_id or ""
        if not s or s in ("full", "—"):
            return "Survey"
        if s == "survey":
            return "Survey"
        if s.startswith("job"):
            return s.replace("_", " ").title()
        return f"Job {s}"


def _human_size(n: Optional[int]) -> str:
    if not n:
        return ""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


# ---------------------------------------------------------------------------
# discovery (Qt-free, tested)
# ---------------------------------------------------------------------------
def _registry_products(workspace: Path) -> tuple[list[ProductItem], set[str]]:
    """Products drawn from runs/registry.json + each run's run.json.

    Returns (items, seen_paths).  ``seen_paths`` are resolved absolute paths
    already accounted for, so the filesystem scan doesn't duplicate them.
    """
    items: list[ProductItem] = []
    seen: set[str] = set()
    reg_path = workspace / "runs" / "registry.json"
    if not reg_path.is_file():
        return items, seen
    try:
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return items, seen

    for entry in reg.get("runs", []):
        if entry.get("status") != "completed":
            continue
        mp = entry.get("manifest_path", "")
        outputs = []
        if mp and Path(mp).is_file():
            try:
                man = json.loads(Path(mp).read_text(encoding="utf-8"))
                outputs = man.get("outputs", [])
            except (json.JSONDecodeError, OSError):
                outputs = []
        for out in outputs:
            p = out.get("path")
            if not p:
                continue
            rp = str(Path(p).resolve())
            seen.add(rp)
            items.append(ProductItem(
                path=p,
                kind=_kind_for(p, entry.get("task_type")),
                task_type=entry.get("task_type"),
                scope_id=entry.get("scope_id"),
                channel=entry.get("channel"),
                run_id=entry.get("run_id"),
                finished_at=entry.get("finished_at"),
                engine=entry.get("engine"),
                duration_s=entry.get("duration_s"),
                status=entry.get("status"),
                superseded=bool(entry.get("superseded_by")),
                size=out.get("size"),
                source="registry",
            ))
    return items, seen


def _scan_products(workspace: Path, seen: set[str]) -> list[ProductItem]:
    """Fallback: walk the workspace's product roots for product files.

    Bounded (pruned dirs, capped file count, no .jpg) so it is safe to run on
    any workspace regardless of what sits beside it.
    """
    items: list[ProductItem] = []
    roots = [workspace] + [workspace / r for r in _PRODUCT_ROOTS]
    # A .eprproj bundle keeps everything under itself; treat the bundle as root.
    walked = 0
    visited_dirs: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath in visited_dirs:
                dirnames[:] = []
                continue
            visited_dirs.add(dirpath)
            dirnames[:] = [
                d for d in dirnames
                if d.lower() not in _SCAN_PRUNE_DIRS
                and not d.lower().startswith(_SCAN_PRUNE_PREFIXES)
                and not d.lower().endswith(_SCAN_PRUNE_SUFFIXES)]
            for fn in filenames:
                ext = Path(fn).suffix.lower()
                if ext not in _SCAN_EXTS:
                    continue
                fp = Path(dirpath) / fn
                rp = str(fp.resolve())
                if rp in seen:
                    continue
                seen.add(rp)
                walked += 1
                try:
                    size = fp.stat().st_size
                except OSError:
                    size = None
                # Infer scope from the path: the new-layout jobs/job_00N segment,
                # the survey/ scope, or a legacy job_<NNN>_<name> directory.
                scope = None
                parts = fp.parts
                for i, seg in enumerate(parts):
                    if seg == "jobs" and i + 1 < len(parts):
                        scope = parts[i + 1]
                        break
                    if seg == "survey":
                        scope = "survey"
                    m = _LEGACY_JOB_RE.match(seg)
                    if m:
                        scope = f"job_{int(m.group(1)):03d}"
                items.append(ProductItem(
                    path=str(fp), kind=_kind_for(str(fp), None),
                    scope_id=scope, size=size, source="scan",
                ))
                if walked >= _MAX_SCAN_FILES:
                    return items
    return items


def discover_products(workspace_dir: str | Path) -> list[ProductItem]:
    """All products for a workspace: registry-backed first, filesystem-scanned
    for the rest.  Sorted newest-first where a timestamp is known, then by path.
    """
    workspace = Path(workspace_dir)
    if not workspace.exists():
        return []
    items, seen = _registry_products(workspace)
    items.extend(_scan_products(workspace, seen))
    # Drop metadata/json noise (run.json, project.json) — not user products.
    items = [it for it in items
             if it.kind != "metadata" and Path(it.path).name not in
             ("run.json", "project.json", "registry.json")]
    items.sort(key=lambda it: (it.finished_at or "", it.path), reverse=True)
    return items


def group_products(items: list[ProductItem]) -> dict:
    """Group into {scope_label: {kind: [items]}} for the tree."""
    tree: dict = {}
    for it in items:
        tree.setdefault(it.scope_label, {}).setdefault(it.kind, []).append(it)
    return tree


def read_run_history(workspace_dir: str | Path) -> list[dict]:
    """Every run the registry knows about — all statuses — newest first.

    Unlike ``discover_products`` (completed outputs only), this surfaces the full
    ledger: failures, superseded versions, skips.  It's the "what has this
    workspace actually done?" view.  Returns plain dicts (Qt-free, testable).
    """
    reg_path = Path(workspace_dir) / "runs" / "registry.json"
    if not reg_path.is_file():
        return []
    try:
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    runs = list(reg.get("runs", []))
    runs.sort(key=lambda r: (r.get("finished_at") or "", r.get("run_id") or ""),
              reverse=True)
    return runs


# ===========================================================================
# Qt dialog
# ===========================================================================
def _qt():
    """Import Qt lazily so discover_products stays importable without a display."""
    from PySide6.QtWidgets import (
        QDialog, QTreeWidget, QTreeWidgetItem, QSplitter, QVBoxLayout,
        QHBoxLayout, QWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
        QScrollArea, QFrame, QComboBox, QSizePolicy,
    )
    from PySide6.QtGui import QPixmap, QDesktopServices, QImageReader
    from PySide6.QtCore import Qt, QUrl
    return dict(locals())


_KIND_ICON = {
    "orthomosaic": "🗺️", "dem": "⛰️", "raster": "🌡️", "image": "🖼️",
    "mesh": "🧊", "table": "📊", "dataset": "📦", "report": "📄",
    "project": "🅿️", "other": "•",
}
_KIND_LABEL = {
    "orthomosaic": "Orthomosaics", "dem": "Elevation (DEM)", "raster": "Rasters",
    "image": "Images", "mesh": "Meshes / 3D", "table": "Tables (CSV)",
    "dataset": "Datasets (NetCDF)", "report": "Reports", "project": "Projects",
    "other": "Other",
}


def open_product_browser(parent, workspace_dir: str, viewer_opener=None,
                         external_opener=None):
    """Construct and show the Product Browser (non-modal). Returns the dialog.

    ``viewer_opener(path)`` opens a mesh in the app's 3D viewer;
    ``external_opener(path)`` opens a file with the OS default app.  Both are
    injected so this module never imports main_window (keeps layering clean).
    """
    dlg = ProductBrowserDialog(parent, workspace_dir, viewer_opener, external_opener)
    dlg.show()
    dlg.raise_()
    return dlg


try:
    from PySide6.QtWidgets import QDialog as _QDialogBase
except Exception:  # pragma: no cover - allows headless import of discovery only
    _QDialogBase = object


class ProductBrowserDialog(_QDialogBase):
    """A simple, read-only, single-click viewer for every product in a dive."""

    def __init__(self, parent, workspace_dir: str, viewer_opener=None,
                 external_opener=None):
        from PySide6.QtWidgets import (
            QTreeWidget, QSplitter, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
            QPushButton, QLineEdit, QTextEdit, QScrollArea, QComboBox, QTabWidget,
        )
        from PySide6.QtCore import Qt
        super().__init__(parent)
        self._ws = workspace_dir
        self._viewer_opener = viewer_opener
        self._external_opener = external_opener
        self._current: Optional[ProductItem] = None

        self.setWindowTitle("Product Browser")
        self.resize(1040, 680)
        self.setWindowFlag(Qt.Window, True)     # own taskbar entry, minimisable

        outer = QVBoxLayout(self)
        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, 1)

        products_page = QWidget()
        outer = QVBoxLayout(products_page)      # products tab owns this layout
        self._tabs.addTab(products_page, "Products")

        # --- filter row ------------------------------------------------------
        filt = QHBoxLayout()
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter by name, type, or scope…")
        self._search.textChanged.connect(self._apply_filter)
        self._kind_filter = QComboBox()
        self._kind_filter.addItem("All types", "")
        self._kind_filter.currentIndexChanged.connect(self._apply_filter)
        refresh = QPushButton("↻ Refresh")
        refresh.clicked.connect(self.reload)
        filt.addWidget(QLabel("🔎"))
        filt.addWidget(self._search, 1)
        filt.addWidget(self._kind_filter)
        filt.addWidget(refresh)
        outer.addLayout(filt)

        # --- split: tree | preview ------------------------------------------
        split = QSplitter(Qt.Horizontal)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Product", "When", "Size"])
        self._tree.setColumnWidth(0, 300)
        self._tree.setAlternatingRowColors(True)
        self._tree.currentItemChanged.connect(self._on_select)
        self._tree.itemDoubleClicked.connect(self._on_activate)
        split.addWidget(self._tree)

        # right panel
        right = QWidget()
        rlay = QVBoxLayout(right)
        self._title = QLabel("Select a product")
        self._title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._title.setWordWrap(True)
        rlay.addWidget(self._title)

        self._preview_scroll = QScrollArea()
        self._preview_scroll.setWidgetResizable(True)
        self._preview_label = QLabel("")
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setMinimumHeight(240)
        self._preview_label.setStyleSheet("color: gray;")
        self._preview_scroll.setWidget(self._preview_label)
        rlay.addWidget(self._preview_scroll, 1)

        self._meta = QTextEdit()
        self._meta.setReadOnly(True)
        self._meta.setMaximumHeight(150)
        rlay.addWidget(self._meta)

        # action buttons
        btns = QHBoxLayout()
        self._btn_open = QPushButton("Open externally")
        self._btn_open.clicked.connect(lambda: self._open_current_external())
        self._btn_folder = QPushButton("Reveal in folder")
        self._btn_folder.clicked.connect(self._reveal_current)
        self._btn_3d = QPushButton("Open in 3D viewer")
        self._btn_3d.clicked.connect(self._open_current_3d)
        for b in (self._btn_open, self._btn_folder, self._btn_3d):
            b.setEnabled(False)
            btns.addWidget(b)
        btns.addStretch(1)
        rlay.addLayout(btns)

        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 660])
        outer.addWidget(split, 1)

        self._status = QLabel("")
        self._status.setStyleSheet("color: gray; font-size: 11px;")
        outer.addWidget(self._status)

        # --- Run History tab -------------------------------------------------
        self._build_history_tab()

        self._items: list[ProductItem] = []
        self.reload()

    def _build_history_tab(self) -> None:
        from PySide6.QtWidgets import (
            QWidget, QVBoxLayout, QTableWidget, QAbstractItemView, QHeaderView,
            QLabel,
        )
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.addWidget(QLabel(
            "Every run recorded for this workspace — including failures and "
            "runs superseded by a newer version."))
        self._history = QTableWidget(0, 7)
        self._history.setHorizontalHeaderLabels(
            ["", "Task", "Scope", "Finished", "Duration", "Engine", "Run ID"])
        self._history.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._history.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._history.setAlternatingRowColors(True)
        self._history.verticalHeader().setVisible(False)
        hh = self._history.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        lay.addWidget(self._history, 1)
        self._history_status = QLabel("")
        self._history_status.setStyleSheet("color: gray; font-size: 11px;")
        lay.addWidget(self._history_status)
        self._tabs.addTab(page, "Run History")

    def _rebuild_history(self) -> None:
        from PySide6.QtWidgets import QTableWidgetItem
        from PySide6.QtGui import QColor
        from PySide6.QtCore import Qt
        runs = read_run_history(self._ws)
        badge = {"completed": "✓", "failed": "✗", "partial": "◐", "skipped": "⊘"}
        color = {"completed": QColor("#2e7d32"), "failed": QColor("#c62828"),
                 "partial": QColor("#ef6c00"), "skipped": QColor("#757575")}
        self._history.setRowCount(len(runs))
        n_fail = 0
        for row, r in enumerate(runs):
            status = r.get("status", "?")
            superseded = bool(r.get("superseded_by"))
            if status == "failed":
                n_fail += 1
            b = badge.get(status, "•")
            if superseded and status == "completed":
                b = "⟲"
            dur = r.get("duration_s")
            ch = r.get("channel")
            task = r.get("task_type", "?") + (f" / {ch}" if ch else "")
            cells = [
                b, task, str(r.get("scope_id", "")),
                (r.get("finished_at", "") or "").replace("T", " ")[:19],
                f"{dur:.1f} s" if isinstance(dur, (int, float)) else "",
                r.get("engine", "") or "", r.get("run_id", ""),
            ]
            for col, val in enumerate(cells):
                item = QTableWidgetItem(val)
                if col == 0:
                    item.setForeground(color.get(status, QColor("#000")))
                    item.setTextAlignment(Qt.AlignCenter)
                if superseded and status == "completed":
                    item.setForeground(QColor("#9e9e9e"))
                item.setToolTip(r.get("manifest_path", ""))
                self._history.setItem(row, col, item)
        self._history_status.setText(
            f"{len(runs)} run(s) · {n_fail} failed · registry: "
            f"{Path(self._ws) / 'runs' / 'registry.json'}")

    # ------------------------------------------------------------------ data
    def reload(self) -> None:
        self._items = discover_products(self._ws)
        # populate the kind filter with kinds actually present
        kinds = sorted({it.kind for it in self._items})
        cur = self._kind_filter.currentData()
        self._kind_filter.blockSignals(True)
        self._kind_filter.clear()
        self._kind_filter.addItem("All types", "")
        for k in kinds:
            self._kind_filter.addItem(_KIND_LABEL.get(k, k.title()), k)
        idx = self._kind_filter.findData(cur)
        self._kind_filter.setCurrentIndex(idx if idx >= 0 else 0)
        self._kind_filter.blockSignals(False)
        self._rebuild_tree()
        self._rebuild_history()

    def _rebuild_tree(self) -> None:
        from PySide6.QtWidgets import QTreeWidgetItem
        from PySide6.QtCore import Qt
        text = (self._search.text() or "").strip().lower()
        kind_sel = self._kind_filter.currentData() or ""

        def keep(it: ProductItem) -> bool:
            if kind_sel and it.kind != kind_sel:
                return False
            if not text:
                return True
            hay = " ".join(str(x) for x in
                           (it.name, it.kind, it.scope_label, it.task_type,
                            it.channel, it.engine) if x).lower()
            return text in hay

        shown = [it for it in self._items if keep(it)]
        tree = group_products(shown)
        self._tree.clear()
        n = 0
        for scope in sorted(tree.keys()):
            scope_node = QTreeWidgetItem([scope])
            f = scope_node.font(0); f.setBold(True); scope_node.setFont(0, f)
            self._tree.addTopLevelItem(scope_node)
            scope_node.setExpanded(True)
            for kind in sorted(tree[scope].keys(), key=lambda k: _KIND_LABEL.get(k, k)):
                kind_items = tree[scope][kind]
                kind_node = QTreeWidgetItem(
                    [f"{_KIND_ICON.get(kind,'•')}  {_KIND_LABEL.get(kind, kind.title())} "
                     f"({len(kind_items)})"])
                scope_node.addChild(kind_node)
                kind_node.setExpanded(True)
                for it in kind_items:
                    n += 1
                    when = (it.finished_at or "").replace("T", " ")[:16]
                    label = it.name + ("  ⟲" if it.superseded else "")
                    child = QTreeWidgetItem([label, when, _human_size(it.size)])
                    if it.superseded:
                        child.setForeground(0, Qt.gray)
                    if not it.exists:
                        child.setForeground(0, Qt.red)
                        child.setText(0, it.name + "  (missing)")
                    child.setData(0, Qt.UserRole, it)
                    ch = it.channel or ""
                    tip = f"{it.path}"
                    if it.task_type:
                        tip = f"{it.task_type}{('/'+ch) if ch else ''}\n{it.path}"
                    child.setToolTip(0, tip)
                    kind_node.addChild(child)
        self._status.setText(
            f"{n} product(s) shown · {len(self._items)} total · workspace: {self._ws}")

    def _apply_filter(self, *_a) -> None:
        self._rebuild_tree()

    # --------------------------------------------------------------- selection
    def _on_select(self, cur, _prev) -> None:
        from PySide6.QtCore import Qt
        it = cur.data(0, Qt.UserRole) if cur else None
        self._current = it if isinstance(it, ProductItem) else None
        self._render_current()

    def _on_activate(self, item, _col) -> None:
        from PySide6.QtCore import Qt
        it = item.data(0, Qt.UserRole)
        if isinstance(it, ProductItem):
            self._current = it
            if it.kind == "mesh":
                self._open_current_3d()
            else:
                self._open_current_external()

    def _render_current(self) -> None:
        it = self._current
        have = it is not None and it.exists
        self._btn_open.setEnabled(have)
        self._btn_folder.setEnabled(it is not None)
        self._btn_3d.setEnabled(have and it.kind == "mesh" and self._viewer_opener is not None)
        if it is None:
            self._title.setText("Select a product")
            self._preview_label.setText("")
            self._meta.setPlainText("")
            return
        self._title.setText(f"{_KIND_ICON.get(it.kind,'•')}  {it.name}")
        self._render_preview(it)
        self._render_meta(it)

    def _render_preview(self, it: ProductItem) -> None:
        from PySide6.QtGui import QPixmap, QImageReader
        from PySide6.QtCore import Qt
        self._preview_label.setPixmap(QPixmap())      # clear
        if not it.exists:
            self._preview_label.setText("File is missing from disk.")
            return
        if it.kind in ("image", "raster", "orthomosaic", "dem"):
            reader = QImageReader(it.path)
            reader.setAutoTransform(True)
            img = reader.read()
            if not img.isNull():
                pm = QPixmap.fromImage(img)
                w = max(320, self._preview_scroll.viewport().width() - 24)
                self._preview_label.setPixmap(
                    pm.scaled(w, 640, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                return
            self._preview_label.setText(
                "🗺️  Georeferenced raster.\nNo inline preview (open externally, "
                "or drape it in the 3D viewer).")
        elif it.kind == "table":
            self._preview_label.setText(self._csv_head(it.path))
            self._preview_label.setStyleSheet(
                "color: #333; font-family: monospace; font-size: 11px;")
            self._preview_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            return
        elif it.kind == "mesh":
            self._preview_label.setText("🧊  3D mesh.\nOpen in the 3D viewer to inspect.")
        elif it.kind == "report":
            self._preview_label.setText("📄  Report.\nOpen externally to read.")
        else:
            self._preview_label.setText(f"{_KIND_ICON.get(it.kind,'•')}  {it.name}")
        self._preview_label.setStyleSheet("color: gray;")
        self._preview_label.setAlignment(Qt.AlignCenter)

    @staticmethod
    def _csv_head(path: str, rows: int = 40) -> str:
        try:
            lines = []
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i >= rows:
                        lines.append("…")
                        break
                    lines.append(line.rstrip("\n"))
            return "\n".join(lines)
        except OSError as exc:
            return f"(could not read: {exc})"

    def _render_meta(self, it: ProductItem) -> None:
        rows = [
            ("Type", _KIND_LABEL.get(it.kind, it.kind)),
            ("Scope", it.scope_label),
            ("Task", it.task_type or "—"),
            ("Channel", it.channel or "—"),
            ("Engine", it.engine or "—"),
            ("Finished", (it.finished_at or "—").replace("T", " ")),
            ("Duration", f"{it.duration_s:.1f} s" if it.duration_s else "—"),
            ("Size", _human_size(it.size) or "—"),
            ("Run", it.run_id or "—"),
            ("Source", "provenance registry" if it.source == "registry"
             else "filesystem scan (no run record)"),
            ("Path", it.path),
        ]
        if it.superseded:
            rows.insert(0, ("⚠ Status", "superseded by a newer run"))
        html = "<table cellspacing='4'>" + "".join(
            f"<tr><td style='color:#666'><b>{k}</b></td><td>{v}</td></tr>"
            for k, v in rows) + "</table>"
        self._meta.setHtml(html)

    # ----------------------------------------------------------------- actions
    def _open_current_external(self) -> None:
        if not (self._current and self._current.exists):
            return
        if self._external_opener:
            self._external_opener(self._current.path)
        else:
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._current.path))

    def _reveal_current(self) -> None:
        if not self._current:
            return
        folder = str(Path(self._current.path).parent)
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _open_current_3d(self) -> None:
        if self._current and self._current.exists and self._viewer_opener:
            self._viewer_opener(self._current.path)
