"""workspace_panel.py — Persistent workspace file-structure viewer.

Shows every versioned output product under the workspace directory, annotated
with the settings used to generate each run (read from run.meta.json or
<file>.meta.json sidecars).  Docked to the bottom of the main window.

Directory layout expected by this panel:

    outputs/
      nav_depth/
        run_001/  nav_depth.tif  + run.meta.json
      nav_trackline/
        run_001/
          nav_trackline.ply + run.meta.json + grid.csv.gz
          slices/
            run_001/  *.png + run.meta.json
      sensor_2d/{channel}/run_001/  {channel}_2d.tif
      sensor_3d/{channel}/
        run_001/
          {channel}_3d.ply + run.meta.json + grid.csv.gz
          slices/
            run_001/  *.png + run.meta.json

Clicking a run_NNN node under nav_trackline or sensor_3d emits run_selected(path)
so the main window can store it as the target for "Generate Depth Slices".

Backward-compatible with the old flat structure:
    outputs/nav/nav_trackline.ply  etc.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QFileSystemWatcher, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)


def _fmt_size(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    if n_bytes < 1024 ** 3:
        return f"{n_bytes / 1024 ** 2:.1f} MB"
    return f"{n_bytes / 1024 ** 3:.2f} GB"


def _fmt_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")


def _read_meta(path: str) -> Optional[dict]:
    """Read meta from {path}.meta.json (file product) or {path}/run.meta.json (dir product)."""
    p = Path(path)
    candidates = [
        p / "run.meta.json" if p.is_dir() else Path(str(path) + ".meta.json"),
        Path(str(path) + ".meta.json"),
    ]
    for c in candidates:
        try:
            with open(c) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _settings_summary(meta: dict) -> str:
    """One-line summary of the settings in a meta dict."""
    if not meta:
        return ""
    parts = []
    s = meta.get("settings", {})
    if "cell_size_m" in s:
        parts.append(f"{s['cell_size_m']} m")
    if "fill_method" in s:
        parts.append(s["fill_method"])
    if "crs" in s:
        parts.append(s["crs"].upper())
    if "aggregation" in s:
        parts.append(s["aggregation"])
    if "altitude_step_m" in s:
        parts.append(f"step={s['altitude_step_m']} m")
    if "color_mode" in s:
        parts.append(s["color_mode"])
    return "  ·  ".join(parts)


# Colour for items that have metadata (tracked runs)
_META_COLOR  = QColor("#4fc3f7")
# Colour for directory/group headers
_GROUP_COLOR = QColor("#aaaaaa")
# Colour for legacy/untracked files
_LEGACY_COLOR = QColor("#f0ad4e")
# Colour for selected-target runs
_TARGET_COLOR = QColor("#a5d6a7")

# UserRole assignments
_ROLE_PATH      = Qt.UserRole       # absolute path string
_ROLE_META      = Qt.UserRole + 1   # meta dict (if any)
_ROLE_NODE_TYPE = Qt.UserRole + 2   # "3d_run" | "slice_run" | "photo_run" | None


class WorkspacePanel(QWidget):
    """Read-only workspace file-structure viewer with per-run metadata.

    Signals:
        run_selected(str) — emitted when the user clicks a 3D model run node;
                            carries the absolute path of that run directory.
    """

    run_selected               = Signal(str)  # user clicked a 3D model run node
    make_slices_requested      = Signal(str)  # user chose "Make Slices" from context menu
    open_in_viewer_requested   = Signal(str)  # "Open in 3D Viewer" — path to a PLY/OBJ
    open_in_metashape_requested = Signal(str)  # "Open in Metashape GUI" — path to a .psx

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root: Optional[str] = None
        self._interp_mtime: float | None = None  # mtime of current interp_full.csv
        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._schedule_refresh)

        # Configured source files — set by MainWindow via set_sources()
        self._src_video_dir: str = ""
        self._src_videos: list = []
        self._src_nav_file = None        # NavigationConfig | None
        self._src_sensor_files: list = []  # list[SensorFileConfig]
        self._src_depth_source = None    # SensorFileConfig | None
        self._src_speed_source = None    # SensorFileConfig | None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Workspace Files")
        title.setStyleSheet("font-weight: bold;")
        bar.addWidget(title)
        bar.addStretch()
        refresh_btn = QPushButton("⟳ Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.setToolTip("Re-scan the workspace directory")
        refresh_btn.clicked.connect(self.refresh)
        bar.addWidget(refresh_btn)
        open_btn = QPushButton("Open Folder")
        open_btn.setFixedWidth(90)
        open_btn.setToolTip("Open the workspace directory in the system file manager")
        open_btn.clicked.connect(self._open_folder)
        bar.addWidget(open_btn)
        layout.addLayout(bar)

        # --- Project status strip ---
        self._status_strip = QLabel()
        self._status_strip.setWordWrap(True)
        self._status_strip.setStyleSheet(
            "font-size: 10px; padding: 3px 5px; "
            "background: #1e1e2e; border: 1px solid #333; border-radius: 3px;"
        )
        self._status_strip.setText("(no workspace)")
        layout.addWidget(self._status_strip)

        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter, stretch=1)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(["Name / Run", "Settings", "Size", "Generated"])
        self._tree.header().setStretchLastSection(False)
        self._tree.setColumnWidth(0, 180)
        self._tree.setColumnWidth(1, 220)
        self._tree.setColumnWidth(2, 72)
        self._tree.setColumnWidth(3, 130)
        self._tree.currentItemChanged.connect(self._on_selection_changed)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        splitter.addWidget(self._tree)

        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setPlaceholderText("Select a run to see its full settings.")
        splitter.addWidget(self._detail)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_root(self, path: Optional[str]) -> None:
        """Set the workspace root and refresh the tree."""
        if self._watcher.directories():
            self._watcher.removePaths(self._watcher.directories())
        self._root = path
        if path and os.path.isdir(path):
            self._watcher.addPath(path)
            inputs = str(Path(path) / "inputs")
            if os.path.isdir(inputs):
                self._watcher.addPath(inputs)
                for dirpath, dirnames, _ in os.walk(inputs):
                    for d in dirnames:
                        self._watcher.addPath(os.path.join(dirpath, d))
            outputs = str(Path(path) / "outputs")
            if os.path.isdir(outputs):
                self._watcher.addPath(outputs)
                for dirpath, dirnames, _ in os.walk(outputs):
                    # Don't watch the grid cache — it changes often and is internal
                    dirnames[:] = [d for d in dirnames if d != "_grid_cache"]
                    for d in dirnames:
                        self._watcher.addPath(os.path.join(dirpath, d))
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the tree from the current workspace root."""
        expanded = self._save_expanded()
        self._tree.clear()
        self._detail.clear()

        # Cache interp_full.csv mtime for staleness checks during tree population
        self._interp_mtime = None
        if self._root:
            interp = Path(self._root) / "interp_full.csv"
            if interp.exists():
                try:
                    self._interp_mtime = interp.stat().st_mtime
                except OSError:
                    pass

        if not self._root or not os.path.isdir(self._root):
            placeholder = QTreeWidgetItem(["(no workspace loaded)", "", "", ""])
            self._tree.addTopLevelItem(placeholder)
            self._update_status_strip()
            return

        self._populate_tree(expanded)
        self._update_status_strip()

    def get_context_summary(self) -> str:
        """Return a plain-text workspace inventory for use in Claude's system prompt."""
        if not self._root or not os.path.isdir(self._root):
            return "No workspace is loaded."

        root = Path(self._root)
        lines: list[str] = []

        interp = root / "interp_full.csv"
        if interp.exists():
            stat = interp.stat()
            lines.append(f"interp_full.csv: present ({_fmt_size(stat.st_size)}, {_fmt_mtime(stat.st_mtime)})")
        else:
            lines.append("interp_full.csv: not yet built")

        outputs = root / "outputs"
        if not outputs.is_dir():
            lines.append("No outputs directory found.")
            return "\n".join(lines)

        # Nav 3D PLY runs
        nav_tl = outputs / "nav_trackline"
        if nav_tl.is_dir():
            runs = sorted(d for d in nav_tl.iterdir() if d.is_dir() and d.name.startswith("run_"))
            if runs:
                lines.append(f"Nav 3D PLY runs ({len(runs)}):")
                for r in runs:
                    m = _read_meta(str(r))
                    s = m.get("settings", {}) if m else {}
                    gen = m.get("generated", "") if m else ""
                    lines.append(f"  {r.name}: cell={s.get('cell_size_m','?')}m  generated={gen}")
                    slices_dir = r / "slices"
                    if slices_dir.is_dir():
                        n = sum(1 for d in slices_dir.iterdir() if d.is_dir() and d.name.startswith("run_"))
                        if n:
                            lines.append(f"    {n} slice set(s)")

        # Sensor 3D PLY runs
        sensor_3d = outputs / "sensor_3d"
        if sensor_3d.is_dir():
            for ch_dir in sorted(ch for ch in sensor_3d.iterdir() if ch.is_dir()):
                runs = sorted(d for d in ch_dir.iterdir() if d.is_dir() and d.name.startswith("run_"))
                if runs:
                    lines.append(f"Sensor 3D PLY — channel '{ch_dir.name}' ({len(runs)} run(s)):")
                    for r in runs:
                        m = _read_meta(str(r))
                        s = m.get("settings", {}) if m else {}
                        gen = m.get("generated", "") if m else ""
                        lines.append(
                            f"  {r.name}: cell={s.get('cell_size_m','?')}m  "
                            f"agg={s.get('aggregation','?')}  fill={s.get('fill_method','?')}  generated={gen}"
                        )
                        slices_dir = r / "slices"
                        if slices_dir.is_dir():
                            n = sum(1 for d in slices_dir.iterdir() if d.is_dir() and d.name.startswith("run_"))
                            if n:
                                lines.append(f"    {n} slice set(s)")

        # Nav/Sensor 2D GeoTIFF runs (brief)
        for product_dir, label in [
            (outputs / "nav_depth",  "Nav Depth 2D GeoTIFF"),
            (outputs / "sensor_2d", "Sensor 2D GeoTIFF"),
        ]:
            if product_dir.is_dir():
                count = sum(
                    1 for d in product_dir.rglob("run_*") if d.is_dir()
                )
                if count:
                    lines.append(f"{label}: {count} run(s)")

        return "\n".join(lines) if lines else "Workspace loaded but no outputs found."

    def set_sources(
        self,
        video_dir: str,
        videos: list,
        nav_file,
        sensor_files: list,
        depth_source=None,
        speed_source=None,
    ) -> None:
        """Update configured source file references and rebuild the tree.

        Called by MainWindow whenever videos, nav, or sensor configuration changes.
        Does not modify the workspace directory — only affects the Source Files
        section at the top of the tree.
        """
        self._src_video_dir    = video_dir or ""
        self._src_videos       = list(videos)
        self._src_nav_file     = nav_file
        self._src_sensor_files = list(sensor_files)
        self._src_depth_source = depth_source
        self._src_speed_source = speed_source
        self.refresh()

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------

    def _populate_tree(self, expanded: set) -> None:
        root = Path(self._root)

        # --- Source Files section (configured inputs, not on-disk workspace files) ---
        self._add_source_subtree(expanded)

        interp = root / "interp_full.csv"
        if interp.exists():
            stat = interp.stat()
            item = QTreeWidgetItem(self._tree.invisibleRootItem(), [
                "interp_full.csv", "", _fmt_size(stat.st_size), _fmt_mtime(stat.st_mtime)
            ])
            item.setData(0, _ROLE_PATH, str(interp))

        # --- Inputs section ---
        inputs = root / "inputs"
        if inputs.is_dir():
            in_node = self._group_node(
                self._tree.invisibleRootItem(), "inputs/", str(inputs), expanded
            )
            for subdir, label in [("nav", "Navigation"), ("sensor", "Sensor")]:
                d = inputs / subdir
                if d.is_dir():
                    files = sorted(f for f in d.iterdir() if f.is_file())
                    if files:
                        sub_node = self._group_node(in_node, label, str(d), expanded)
                        for f in files:
                            stat = f.stat()
                            child = QTreeWidgetItem(sub_node, [
                                f.name, "", _fmt_size(stat.st_size), _fmt_mtime(stat.st_mtime)
                            ])
                            child.setData(0, _ROLE_PATH, str(f))

        # --- Outputs section ---
        outputs = root / "outputs"
        if not outputs.is_dir():
            return

        out_node = self._group_node(self._tree.invisibleRootItem(), "outputs/", str(outputs), expanded)
        self._add_outputs_subtree(out_node, outputs, expanded)

    def _add_source_subtree(self, expanded: set) -> None:
        """Add a collapsible 'Source Files' section for configured inputs."""
        has_videos  = bool(self._src_video_dir or self._src_videos)
        has_nav     = self._src_nav_file is not None
        has_sensors = bool(self._src_sensor_files or self._src_depth_source or self._src_speed_source)

        if not (has_videos or has_nav or has_sensors):
            return

        root_item = self._tree.invisibleRootItem()
        src_node  = self._group_node(root_item, "Source Files", "__sources__", expanded)

        # --- Videos ---
        if has_videos:
            dir_label = self._src_video_dir or "(directory not set)"
            vid_node  = self._group_node(src_node, f"Videos  —  {dir_label}", "__vid__", expanded)
            if self._src_videos:
                for v in self._src_videos:
                    full_path = Path(self._src_video_dir) / v.filename if self._src_video_dir else None
                    size_str  = ""
                    if full_path and full_path.exists():
                        size_str = _fmt_size(full_path.stat().st_size)
                    time_str = (
                        f"{v.start_time.strftime('%H:%M:%S')} → {v.end_time.strftime('%H:%M:%S')}"
                        if v.start_time and v.end_time else ""
                    )
                    child = QTreeWidgetItem(vid_node, [v.filename, time_str, size_str, ""])
                    if full_path:
                        child.setData(0, _ROLE_PATH, str(full_path))
            else:
                ph = QTreeWidgetItem(vid_node, ["(no videos scanned)", "", "", ""])
                ph.setForeground(0, _GROUP_COLOR)

        # --- Navigation ---
        if has_nav:
            nav  = self._src_nav_file
            srcs = nav._all_sources if hasattr(nav, "_all_sources") else []
            seen: set[str] = set()
            nav_node = self._group_node(src_node, "Navigation", "__nav__", expanded)
            for src in srcs:
                p = Path(str(src.csv_path)) if src.csv_path else None
                if p is None or str(p) in seen:
                    continue
                seen.add(str(p))
                size_str = _fmt_size(p.stat().st_size) if p.exists() else "(not found)"
                child = QTreeWidgetItem(nav_node, [p.name, "", size_str, ""])
                child.setData(0, _ROLE_PATH, str(p))
                child.setForeground(0, _META_COLOR)
            if not seen:
                ph = QTreeWidgetItem(nav_node, ["(no source files found)", "", "", ""])
                ph.setForeground(0, _GROUP_COLOR)

        # --- Sensors ---
        if has_sensors:
            sensor_node = self._group_node(src_node, "Sensors", "__sensors__", expanded)
            all_sensors = list(self._src_sensor_files)
            if self._src_depth_source:
                all_sensors.append(self._src_depth_source)
            if self._src_speed_source:
                all_sensors.append(self._src_speed_source)
            seen_sensors: set[str] = set()
            for sf in all_sensors:
                p = Path(str(sf.csv_path)) if sf.csv_path else None
                if p is None or str(p) in seen_sensors:
                    continue
                seen_sensors.add(str(p))
                ch_names = ", ".join(
                    ch.display_name for ch in sf.channels
                ) if sf.channels else ""
                size_str = _fmt_size(p.stat().st_size) if p.exists() else "(not found)"
                child = QTreeWidgetItem(sensor_node, [p.name, ch_names, size_str, ""])
                child.setData(0, _ROLE_PATH, str(p))
                child.setForeground(0, _META_COLOR)
            if not seen_sensors:
                ph = QTreeWidgetItem(sensor_node, ["(no sensor files configured)", "", "", ""])
                ph.setForeground(0, _GROUP_COLOR)

    def _add_outputs_subtree(self, parent: QTreeWidgetItem, outputs: Path, expanded: set) -> None:
        """Add all output product groups to the tree, versioned-first then legacy."""

        # --- Nav 2D GeoTIFF (non-3D, no slices child) ---
        nav_depth = outputs / "nav_depth"
        if nav_depth.is_dir():
            node = self._group_node(parent, "Nav Depth (2D GeoTIFF)", str(nav_depth), expanded)
            self._add_versioned_runs(node, nav_depth, expanded, node_type=None)

        # --- Nav 3D trackline (3D runs + slices children) ---
        nav_tl = outputs / "nav_trackline"
        if nav_tl.is_dir():
            node = self._group_node(parent, "Nav Trackline (3D PLY)", str(nav_tl), expanded)
            self._add_versioned_runs(node, nav_tl, expanded, node_type="3d_run")

        # --- Sensor 2D: one sub-group per channel (no slices) ---
        sensor_2d = outputs / "sensor_2d"
        if sensor_2d.is_dir():
            channels = sorted(ch for ch in sensor_2d.iterdir() if ch.is_dir())
            if channels:
                node = self._group_node(parent, "Sensor 2D (GeoTIFF)", str(sensor_2d), expanded)
                for ch_dir in channels:
                    ch_node = self._group_node(node, ch_dir.name, str(ch_dir), expanded)
                    self._add_versioned_runs(ch_node, ch_dir, expanded, node_type=None)

        # --- Sensor 3D: one sub-group per channel (3D runs + slices children) ---
        sensor_3d = outputs / "sensor_3d"
        if sensor_3d.is_dir():
            channels = sorted(ch for ch in sensor_3d.iterdir() if ch.is_dir())
            if channels:
                node = self._group_node(parent, "Sensor 3D (PLY)", str(sensor_3d), expanded)
                for ch_dir in channels:
                    ch_node = self._group_node(node, ch_dir.name, str(ch_dir), expanded)
                    self._add_versioned_runs(ch_node, ch_dir, expanded, node_type="3d_run")

        # --- Photogrammetry: job_NNN/run_NNN with dense cloud / mesh / report ---
        photogrammetry = outputs / "photogrammetry"
        if photogrammetry.is_dir():
            self._add_photogrammetry_subtree(parent, photogrammetry, expanded)

        # --- Legacy flat layout (backward compat) ---
        self._add_legacy_products(parent, outputs, expanded)

    def _add_photogrammetry_subtree(
        self, parent: QTreeWidgetItem, photogrammetry: Path, expanded: set
    ) -> None:
        """Add Photogrammetry → job_NNN → run_NNN nodes.

        Each run directory holds Metashape/COLMAP products (dense_cloud.ply,
        mesh.obj, report.pdf, project.psx, …).  Run nodes are tagged
        node_type="photo_run" so the context menu can offer 'Open in 3D Viewer',
        'Open in Metashape GUI', and 'Open report'.
        """
        jobs = sorted(
            (d for d in photogrammetry.iterdir() if d.is_dir() and d.name.startswith("job_")),
            key=lambda d: d.name,
        )
        if not jobs:
            return
        group = self._group_node(parent, "Photogrammetry", str(photogrammetry), expanded)
        for job_dir in jobs:
            runs = sorted(
                (d for d in job_dir.iterdir() if d.is_dir() and d.name.startswith("run_")),
                key=lambda d: d.name,
            )
            if not runs:
                continue
            job_node = self._group_node(group, job_dir.name, str(job_dir), expanded)
            for run_dir in runs:
                meta = _read_meta(str(run_dir))
                summary = _settings_summary(meta) if meta else self._photo_run_summary(run_dir)
                generated = meta.get("generated", "") if meta else ""
                size_str = self._run_size(run_dir)
                item = QTreeWidgetItem(job_node, [run_dir.name, summary, size_str, generated])
                item.setData(0, _ROLE_PATH, str(run_dir))
                item.setData(0, _ROLE_NODE_TYPE, "photo_run")
                if meta:
                    item.setData(0, _ROLE_META, meta)
                    item.setForeground(0, _META_COLOR)
                else:
                    item.setForeground(0, _LEGACY_COLOR)
                self._add_run_files(item, run_dir)

    @staticmethod
    def _photo_run_summary(run_dir: Path) -> str:
        """Short description of what products a photogrammetry run produced."""
        bits = []
        if (run_dir / "dense_cloud.ply").exists():
            bits.append("dense cloud")
        elif (run_dir / "sparse_cloud.ply").exists():
            bits.append("sparse cloud")
        if (run_dir / "mesh_textured.obj").exists():
            bits.append("textured mesh")
        elif (run_dir / "mesh.obj").exists():
            bits.append("mesh")
        return ", ".join(bits)

    def _add_versioned_runs(
        self,
        parent: QTreeWidgetItem,
        product_dir: Path,
        expanded: set,
        node_type: Optional[str] = None,
    ) -> None:
        """Add run_NNN items under a product directory node.

        node_type: "3d_run" for nav/sensor 3D PLY runs (supports slices children),
                   None for 2D GeoTIFF runs.
        """
        runs = sorted(
            (d for d in product_dir.iterdir() if d.is_dir() and d.name.startswith("run_")),
            key=lambda d: d.name,
        )
        if not runs:
            placeholder = QTreeWidgetItem(parent, ["(no runs yet)", "", "", ""])
            placeholder.setForeground(0, _GROUP_COLOR)
            return

        for run_dir in runs:
            meta = _read_meta(str(run_dir))
            summary = _settings_summary(meta) if meta else ""
            generated = meta.get("generated", "") if meta else ""
            size_str = self._run_size(run_dir)

            # Staleness check: did interp_full.csv change after this run was built?
            is_stale = False
            if (
                meta
                and self._interp_mtime is not None
                and node_type == "3d_run"
            ):
                src_mtime = meta.get("interp_source", {}).get("mtime")
                if src_mtime is not None and self._interp_mtime > src_mtime + 1:
                    is_stale = True

            display_name = ("⚠ " + run_dir.name) if is_stale else run_dir.name
            item = QTreeWidgetItem(parent, [display_name, summary, size_str, generated])
            item.setData(0, _ROLE_PATH, str(run_dir))
            item.setData(0, _ROLE_NODE_TYPE, node_type)
            if is_stale:
                item.setToolTip(0, "interp_full.csv has been updated since this run was built. "
                                   "Consider regenerating.")
                item.setForeground(0, _LEGACY_COLOR)
            elif meta:
                item.setData(0, _ROLE_META, meta)
                item.setForeground(0, _META_COLOR)
            else:
                item.setForeground(0, _LEGACY_COLOR)
            if meta:
                item.setData(0, _ROLE_META, meta)

            file_items = self._add_run_files(item, run_dir)

            # For 3D runs, show slices/ as a child group
            if node_type == "3d_run":
                slices_dir = run_dir / "slices"
                if slices_dir.is_dir():
                    slice_runs = sorted(
                        d for d in slices_dir.iterdir()
                        if d.is_dir() and d.name.startswith("run_")
                    )
                    slices_node = self._group_node(item, "slices/", str(slices_dir), expanded)
                    if slice_runs:
                        for sr in slice_runs:
                            sr_meta = _read_meta(str(sr))
                            sr_summary = _settings_summary(sr_meta) if sr_meta else ""
                            sr_generated = sr_meta.get("generated", "") if sr_meta else ""
                            pngs = list(sr.glob("*.png"))
                            sr_size = _fmt_size(sum(p.stat().st_size for p in pngs)) if pngs else ""
                            sr_item = QTreeWidgetItem(slices_node, [
                                sr.name,
                                sr_summary or f"{len(pngs)} PNGs",
                                sr_size,
                                sr_generated,
                            ])
                            sr_item.setData(0, _ROLE_PATH, str(sr))
                            sr_item.setData(0, _ROLE_NODE_TYPE, "slice_run")
                            if sr_meta:
                                sr_item.setData(0, _ROLE_META, sr_meta)
                                sr_item.setForeground(0, _META_COLOR)
                    else:
                        ph = QTreeWidgetItem(slices_node, ["(no slice sets yet)", "", "", ""])
                        ph.setForeground(0, _GROUP_COLOR)
                    file_items += 1  # count slices group

            if file_items <= 3:
                item.setExpanded(True)

    @staticmethod
    def _run_size(run_dir: Path) -> str:
        """Return a size string for the primary product in a run directory."""
        total = 0
        for p in run_dir.iterdir():
            if p.is_file() and not p.name.endswith(".meta.json") and not p.name.endswith(".csv.gz"):
                total += p.stat().st_size
        return _fmt_size(total) if total else ""

    def _add_run_files(self, parent: QTreeWidgetItem, run_dir: Path) -> int:
        """Add child items for each product file inside a run directory.

        Returns the number of items added.
        """
        count = 0
        skipped_suffixes = {".meta.json", ".gz"}
        for p in sorted(run_dir.iterdir()):
            if not p.is_file():
                continue
            if any(p.name.endswith(s) for s in skipped_suffixes):
                continue
            stat = p.stat()
            child = QTreeWidgetItem(parent, [
                p.name, "", _fmt_size(stat.st_size), _fmt_mtime(stat.st_mtime)
            ])
            child.setData(0, _ROLE_PATH, str(p))
            count += 1

        # PNG slice counts get collapsed to a summary
        pngs = list(run_dir.glob("*.png"))
        if pngs and count == 0:
            child = QTreeWidgetItem(parent, [f"{len(pngs)} PNG slices", "", "", ""])
            child.setForeground(0, _GROUP_COLOR)
            count = 1

        return count

    def _add_legacy_products(
        self, parent: QTreeWidgetItem, outputs: Path, expanded: set
    ) -> None:
        """Add items for the old flat output structure (outputs/nav/, outputs/sensor_*/…)."""
        legacy_files = [
            outputs / "nav" / "nav_trackline.ply",
            outputs / "nav" / "nav_depth.tif",
        ]
        legacy_slice_dirs = [outputs / "nav" / "slices"]

        for ch_dir in sorted((outputs / "sensor_2d").iterdir()) if (outputs / "sensor_2d").is_dir() else []:
            if ch_dir.is_dir():
                legacy_files.append(ch_dir / f"{ch_dir.name}_2d.tif")
        for ch_dir in sorted((outputs / "sensor_3d").iterdir()) if (outputs / "sensor_3d").is_dir() else []:
            if ch_dir.is_dir():
                legacy_files.append(ch_dir / f"{ch_dir.name}_3d.ply")
        for ch_dir in sorted((outputs / "sensor_slices").iterdir()) if (outputs / "sensor_slices").is_dir() else []:
            if ch_dir.is_dir():
                legacy_slice_dirs.append(ch_dir)

        legacy_items = []
        for p in legacy_files:
            if p.exists():
                legacy_items.append(p)
        for d in legacy_slice_dirs:
            if d.is_dir() and list(d.glob("*.png")):
                legacy_items.append(d)

        if not legacy_items:
            return

        legacy_node = self._group_node(parent, "Legacy outputs (pre-versioning)", str(outputs), expanded)
        legacy_node.setForeground(0, _LEGACY_COLOR)
        for p in legacy_items:
            if p.is_file():
                stat = p.stat()
                meta = _read_meta(str(p))
                summary = _settings_summary(meta) if meta else "(no metadata)"
                generated = meta.get("generated", "") if meta else ""
                item = QTreeWidgetItem(legacy_node, [
                    p.name, summary, _fmt_size(stat.st_size), generated or _fmt_mtime(stat.st_mtime)
                ])
                item.setData(0, _ROLE_PATH, str(p))
                if meta:
                    item.setData(0, _ROLE_META, meta)
                    item.setForeground(0, _LEGACY_COLOR)
            else:
                pngs = list(p.glob("*.png"))
                meta = _read_meta(str(p))
                summary = _settings_summary(meta) if meta else "(no metadata)"
                generated = meta.get("generated", "") if meta else ""
                item = QTreeWidgetItem(legacy_node, [
                    f"{p.parent.name}/{p.name}/  ({len(pngs)} PNGs)",
                    summary, "", generated,
                ])
                item.setData(0, _ROLE_PATH, str(p))
                if meta:
                    item.setData(0, _ROLE_META, meta)
                item.setForeground(0, _LEGACY_COLOR)

    # ------------------------------------------------------------------
    # Status strip
    # ------------------------------------------------------------------

    def _update_status_strip(self) -> None:
        """Recompute and display the project-level status summary."""
        if not self._root or not os.path.isdir(self._root):
            self._status_strip.setText("<span style='color:#888;'>No workspace loaded.</span>")
            return

        root = Path(self._root)
        parts: list[str] = []

        # interp_full.csv status
        interp = root / "interp_full.csv"
        if interp.exists():
            stat = interp.stat()
            parts.append(
                f"<span style='color:#4caf50;'>&#x2714;</span> "
                f"<b>interp_full.csv</b> {_fmt_size(stat.st_size)}"
            )
            # Channel count (fast header-only read)
            try:
                import pandas as pd
                import numpy as np
                _NAV = frozenset({
                    "unix_time","lat","lon","easting","northing","depth",
                    "water_depth","alt","heading","pitch","roll","utm_zone","frame_filename",
                })
                df = pd.read_csv(str(interp), nrows=1)
                n_ch = len([c for c in df.select_dtypes(include=[np.number]).columns if c not in _NAV])
                parts.append(f"<b>{n_ch}</b> channel{'s' if n_ch != 1 else ''}")
            except Exception:
                pass
        else:
            parts.append(
                "<span style='color:#f44336;'>&#x26A0;</span> "
                "<b>interp_full.csv</b> not found"
            )

        # Run count across all product types
        outputs = root / "outputs"
        n_runs = 0
        if outputs.is_dir():
            for dirpath, dirnames, _ in os.walk(str(outputs)):
                dirnames[:] = [d for d in dirnames if d not in ("_grid_cache", "slices")]
                n_runs += sum(1 for d in dirnames if d.startswith("run_"))
        if n_runs:
            parts.append(f"<b>{n_runs}</b> run{'s' if n_runs != 1 else ''}")

        self._status_strip.setText("  &nbsp;&#x25CF;&nbsp;  ".join(parts))

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _on_tree_context_menu(self, pos) -> None:
        """Show a context menu for 3D run nodes offering 'Make Slices'."""
        from PySide6.QtWidgets import QMenu
        item = self._tree.itemAt(pos)
        if item is None:
            return
        node_type = item.data(0, _ROLE_NODE_TYPE)
        path = item.data(0, _ROLE_PATH)
        if not path or node_type not in ("3d_run", "photo_run"):
            return

        if node_type == "3d_run":
            run_name = Path(path).name
            parent_name = Path(path).parent.name
            menu = QMenu(self)
            slices_action = menu.addAction(f"Make Slices from {parent_name}/{run_name}")
            viewer_action = menu.addAction("Open in 3D Viewer")
            chosen = menu.exec(self._tree.viewport().mapToGlobal(pos))
            if chosen is slices_action:
                self.make_slices_requested.emit(path)
            elif chosen is viewer_action:
                # Find the primary PLY inside this run dir and emit it
                plys = sorted(Path(path).glob("*.ply"))
                if plys:
                    self.open_in_viewer_requested.emit(str(plys[0]))
            return

        # node_type == "photo_run": offer viewer / mesh / Metashape GUI / report
        self._photo_run_context_menu(Path(path), pos)

    def _photo_run_context_menu(self, run_dir: Path, pos) -> None:
        """Context menu for a photogrammetry run: pick the best product to open."""
        from PySide6.QtWidgets import QMenu

        # Prefer the dense cloud for the 3D viewer; fall back to sparse.
        cloud = None
        for name in ("dense_cloud.ply", "sparse_cloud.ply"):
            if (run_dir / name).exists():
                cloud = run_dir / name
                break
        # Mesh: textured preferred over plain.
        mesh = None
        for name in ("mesh_textured.obj", "mesh.obj"):
            if (run_dir / name).exists():
                mesh = run_dir / name
                break
        psx    = run_dir / "project.psx"
        report = run_dir / "report.pdf"

        menu = QMenu(self)
        cloud_action  = menu.addAction("Open point cloud in 3D Viewer") if cloud else None
        mesh_action   = menu.addAction("Open mesh in 3D Viewer") if mesh else None
        psx_action    = menu.addAction("Open in Metashape GUI") if psx.exists() else None
        report_action = menu.addAction("Open report (PDF)") if report.exists() else None
        if menu.isEmpty():
            menu.addAction("(no openable products in this run)").setEnabled(False)

        chosen = menu.exec(self._tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is cloud_action and cloud:
            self.open_in_viewer_requested.emit(str(cloud))
        elif chosen is mesh_action and mesh:
            self.open_in_viewer_requested.emit(str(mesh))
        elif chosen is psx_action:
            self.open_in_metashape_requested.emit(str(psx))
        elif chosen is report_action:
            self._open_path_with_os(str(report))

    @staticmethod
    def _open_path_with_os(path: str) -> None:
        """Open a file with the OS default application."""
        if sys.platform == "win32":
            os.startfile(path)  # noqa: SLF001 — Windows-only API
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _group_node(
        self,
        parent: QTreeWidgetItem,
        label: str,
        path: str,
        expanded: set,
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [label, "", "", ""])
        item.setForeground(0, _GROUP_COLOR)
        item.setData(0, _ROLE_PATH, path)
        item.setExpanded(path in expanded or not expanded)
        return item

    def _save_expanded(self) -> set:
        """Return the set of path-keyed expanded node identifiers."""
        expanded: set = set()
        it = QTreeWidgetItemIterator(self._tree)
        while it.value():
            item = it.value()
            if item.isExpanded():
                key = item.data(0, _ROLE_PATH)
                if key:
                    expanded.add(key)
            it += 1
        return expanded

    # ------------------------------------------------------------------
    # Selection detail
    # ------------------------------------------------------------------

    def _on_selection_changed(self, current: Optional[QTreeWidgetItem], _prev) -> None:
        if current is None:
            self._detail.clear()
            return

        path      = current.data(0, _ROLE_PATH)
        meta      = current.data(0, _ROLE_META)
        node_type = current.data(0, _ROLE_NODE_TYPE)

        # Emit run_selected when the user clicks a 3D model run node
        if node_type == "3d_run" and path:
            self.run_selected.emit(path)

        lines: list[str] = []
        if path:
            lines.append(f"Path: {path}")

        if meta:
            lines.append("")
            if product := meta.get("product", ""):
                lines.append(f"Product: {product}")
            if generated := meta.get("generated", ""):
                lines.append(f"Generated: {generated}")
            if channel := meta.get("channel", ""):
                lines.append(f"Channel: {channel}")
            src = meta.get("interp_source", {})
            if src:
                import time as _time
                src_mtime = src.get("mtime")
                src_age = (
                    "stale — interp_full.csv updated since this run"
                    if (self._interp_mtime and src_mtime and self._interp_mtime > src_mtime + 1)
                    else "current"
                )
                lines.append(f"Source data: {src_age}")
            settings = meta.get("settings", {})
            if settings:
                lines.append("")
                lines.append("Settings:")
                for k, v in settings.items():
                    lines.append(f"  {k}: {v}")
        elif path:
            lines.append("")
            lines.append("(no metadata — product predates run tracking)")

        # Sibling comparison for run nodes
        if node_type in ("3d_run", "slice_run") and path and meta:
            siblings = self._get_sibling_metas(path)
            if len(siblings) > 1:
                lines.append("")
                lines.append(f"── Siblings ({len(siblings)} runs) ──")
                my_settings = meta.get("settings", {})
                for sib_name, sib_meta in sorted(siblings.items()):
                    if sib_name == Path(path).name:
                        continue
                    sib_settings = sib_meta.get("settings", {})
                    diffs = []
                    all_keys = set(my_settings) | set(sib_settings)
                    for k in sorted(all_keys):
                        mv = my_settings.get(k, "—")
                        sv = sib_settings.get(k, "—")
                        if mv != sv:
                            diffs.append(f"{k}: {sv}")
                    sib_gen = sib_meta.get("generated", "")
                    if diffs:
                        lines.append(f"  {sib_name}  [{', '.join(diffs)}]  {sib_gen}")
                    else:
                        lines.append(f"  {sib_name}  (same settings)  {sib_gen}")

        self._detail.setPlainText("\n".join(lines))

    @staticmethod
    def _get_sibling_metas(run_path: str) -> dict:
        """Return {run_name: meta_dict} for all run_NNN siblings of run_path."""
        result = {}
        parent = Path(run_path).parent
        try:
            for d in sorted(parent.iterdir()):
                if d.is_dir() and d.name.startswith("run_"):
                    m = _read_meta(str(d))
                    if m:
                        result[d.name] = m
        except OSError:
            pass
        return result

    # ------------------------------------------------------------------
    # Auto-refresh and folder open
    # ------------------------------------------------------------------

    def _schedule_refresh(self, _path: str = "") -> None:
        QTimer.singleShot(600, self.refresh)

    def _open_folder(self) -> None:
        if not self._root:
            return
        path = self._root
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
