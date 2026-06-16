"""
viewer_widget.py — 3D point-cloud / mesh overlay viewer

Standalone QMainWindow that embeds a PyVista/VTK viewport.  Multiple PLY,
OBJ, or GeoTIFF-derived mesh files can be loaded as independent layers and
viewed simultaneously in a shared world coordinate frame.

Typical use: load a sensor 3D PLY (UTM world coordinates) alongside a
photogrammetry dense cloud or mesh from the same location.  Both appear in
the same viewport because they share the same UTM coordinate system.

Usage (from main_window.py):
    viewer = get_viewer()          # singleton; creates once
    viewer.load_file(path)         # adds a layer and shows the window
    viewer.show()
    viewer.raise_()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QComboBox,
)

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_OK = True
except ImportError:
    PYVISTA_OK = False


# ---------------------------------------------------------------------------
# Potree (octree streaming viewer) integration
# ---------------------------------------------------------------------------

def detect_potree_converter() -> Optional[str]:
    """Return the PotreeConverter executable path, or None if not installed."""
    import shutil
    for name in ("PotreeConverter", "potree_converter", "PotreeConverter.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


class PotreeWorker(QObject):
    """Runs PotreeConverter on a point cloud in a background thread.

    PotreeConverter 1.x can emit a self-contained HTML page (--generate-page);
    2.x produces an octree that needs a Potree page to view.  We try the
    1.x page-generating form first and fall back to a bare conversion,
    reporting whichever output we can open.
    """

    finished = Signal(str)   # path to open (html file or output dir)
    error    = Signal(str)
    log      = Signal(str)

    def __init__(self, converter: str, input_path: str, output_dir: str):
        super().__init__()
        self.converter = converter
        self.input_path = input_path
        self.output_dir = output_dir

    def run(self) -> None:
        import subprocess
        from pathlib import Path as _P
        try:
            out = _P(self.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            page_name = "index"
            # Attempt the 1.x page-generating invocation first.
            cmds = [
                [self.converter, self.input_path, "-o", str(out),
                 "--generate-page", page_name],
                [self.converter, self.input_path, "-o", str(out)],
            ]
            last_err = ""
            for cmd in cmds:
                self.log.emit("Running: " + " ".join(cmd))
                proc = subprocess.run(cmd, capture_output=True, text=True)
                for line in (proc.stdout or "").splitlines():
                    if line.strip():
                        self.log.emit("  " + line.rstrip())
                if proc.returncode == 0:
                    # Prefer a generated HTML page; else hand back the directory.
                    html = self._find_html(out, page_name)
                    self.finished.emit(html or str(out))
                    return
                last_err = (proc.stderr or proc.stdout or "").strip()
            self.error.emit(last_err or "PotreeConverter failed.")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    @staticmethod
    def _find_html(out, page_name: str) -> Optional[str]:
        from pathlib import Path as _P
        for cand in (out / f"{page_name}.html", out / "index.html"):
            if cand.exists():
                return str(cand)
        htmls = list(_P(out).rglob("*.html"))
        return str(htmls[0]) if htmls else None


# ---------------------------------------------------------------------------
# Layer data class
# ---------------------------------------------------------------------------

# Above this many points, a cloud is downsampled on load unless the user asks
# for full resolution.  Keeps interaction smooth and avoids GPU-memory blowups
# with photogrammetry dense clouds (often 10–50M points).
DEFAULT_MAX_POINTS = 2_000_000


class Layer:
    """Metadata and state for one loaded file."""

    def __init__(self, path: str, name: str):
        self.path         = path
        self.name         = name
        self.visible      = True
        self.opacity      = 1.0
        self.color_rgb    = False   # True = use file RGB; False = scalar/viridis
        self.actor        = None    # vtk actor reference (set on load)
        self.mesh         = None    # the mesh actually rendered (possibly downsampled)
        self.full_points  = 0       # point count in the file on disk
        self.shown_points = 0       # point count currently rendered
        self.downsampled  = False   # True when shown < full
        self.force_full   = False   # user asked to render every point

    def display_name(self) -> str:
        return self.name or Path(self.path).name

    def count_label(self) -> str:
        if self.downsampled:
            return f"{self.shown_points:,} / {self.full_points:,}"
        return f"{self.full_points:,}"


# ---------------------------------------------------------------------------
# Layer row widget (embedded in QListWidget)
# ---------------------------------------------------------------------------

class LayerRowWidget(QFrame):
    """One row in the layer list: [eye] [name] [opacity] [mode] [x]"""

    visibility_changed = Signal(bool)
    opacity_changed    = Signal(float)
    color_mode_changed = Signal(bool)  # True = RGB, False = scalar
    remove_requested   = Signal()
    full_res_requested = Signal()      # user clicked "Full" on a downsampled layer

    def __init__(self, layer: Layer, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layer = layer
        self.setFrameShape(QFrame.StyledPanel)
        self.setLineWidth(1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        self.vis_check = QCheckBox()
        self.vis_check.setChecked(layer.visible)
        self.vis_check.setToolTip("Toggle visibility")
        self.vis_check.toggled.connect(self.visibility_changed)
        layout.addWidget(self.vis_check)

        name_col = QVBoxLayout()
        name_col.setContentsMargins(0, 0, 0, 0)
        name_col.setSpacing(0)
        self.name_label = QLabel(layer.display_name())
        self.name_label.setToolTip(layer.path)
        self.name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        font = QFont()
        font.setPointSize(9)
        self.name_label.setFont(font)
        name_col.addWidget(self.name_label)

        # Point-count line, shows "shown / full" when downsampled.
        self.count_label = QLabel(layer.count_label() + " pts")
        cf = QFont(); cf.setPointSize(7)
        self.count_label.setFont(cf)
        self.count_label.setStyleSheet(
            "color: #d08770;" if layer.downsampled else "color: #888;"
        )
        if layer.downsampled:
            self.count_label.setToolTip(
                "This cloud was downsampled for smooth display.\n"
                "Click 'Full' to render every point (may be slow)."
            )
        name_col.addWidget(self.count_label)
        layout.addLayout(name_col, stretch=1)

        # "Full" button — only meaningful while the layer is downsampled.
        self.full_btn = QPushButton("Full")
        self.full_btn.setFixedWidth(40)
        self.full_btn.setToolTip("Reload this layer at full resolution (every point)")
        self.full_btn.setVisible(layer.downsampled)
        self.full_btn.clicked.connect(self.full_res_requested)
        layout.addWidget(self.full_btn)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["RGB", "Viridis"])
        self.mode_combo.setCurrentIndex(0 if layer.color_rgb else 1)
        self.mode_combo.setToolTip("Color mode: RGB from file, or viridis mapped to scalar value")
        self.mode_combo.setFixedWidth(72)
        self.mode_combo.currentIndexChanged.connect(
            lambda i: self.color_mode_changed.emit(i == 0)
        )
        layout.addWidget(self.mode_combo)

        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setValue(layer.opacity)
        self.opacity_spin.setFixedWidth(58)
        self.opacity_spin.setToolTip("Opacity (0 = transparent, 1 = opaque)")
        self.opacity_spin.valueChanged.connect(self.opacity_changed)
        layout.addWidget(self.opacity_spin)

        rm_btn = QPushButton("✕")
        rm_btn.setFixedSize(20, 20)
        rm_btn.setToolTip("Remove layer")
        rm_btn.clicked.connect(self.remove_requested)
        layout.addWidget(rm_btn)


# ---------------------------------------------------------------------------
# Main viewer window
# ---------------------------------------------------------------------------

class PointCloudViewer(QMainWindow):
    """Standalone window: layer list + shared PyVista VTK viewport.

    Supports PLY point clouds and OBJ meshes.  All loaded files share the
    same world coordinate frame — layers in UTM (easting/northing/depth)
    are overlay-compatible out of the box.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("3D Product Viewer")
        self.resize(1280, 800)

        self._layers: list[Layer] = []
        self._list_items: dict[int, QListWidgetItem] = {}  # layer id → list item
        self._potree_thread = None
        self._potree_worker = None

        self._build_ui()

    def _build_ui(self) -> None:
        # ── Toolbar ──────────────────────────────────────────────────────────
        tb = QToolBar("Viewer", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        add_btn = QPushButton("Add Layer…")
        add_btn.clicked.connect(self._add_layer_dialog)
        tb.addWidget(add_btn)

        tb.addSeparator()

        reset_btn = QPushButton("Reset Camera")
        reset_btn.clicked.connect(self._reset_camera)
        tb.addWidget(reset_btn)

        bg_btn = QPushButton("Toggle Background")
        bg_btn.clicked.connect(self._toggle_background)
        tb.addWidget(bg_btn)
        self._dark_bg = True

        tb.addSeparator()

        screenshot_btn = QPushButton("Screenshot…")
        screenshot_btn.clicked.connect(self._take_screenshot)
        tb.addWidget(screenshot_btn)

        tb.addSeparator()

        # Potree fallback for very large clouds (octree streaming in a browser).
        self._potree_btn = QPushButton("Open in Browser (Potree)…")
        self._potree_btn.setToolTip(
            "Convert the selected layer to a Potree octree and open it in your\n"
            "browser — the right tool for clouds too large to render in-app\n"
            "(>20M points). Requires PotreeConverter on PATH."
        )
        self._potree_btn.clicked.connect(self._export_potree)
        tb.addWidget(self._potree_btn)

        # ── Central splitter ─────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(splitter)

        # Left: layer list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        left_header = QLabel("Layers")
        font = QFont()
        font.setBold(True)
        left_header.setFont(font)
        left_layout.addWidget(left_header)

        self._layer_list = QListWidget()
        self._layer_list.setSelectionMode(QListWidget.SingleSelection)
        left_layout.addWidget(self._layer_list)

        left.setMinimumWidth(280)
        left.setMaximumWidth(420)
        splitter.addWidget(left)

        # Right: PyVista viewport
        if PYVISTA_OK:
            self._plotter = QtInteractor(self)
            self._plotter.set_background("#1a1a2e")  # dark blue-black
            splitter.addWidget(self._plotter)
        else:
            placeholder = QLabel(
                "PyVista is not installed.\n\n"
                "Install with:\n"
                "  pip install pyvista pyvistaqt"
            )
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #888; font-size: 13px;")
            splitter.addWidget(placeholder)
            self._plotter = None

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Status bar
        self.statusBar().showMessage("No layers loaded.")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def load_file(self, path: str, name: Optional[str] = None,
                  force_full: bool = False) -> None:
        """Load a PLY or OBJ file as a new layer.  Thread-safe: call from main thread.

        Clouds larger than DEFAULT_MAX_POINTS are downsampled for display unless
        force_full is True.  The on-disk file is never modified.
        """
        path = str(path)
        name = name or Path(path).stem.replace("_", " ")

        # Don't load the same file twice
        if any(l.path == path for l in self._layers):
            self.statusBar().showMessage(f"Already loaded: {name}")
            return

        if not PYVISTA_OK:
            return

        try:
            mesh = pv.read(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"Cannot load:\n{path}\n\n{exc}")
            return

        layer = Layer(path, name)
        layer.color_rgb  = self._mesh_has_rgb(mesh)
        layer.force_full = force_full
        layer.full_points = self._point_count(mesh)

        render_mesh, downsampled = self._maybe_downsample(mesh, force_full)
        layer.mesh         = render_mesh
        layer.downsampled  = downsampled
        layer.shown_points = self._point_count(render_mesh)

        self._render_layer(layer)
        self._layers.append(layer)
        self._add_list_row(layer)

        if downsampled:
            self.statusBar().showMessage(
                f"Loaded: {name}  —  showing {layer.shown_points:,} of "
                f"{layer.full_points:,} points (downsampled for display)"
            )
        else:
            self.statusBar().showMessage(f"Loaded: {name}  ({layer.full_points:,} points)")
        self.show()
        self.raise_()

    def reload_layer_full(self, layer: Layer) -> None:
        """Re-render a layer at full resolution (every point)."""
        if not PYVISTA_OK:
            return
        try:
            mesh = pv.read(layer.path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"Cannot reload:\n{layer.path}\n\n{exc}")
            return
        self.statusBar().showMessage(
            f"Loading {layer.display_name()} at full resolution "
            f"({layer.full_points:,} points)…"
        )
        QApplication.processEvents()
        layer.force_full   = True
        layer.mesh         = mesh
        layer.downsampled  = False
        layer.shown_points = self._point_count(mesh)
        self._render_layer(layer)
        self._refresh_row_for(layer)
        self.statusBar().showMessage(
            f"{layer.display_name()}: full resolution ({layer.shown_points:,} points)"
        )

    def remove_layer(self, layer: Layer) -> None:
        if layer.actor is not None and self._plotter:
            self._plotter.remove_actor(layer.actor)
        idx = self._layers.index(layer)
        self._layers.remove(layer)
        self._layer_list.takeItem(idx)
        self._plotter.update() if self._plotter else None

    # -----------------------------------------------------------------------
    # Private: viewport helpers
    # -----------------------------------------------------------------------

    def _mesh_has_rgb(self, mesh) -> bool:
        """Return True if the mesh has meaningful RGB vertex colors."""
        if "RGB" in mesh.point_data.keys():
            return True
        if "Red" in mesh.point_data.keys():
            return True
        if mesh.active_scalars_name and mesh.active_scalars_name.upper() in ("RGB", "RGBA"):
            return True
        # PLY files loaded by pyvista often expose colors as 'RGB' or individual channels
        color_names = {"Red", "Green", "Blue", "R", "G", "B", "RGB", "RGBA"}
        return bool(color_names & set(mesh.point_data.keys()))

    def _maybe_downsample(self, mesh, force_full: bool):
        """Return (render_mesh, downsampled_bool).

        Surface meshes (with faces) are decimated by reduction ratio; point
        clouds are randomly subsampled (preserving per-point RGB/scalars).
        """
        n = self._point_count(mesh)
        if force_full or n <= DEFAULT_MAX_POINTS:
            return mesh, False
        target = DEFAULT_MAX_POINTS
        try:
            import numpy as np
            # n_faces returns polygonal faces on current pyvista; guard for older versions.
            n_faces = getattr(mesh, "n_faces", 0) or 0
            has_faces = isinstance(mesh, pv.PolyData) and n_faces > 0
            if has_faces:
                reduction = max(0.0, min(0.999, 1.0 - target / n))
                try:
                    return mesh.triangulate().decimate(reduction), True
                except Exception:
                    pass  # fall through to point subsample
            rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(n, size=target, replace=False))
            sub = mesh.extract_points(idx, adjacent_cells=False, include_cells=False)
            return sub, True
        except Exception:
            # If anything goes wrong, render the full mesh rather than fail.
            return mesh, False

    def _render_layer(self, layer: "Layer") -> None:
        """(Re)create the layer's actor from layer.mesh as a vtkLODActor.

        pyvista's add_mesh configures the mapper's colouring correctly (RGB
        direct-scalars or viridis LUT); we then swap the plain Actor for a
        vtkLODActor that reuses that mapper, giving automatic level-of-detail
        during camera interaction at no cost to still-frame quality.
        """
        if not self._plotter:
            return
        if layer.actor is not None:
            try:
                self._plotter.remove_actor(layer.actor)
            except Exception:
                pass
            layer.actor = None

        mesh = layer.mesh
        kwargs = {"opacity": layer.opacity, "name": layer.path}
        if layer.color_rgb and self._mesh_has_rgb(mesh):
            kwargs["rgb"] = True
            kwargs["scalars"] = None
        else:
            kwargs["cmap"] = "viridis"
            kwargs["show_scalar_bar"] = False

        actor = self._plotter.add_mesh(mesh, **kwargs)
        layer.actor = self._promote_to_lod(actor, layer)
        layer.actor.SetVisibility(layer.visible)
        self._plotter.update()

    def _promote_to_lod(self, actor, layer: "Layer"):
        """Swap a plain vtk Actor for a vtkLODActor reusing its mapper/property."""
        try:
            import vtk
            mapper = actor.GetMapper()
            prop   = actor.GetProperty()
            self._plotter.remove_actor(actor)
            lod = vtk.vtkLODActor()
            lod.SetMapper(mapper)
            lod.SetProperty(prop)
            # Medium LOD: a random point cloud capped well below the full set so
            # rotation stays interactive even at full resolution.
            lod.SetNumberOfCloudPoints(min(layer.shown_points or 50000, 100_000))
            self._plotter.add_actor(lod, name=layer.path, render=False)
            return lod
        except Exception:
            # LOD is an optimisation; fall back to the plain actor on any failure.
            return actor

    def _point_count(self, mesh) -> int:
        try:
            return mesh.n_points
        except Exception:
            return 0

    def _refresh_row_for(self, layer: "Layer") -> None:
        """Update the list row widget (count + Full button) for a layer."""
        try:
            idx = self._layers.index(layer)
        except ValueError:
            return
        item = self._layer_list.item(idx)
        if item is None:
            return
        row = self._layer_list.itemWidget(item)
        if row is None:
            return
        row.count_label.setText(layer.count_label() + " pts")
        row.count_label.setStyleSheet("color: #d08770;" if layer.downsampled else "color: #888;")
        row.full_btn.setVisible(layer.downsampled)

    # -----------------------------------------------------------------------
    # Private: layer list management
    # -----------------------------------------------------------------------

    def _add_list_row(self, layer: Layer) -> None:
        row_widget = LayerRowWidget(layer)

        row_widget.visibility_changed.connect(
            lambda v, l=layer: self._on_visibility(l, v)
        )
        row_widget.opacity_changed.connect(
            lambda v, l=layer: self._on_opacity(l, v)
        )
        row_widget.color_mode_changed.connect(
            lambda rgb, l=layer: self._on_color_mode(l, rgb)
        )
        row_widget.remove_requested.connect(
            lambda l=layer: self.remove_layer(l)
        )
        row_widget.full_res_requested.connect(
            lambda l=layer: self.reload_layer_full(l)
        )

        item = QListWidgetItem(self._layer_list)
        item.setSizeHint(row_widget.sizeHint())
        self._layer_list.addItem(item)
        self._layer_list.setItemWidget(item, row_widget)

    def _on_visibility(self, layer: Layer, visible: bool) -> None:
        layer.visible = visible
        if layer.actor:
            layer.actor.SetVisibility(visible)
        if self._plotter:
            self._plotter.update()

    def _on_opacity(self, layer: Layer, value: float) -> None:
        layer.opacity = value
        if layer.actor:
            layer.actor.GetProperty().SetOpacity(value)
        if self._plotter:
            self._plotter.update()

    def _on_color_mode(self, layer: Layer, use_rgb: bool) -> None:
        """Re-render the (already-loaded) mesh with a different colour mode."""
        layer.color_rgb = use_rgb
        if not self._plotter or layer.mesh is None:
            return
        self._render_layer(layer)

    # -----------------------------------------------------------------------
    # Toolbar actions
    # -----------------------------------------------------------------------

    def _add_layer_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add 3D Layer",
            "",
            "3D Files (*.ply *.obj *.stl *.vtk);;All Files (*)",
        )
        for path in paths:
            self.load_file(path)

    def _reset_camera(self) -> None:
        if self._plotter:
            self._plotter.reset_camera()
            self._plotter.update()

    def _toggle_background(self) -> None:
        if not self._plotter:
            return
        self._dark_bg = not self._dark_bg
        self._plotter.set_background("#1a1a2e" if self._dark_bg else "#f0f0f0")
        self._plotter.update()

    def _take_screenshot(self) -> None:
        if not self._plotter:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "", "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            self._plotter.screenshot(path)
            self.statusBar().showMessage(f"Screenshot saved: {path}")

    # -----------------------------------------------------------------------
    # Potree export (octree streaming for very large clouds)
    # -----------------------------------------------------------------------

    def _selected_layer(self) -> Optional["Layer"]:
        item = self._layer_list.currentItem()
        if item is None:
            return self._layers[0] if self._layers else None
        idx = self._layer_list.row(item)
        return self._layers[idx] if 0 <= idx < len(self._layers) else None

    def _export_potree(self) -> None:
        """Convert a cloud to a Potree octree and open it in the browser."""
        converter = detect_potree_converter()
        if not converter:
            QMessageBox.information(
                self, "PotreeConverter not found",
                "PotreeConverter is not installed or not on PATH.\n\n"
                "Potree renders arbitrarily large clouds by streaming an octree "
                "in the browser — ideal for >20M-point dense clouds.\n\n"
                "Install it from https://github.com/potree/PotreeConverter and "
                "ensure 'PotreeConverter' is on your PATH, then try again."
            )
            return

        # Source file: the selected layer's path, or ask.
        layer = self._selected_layer()
        src = layer.path if layer else ""
        if not src:
            src, _ = QFileDialog.getOpenFileName(
                self, "Cloud to convert", "", "Point clouds (*.ply *.las *.laz);;All Files (*)"
            )
        if not src:
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Output directory for Potree octree")
        if not out_dir:
            return
        out_dir = str(Path(out_dir) / (Path(src).stem + "_potree"))

        if self._potree_thread is not None:
            QMessageBox.warning(self, "Busy", "A Potree conversion is already running.")
            return

        from PySide6.QtCore import QThread
        self.statusBar().showMessage(f"Converting {Path(src).name} to Potree octree…")
        thread = QThread(self)
        worker = PotreeWorker(converter, src, out_dir)
        self._potree_thread = thread
        self._potree_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(lambda m: self.statusBar().showMessage(m))
        worker.finished.connect(self._on_potree_finished)
        worker.error.connect(self._on_potree_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_potree_worker)
        thread.start()

    def _clear_potree_worker(self) -> None:
        self._potree_thread = None
        self._potree_worker = None

    def _on_potree_finished(self, target: str) -> None:
        import webbrowser
        p = Path(target)
        if p.suffix.lower() == ".html":
            webbrowser.open(p.as_uri())
            self.statusBar().showMessage(f"Opened Potree page: {target}")
        else:
            # Bare octree (PotreeConverter 2.x) — no standalone page produced.
            webbrowser.open(p.as_uri())
            QMessageBox.information(
                self, "Potree octree ready",
                f"Octree written to:\n{target}\n\n"
                "PotreeConverter 2.x does not emit a standalone page. Serve this "
                "folder with a Potree viewer (https://github.com/potree/potree) "
                "to view it in the browser."
            )

    def _on_potree_error(self, message: str) -> None:
        QMessageBox.critical(self, "Potree conversion failed", message)
        self.statusBar().showMessage("Potree conversion failed.")


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_viewer_instance: Optional[PointCloudViewer] = None


def get_viewer(parent: Optional[QWidget] = None) -> PointCloudViewer:
    """Return the singleton PointCloudViewer, creating it if needed."""
    global _viewer_instance
    if _viewer_instance is None:
        _viewer_instance = PointCloudViewer(parent)
    return _viewer_instance


def viewer_available() -> bool:
    """Return True if pyvista is importable."""
    return PYVISTA_OK
