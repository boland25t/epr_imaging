"""
task_config_dialog.py — Create/edit dialog for a single stack Task.

Given a Task (new or existing), the list of available jobs, the sensor
channels present in the data, and the sampling tasks currently in the stack,
this dialog shows:
  • a Target selector  (Full dataset, or one of the configured jobs)
  • a Channels selector (per-channel task types only)
  • a type-specific settings form

For photogrammetry tasks the form is a scroll area with full Metashape/COLMAP
parameter control and a frame-source selector that can link to a sampling task
already in the stack (so no manual directory path is needed).

On accept the dialog writes the chosen values back into the Task object.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from models import Task, TASK_INFO


class TaskConfigDialog(QDialog):
    """Configure one Task's target, channels, and settings."""

    def __init__(
        self,
        task: Task,
        available_jobs: list[tuple],           # [(job_id, name), ...]
        available_channels: list[str],
        stack_sampling_tasks: list[tuple] = (), # [(task_id, display_label), ...]
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._task = task
        self._jobs = available_jobs
        self._channels = available_channels
        self._sampling_tasks = list(stack_sampling_tasks)
        self._widgets: dict = {}

        info = TASK_INFO.get(task.task_type, {})
        self.setWindowTitle(f"Configure: {info.get('label', task.task_type)}")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Target ────────────────────────────────────────────────────────────
        # Axis-1 batching: a task can run over the full dataset, a chosen set of
        # jobs (multi-select), or every job at once.
        tgt_group = QGroupBox("Target — which data this task runs on")
        tgt_layout = QVBoxLayout(tgt_group)
        self._tgt_full = QRadioButton("Full dataset")
        self._tgt_jobs = QRadioButton("Selected jobs")
        self._tgt_all  = QRadioButton("All jobs")
        tgt_layout.addWidget(self._tgt_full)
        tgt_layout.addWidget(self._tgt_jobs)

        self._job_list = QListWidget()
        self._job_list.setMaximumHeight(120)
        for job_id, name in available_jobs:
            item = QListWidgetItem(name or f"Job #{job_id}")
            item.setData(Qt.UserRole, job_id)
            item.setData(Qt.UserRole + 1, name)
            item.setCheckState(Qt.Unchecked)
            self._job_list.addItem(item)
        tgt_layout.addWidget(self._job_list)
        tgt_layout.addWidget(self._tgt_all)
        if not available_jobs:
            self._tgt_jobs.setEnabled(False)
            self._tgt_all.setEnabled(False)
            note = QLabel("No jobs with intervals — create one on the Jobs tab to batch.")
            note.setStyleSheet("color: #888; font-size: 10px;")
            tgt_layout.addWidget(note)
        self._restore_target_selection()
        self._tgt_jobs.toggled.connect(
            lambda on: self._job_list.setEnabled(on)
        )
        self._job_list.setEnabled(self._tgt_jobs.isChecked())
        layout.addWidget(tgt_group)

        # ── Channels (per-channel types only) ─────────────────────────────────
        if info.get("per_channel"):
            ch_group  = QGroupBox("Sensor channels  (none checked = all channels)")
            ch_layout = QVBoxLayout(ch_group)
            self._channel_list = QListWidget()
            self._channel_list.setMaximumHeight(120)
            checked = set(task.channels)
            for ch in available_channels:
                item = QListWidgetItem(ch)
                item.setCheckState(Qt.Checked if (ch in checked or not checked) else Qt.Unchecked)
                self._channel_list.addItem(item)
            ch_layout.addWidget(self._channel_list)
            if not available_channels:
                note = QLabel("No channels found — build interp_full.csv first.")
                note.setStyleSheet("color: #c0392b; font-size: 10px;")
                ch_layout.addWidget(note)
            layout.addWidget(ch_group)
        else:
            self._channel_list = None

        # ── Type-specific settings ────────────────────────────────────────────
        if task.task_type == "photogrammetry":
            layout.addWidget(self._build_photogrammetry_form())
        else:
            settings_group = QGroupBox("Settings")
            self._form = QFormLayout(settings_group)
            self._build_settings_form()
            layout.addWidget(settings_group)

        # ── Buttons ───────────────────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _restore_target_selection(self) -> None:
        """Set the radio + job checklist from the task's current target."""
        tgt  = self._task.target or {"kind": "full"}
        kind = tgt.get("kind", "full")
        # Collect the job ids the target currently references.
        sel_ids: set[int] = set()
        if kind == "job":
            sel_ids = {int(tgt.get("job_id", -1))}
        elif kind == "jobs":
            sel_ids = {int(j.get("job_id", -1)) for j in tgt.get("jobs", [])}

        if kind == "all_jobs" and self._tgt_all.isEnabled():
            self._tgt_all.setChecked(True)
        elif kind in ("job", "jobs") and self._tgt_jobs.isEnabled():
            self._tgt_jobs.setChecked(True)
            for i in range(self._job_list.count()):
                item = self._job_list.item(i)
                if item.data(Qt.UserRole) in sel_ids:
                    item.setCheckState(Qt.Checked)
        else:
            self._tgt_full.setChecked(True)

    def _collect_target(self) -> dict:
        """Build the target dict from the radio + job checklist."""
        if self._tgt_all.isChecked():
            return {"kind": "all_jobs"}
        if self._tgt_jobs.isChecked():
            jobs = []
            for i in range(self._job_list.count()):
                item = self._job_list.item(i)
                if item.checkState() == Qt.Checked:
                    jobs.append({
                        "job_id": item.data(Qt.UserRole),
                        "name":   item.data(Qt.UserRole + 1),
                    })
            if jobs:
                return {"kind": "jobs", "jobs": jobs}
            # No jobs checked → fall back to full so the task still runs.
            return {"kind": "full"}
        return {"kind": "full"}

    def _dspin(self, lo, hi, val, suffix="", step=None, dec=None) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        if suffix:
            w.setSuffix(suffix)
        if step is not None:
            w.setSingleStep(step)
        if dec is not None:
            w.setDecimals(dec)
        return w

    def _combo(self, items: list[str], current: str) -> QComboBox:
        c = QComboBox()
        c.addItems(items)
        if current in items:
            c.setCurrentText(current)
        return c

    def _check(self, label: str, checked: bool) -> QCheckBox:
        cb = QCheckBox(label)
        cb.setChecked(checked)
        return cb

    # -----------------------------------------------------------------------
    # Non-photogrammetry settings forms
    # -----------------------------------------------------------------------

    def _build_settings_form(self) -> None:
        t = self._task.task_type
        s = self._task.settings
        w = self._widgets
        form = self._form

        if t == "sampling":
            w["mode"] = QComboBox()
            w["mode"].addItems(["Fixed rate", "Dynamic spacing"])
            w["mode"].setCurrentText("Dynamic spacing" if s.get("mode") == "dynamic" else "Fixed rate")
            form.addRow("Mode:", w["mode"])
            w["frame_rate"] = self._dspin(0.01, 60.0, s.get("frame_rate", 1.0), " Hz", 0.1, 2)
            form.addRow("Frame rate:", w["frame_rate"])
            w["spacing_m"] = self._dspin(0.05, 100.0, s.get("spacing_m", 1.0), " m", 0.1, 2)
            form.addRow("Target spacing:", w["spacing_m"])
            w["quality"] = self._combo(["high", "medium", "low"], s.get("quality", "high"))
            form.addRow("Frame quality:", w["quality"])
            w["annotate"] = self._check("Annotate frames", bool(s.get("annotate", False)))
            form.addRow("", w["annotate"])
            w["rasters"] = self._check("Generate sensor rasters", bool(s.get("rasters", False)))
            form.addRow("", w["rasters"])

            def _sync_mode():
                dyn = w["mode"].currentText() == "Dynamic spacing"
                w["spacing_m"].setEnabled(dyn)
                w["frame_rate"].setEnabled(not dyn)
            w["mode"].currentIndexChanged.connect(lambda _: _sync_mode())
            _sync_mode()

        elif t == "nav_3d":
            w["cell_size"] = self._dspin(0.1, 100.0, s.get("cell_size", 1.0), " m")
            form.addRow("Cell size:", w["cell_size"])

        elif t == "nav_2d":
            w["cell_size"] = self._dspin(0.1, 100.0, s.get("cell_size", 5.0), " m")
            form.addRow("Cell size:", w["cell_size"])
            w["crs"] = self._combo(["UTM", "WGS84"], s.get("crs", "UTM"))
            form.addRow("CRS:", w["crs"])

        elif t == "sensor_3d":
            w["cell_size"] = self._dspin(0.1, 100.0, s.get("cell_size", 1.0), " m")
            form.addRow("Cell size:", w["cell_size"])
            w["aggregation"] = self._combo(["mean", "min", "max"], s.get("aggregation", "mean"))
            form.addRow("Aggregation:", w["aggregation"])
            w["fill"] = self._combo(
                ["IDW fill", "Kriging fill", "RBF fill", "No fill"],
                s.get("fill", "IDW fill"),
            )
            form.addRow("Fill:", w["fill"])
            w["zero_mask"] = self._dspin(0.0, 50.0, s.get("zero_mask", 5.0), " %", 1.0, 0)
            form.addRow("Near-zero mask:", w["zero_mask"])

        elif t == "sensor_2d":
            w["cell_size"] = self._dspin(0.1, 100.0, s.get("cell_size", 5.0), " m")
            form.addRow("Cell size:", w["cell_size"])
            w["fill"] = self._combo(
                ["IDW fill", "Kriging fill", "RBF fill", "Trackline only (no fill)"],
                s.get("fill", "IDW fill"),
            )
            form.addRow("Fill:", w["fill"])
            w["crs"] = self._combo(["UTM", "WGS84"], s.get("crs", "UTM"))
            form.addRow("CRS:", w["crs"])

        elif t == "depth_slice_geotiffs":
            w["altitude_step"] = self._dspin(0.1, 1000.0, s.get("altitude_step", 5.0), " m")
            form.addRow("Altitude step:", w["altitude_step"])
            w["cell_size"] = self._dspin(0.1, 100.0, s.get("cell_size", 2.0), " m")
            form.addRow("Cell size:", w["cell_size"])
            w["fill"] = self._combo(
                ["IDW fill", "RBF fill", "Trackline only"],
                s.get("fill", "IDW fill"),
            )
            form.addRow("Fill:", w["fill"])

        elif t == "sensor_slices":
            w["altitude_step"] = self._dspin(0.1, 1000.0, s.get("altitude_step", 5.0), " m")
            form.addRow("Altitude step:", w["altitude_step"])
            w["ppc"] = QSpinBox()
            w["ppc"].setRange(1, 20)
            w["ppc"].setValue(int(s.get("ppc", 4)))
            form.addRow("Pixels per cell:", w["ppc"])
            w["color"] = self._combo(
                ["viridis", "plasma", "inferno", "magma", "cividis",
                 "turbo", "coolwarm", "jet", "grayscale"],
                s.get("color", "viridis"),
            )
            form.addRow("Colormap:", w["color"])
            w["log_scale"] = self._check("Log scale", bool(s.get("log_scale", False)))
            w["log_scale"].setToolTip(
                "Apply logarithmic scaling before colour mapping.\n"
                "Spreads low-end variation that linear scale compresses."
            )
            form.addRow("", w["log_scale"])
            w["local_norm"] = self._check("Per-slice colour scale", bool(s.get("local_norm", False)))
            w["local_norm"].setToolTip(
                "Checked: each slice is normalised to its own min/max (interval-wise).\n"
                "Unchecked: all slices share the same colour scale (full-dataset-wise)."
            )
            form.addRow("", w["local_norm"])
            w["manual_range"] = self._check("Manual colour range", bool(s.get("manual_range", False)))
            w["manual_range"].setToolTip(
                "Fix the colour scale to explicit min/max values (raw data units),\n"
                "so slices stay comparable across separate runs / dives / jobs.\n"
                "When off, per-job runs inherit the full-dataset range automatically."
            )
            form.addRow("", w["manual_range"])
            w["vmin"] = self._dspin(-1e9, 1e9, float(s.get("vmin", 0.0)))
            w["vmax"] = self._dspin(-1e9, 1e9, float(s.get("vmax", 1.0)))
            w["vmin"].setDecimals(4)
            w["vmax"].setDecimals(4)
            form.addRow("Colour min:", w["vmin"])
            form.addRow("Colour max:", w["vmax"])

            def _sync_range(*_a) -> None:
                manual = w["manual_range"].isChecked()
                per_slice = w["local_norm"].isChecked()
                on = manual and not per_slice
                w["vmin"].setEnabled(on)
                w["vmax"].setEnabled(on)
                w["manual_range"].setEnabled(not per_slice)
            w["manual_range"].toggled.connect(_sync_range)
            w["local_norm"].toggled.connect(_sync_range)
            _sync_range()

            note = QLabel("Targets the most recent Sensor 3D PLY run for the same channel/target.")
            note.setStyleSheet("color: #888; font-size: 9px; font-style: italic;")
            note.setWordWrap(True)
            form.addRow("", note)

        elif t == "qgis_project":
            w["project_name"] = QLineEdit(s.get("project_name", "EPR Survey"))
            form.addRow("Project name:", w["project_name"])
            note = QLabel("Includes every GeoTIFF in the target's output directory.")
            note.setStyleSheet("color: #888; font-size: 9px; font-style: italic;")
            note.setWordWrap(True)
            form.addRow("", note)

        elif t == "sensor_netcdf":
            w["cell_size"] = self._dspin(0.1, 100.0, s.get("cell_size", 1.0), " m")
            form.addRow("Cell size:", w["cell_size"])
            w["aggregation"] = self._combo(
                ["mean", "min", "max", "count"], s.get("aggregation", "mean"),
            )
            form.addRow("Aggregation:", w["aggregation"])
            w["fill"] = self._combo(
                ["IDW fill", "Kriging fill", "RBF fill", "No fill"],
                s.get("fill", "IDW fill"),
            )
            form.addRow("Fill:", w["fill"])
            note = QLabel(
                "Writes a CF-1.8 gridded NetCDF (depth × northing × easting) in UTM, "
                "for xarray / QGIS / Panoply. One file per selected channel."
            )
            note.setStyleSheet("color: #888; font-size: 9px; font-style: italic;")
            note.setWordWrap(True)
            form.addRow("", note)

        elif t == "qc_report":
            w["max_gap"] = self._dspin(1.0, 36000.0, float(s.get("max_gap_s", 60.0)), " s")
            w["max_gap"].setToolTip(
                "A span between consecutive source sensor samples longer than this\n"
                "counts as a gap. Frames inside a gap received bridged (interpolated\n"
                "across the gap) values rather than true measurements."
            )
            form.addRow("Gap threshold:", w["max_gap"])
            note = QLabel(
                "Reports per-channel coverage, out-of-coverage frames, source gaps, "
                "value stats, and histograms for the target's interp record."
            )
            note.setStyleSheet("color: #888; font-size: 9px; font-style: italic;")
            note.setWordWrap(True)
            form.addRow("", note)

    # -----------------------------------------------------------------------
    # Photogrammetry form (scroll area with grouped sections)
    # -----------------------------------------------------------------------

    def _build_photogrammetry_form(self) -> QScrollArea:
        s = self._task.settings
        w = self._widgets

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(480)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setSpacing(8)
        vbox.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(container)

        # ── Frame Source ──────────────────────────────────────────────────────
        src_group = QGroupBox("Frame Source")
        src_vbox  = QVBoxLayout(src_group)

        w["from_task_radio"] = QRadioButton("From sampling task in stack")
        w["manual_radio"]    = QRadioButton("Manual directory")

        # Task chooser
        w["task_source_combo"] = QComboBox()
        for tid, tlabel in self._sampling_tasks:
            w["task_source_combo"].addItem(tlabel, userData=tid)
        if not self._sampling_tasks:
            w["from_task_radio"].setEnabled(False)
            w["from_task_radio"].setToolTip("Add a Sampling task to the stack first.")

        # Manual dir row
        w["frame_dir"] = QLineEdit(s.get("frame_dir", ""))
        w["frame_dir"].setPlaceholderText("Path to frames directory…")
        browse_btn = QPushButton("…")
        browse_btn.setMaximumWidth(30)
        browse_btn.clicked.connect(self._browse_frame_dir)
        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.addWidget(w["frame_dir"])
        dir_row.addWidget(browse_btn)

        src_vbox.addWidget(w["from_task_radio"])
        src_vbox.addWidget(w["task_source_combo"])
        src_vbox.addWidget(w["manual_radio"])
        src_vbox.addLayout(dir_row)

        # Restore state: depends_on set → from_task, else manual
        if self._task.depends_on is not None and self._sampling_tasks:
            w["from_task_radio"].setChecked(True)
            # Select the matching task in the combo
            for i in range(w["task_source_combo"].count()):
                if w["task_source_combo"].itemData(i) == self._task.depends_on:
                    w["task_source_combo"].setCurrentIndex(i)
                    break
        else:
            w["manual_radio"].setChecked(True)

        def _sync_src():
            from_task = w["from_task_radio"].isChecked()
            w["task_source_combo"].setEnabled(from_task and bool(self._sampling_tasks))
            w["frame_dir"].setEnabled(not from_task)
            browse_btn.setEnabled(not from_task)

        w["from_task_radio"].toggled.connect(lambda _: _sync_src())
        _sync_src()
        vbox.addWidget(src_group)

        # ── Engine ────────────────────────────────────────────────────────────
        eng_group = QGroupBox("Engine")
        eng_form  = QFormLayout(eng_group)
        w["engine"] = self._combo(["Metashape", "COLMAP"], s.get("engine", "Metashape"))
        eng_form.addRow("Engine:", w["engine"])
        vbox.addWidget(eng_group)

        # ── Metashape settings container ──────────────────────────────────────
        self._meta_widget = QWidget()
        meta_vbox = QVBoxLayout(self._meta_widget)
        meta_vbox.setContentsMargins(0, 0, 0, 0)
        meta_vbox.setSpacing(6)

        # -- Alignment --------------------------------------------------------
        align_group = QGroupBox("Alignment")
        align_form  = QFormLayout(align_group)
        w["align_accuracy"] = self._combo(
            ["Lowest", "Low", "Medium", "High", "Highest"],
            s.get("align_accuracy", "High"),
        )
        align_form.addRow("Accuracy:", w["align_accuracy"])
        w["key_point_limit"] = QSpinBox()
        w["key_point_limit"].setRange(0, 200000)
        w["key_point_limit"].setSingleStep(1000)
        w["key_point_limit"].setSpecialValueText("Auto")
        w["key_point_limit"].setValue(int(s.get("key_point_limit", 40000)))
        align_form.addRow("Key point limit:", w["key_point_limit"])
        w["tie_point_limit"] = QSpinBox()
        w["tie_point_limit"].setRange(0, 100000)
        w["tie_point_limit"].setSingleStep(1000)
        w["tie_point_limit"].setSpecialValueText("Auto")
        w["tie_point_limit"].setValue(int(s.get("tie_point_limit", 10000)))
        align_form.addRow("Tie point limit:", w["tie_point_limit"])
        w["generic_preselect"] = self._check("Generic preselection", bool(s.get("generic_preselect", True)))
        align_form.addRow("", w["generic_preselect"])
        w["reference_preselect"] = self._check("Reference preselection (GPS/nav)", bool(s.get("reference_preselect", True)))
        w["reference_preselect"].setToolTip("Uses navigation CSV to pre-seed camera positions before alignment.")
        align_form.addRow("", w["reference_preselect"])
        w["adaptive_fitting"] = self._check("Adaptive camera model fitting", bool(s.get("adaptive_fitting", True)))
        align_form.addRow("", w["adaptive_fitting"])
        w["reset_cameras"] = self._check("Reset cameras before alignment", bool(s.get("reset_cameras", False)))
        align_form.addRow("", w["reset_cameras"])
        meta_vbox.addWidget(align_group)

        # -- Dense Cloud ------------------------------------------------------
        dense_group = QGroupBox("Dense Cloud")
        dense_form  = QFormLayout(dense_group)
        w["build_dense"] = self._check("Build dense cloud", bool(s.get("build_dense", True)))
        dense_form.addRow("", w["build_dense"])
        w["dense_quality"] = self._combo(
            ["Ultra", "High", "Medium", "Low", "Lowest"],
            s.get("dense_quality", "Medium"),
        )
        dense_form.addRow("Quality:", w["dense_quality"])
        w["depth_filter"] = self._combo(
            ["Disabled", "Mild", "Moderate", "Aggressive"],
            s.get("depth_filter", "Moderate"),
        )
        dense_form.addRow("Depth filtering:", w["depth_filter"])
        w["reuse_depth"] = self._check("Reuse depth maps if present", bool(s.get("reuse_depth", False)))
        dense_form.addRow("", w["reuse_depth"])

        def _sync_dense():
            on = w["build_dense"].isChecked()
            for key in ("dense_quality", "depth_filter", "reuse_depth"):
                w[key].setEnabled(on)
        w["build_dense"].toggled.connect(lambda _: _sync_dense())
        _sync_dense()
        meta_vbox.addWidget(dense_group)

        # -- Mesh -------------------------------------------------------------
        mesh_group = QGroupBox("Mesh")
        mesh_form  = QFormLayout(mesh_group)
        w["build_mesh"] = self._check("Build mesh", bool(s.get("build_mesh", False)))
        mesh_form.addRow("", w["build_mesh"])
        w["mesh_surface"] = self._combo(
            ["Arbitrary", "Height Field"],
            s.get("mesh_surface", "Arbitrary"),
        )
        mesh_form.addRow("Surface type:", w["mesh_surface"])
        w["mesh_faces"] = self._combo(
            ["Low", "Medium", "High"],
            s.get("mesh_faces", "Medium"),
        )
        mesh_form.addRow("Face count:", w["mesh_faces"])
        w["mesh_source"] = self._combo(
            ["Dense cloud", "Depth maps"],
            s.get("mesh_source", "Dense cloud"),
        )
        mesh_form.addRow("Source data:", w["mesh_source"])
        w["mesh_vertex_colors"] = self._check("Calculate vertex colors", bool(s.get("mesh_vertex_colors", True)))
        mesh_form.addRow("", w["mesh_vertex_colors"])

        def _sync_mesh():
            on = w["build_mesh"].isChecked()
            for key in ("mesh_surface", "mesh_faces", "mesh_source", "mesh_vertex_colors"):
                w[key].setEnabled(on)
            # texture depends on mesh
            if "build_texture" in w:
                w["build_texture"].setEnabled(on)
                _sync_texture()
        w["build_mesh"].toggled.connect(lambda _: _sync_mesh())
        meta_vbox.addWidget(mesh_group)

        # -- Texture ----------------------------------------------------------
        tex_group = QGroupBox("Texture")
        tex_form  = QFormLayout(tex_group)
        w["build_texture"] = self._check("Build texture", bool(s.get("build_texture", False)))
        tex_form.addRow("", w["build_texture"])
        w["texture_size"] = self._combo(
            ["1024", "2048", "4096", "8192"],
            str(s.get("texture_size", 4096)),
        )
        tex_form.addRow("Texture size (px):", w["texture_size"])
        w["texture_blending"] = self._combo(
            ["Mosaic", "Average", "Min", "Max", "Disabled"],
            s.get("texture_blending", "Mosaic"),
        )
        tex_form.addRow("Blending mode:", w["texture_blending"])
        w["texture_fill_holes"] = self._check("Fill texture holes", bool(s.get("texture_fill_holes", True)))
        tex_form.addRow("", w["texture_fill_holes"])

        def _sync_texture():
            on = w["build_texture"].isChecked() and w["build_mesh"].isChecked()
            for key in ("texture_size", "texture_blending", "texture_fill_holes"):
                w[key].setEnabled(on)
        w["build_texture"].toggled.connect(lambda _: _sync_texture())

        # Now sync mesh (depends on texture being defined first)
        _sync_mesh()
        _sync_texture()
        meta_vbox.addWidget(tex_group)

        # -- Export & Project -------------------------------------------------
        exp_group = QGroupBox("Export & Project")
        exp_form  = QFormLayout(exp_group)
        w["export_dense_ply"] = self._check(
            "Export dense cloud as PLY (for 3D viewer)",
            bool(s.get("export_dense_ply", True)),
        )
        exp_form.addRow("", w["export_dense_ply"])
        w["export_mesh_obj"] = self._check("Export mesh as OBJ", bool(s.get("export_mesh_obj", False)))
        exp_form.addRow("", w["export_mesh_obj"])
        w["save_project"] = self._check(
            "Save Metashape project (.psx)",
            bool(s.get("save_project", True)),
        )
        exp_form.addRow("", w["save_project"])
        meta_vbox.addWidget(exp_group)

        # -- Georeference -----------------------------------------------------
        geo_group = QGroupBox("Georeference")
        geo_form  = QFormLayout(geo_group)
        w["use_nav_reference"] = self._check(
            "Use navigation CSV as camera reference",
            bool(s.get("use_nav_reference", True)),
        )
        w["use_nav_reference"].setToolTip(
            "Pre-seeds camera lat/lon/alt from interp_full.csv before alignment.\n"
            "Enables reference preselection and improves geolocation accuracy."
        )
        geo_form.addRow("", w["use_nav_reference"])
        w["nav_accuracy_h"] = self._dspin(0.001, 100.0, s.get("nav_accuracy_h", 0.1), " m", 0.01, 3)
        geo_form.addRow("Horizontal accuracy:", w["nav_accuracy_h"])
        w["nav_accuracy_v"] = self._dspin(0.001, 100.0, s.get("nav_accuracy_v", 0.5), " m", 0.01, 3)
        geo_form.addRow("Vertical accuracy:", w["nav_accuracy_v"])

        def _sync_geo():
            on = w["use_nav_reference"].isChecked()
            w["nav_accuracy_h"].setEnabled(on)
            w["nav_accuracy_v"].setEnabled(on)
        w["use_nav_reference"].toggled.connect(lambda _: _sync_geo())
        _sync_geo()
        meta_vbox.addWidget(geo_group)

        vbox.addWidget(self._meta_widget)

        # ── COLMAP settings container ─────────────────────────────────────────
        self._colmap_widget = QWidget()
        colmap_vbox = QVBoxLayout(self._colmap_widget)
        colmap_vbox.setContentsMargins(0, 0, 0, 0)
        colmap_vbox.setSpacing(6)

        # -- COLMAP: SfM / matching --------------------------------------------
        colmap_group = QGroupBox("COLMAP — Matching & SfM")
        colmap_form  = QFormLayout(colmap_group)
        w["max_features"] = QSpinBox()
        w["max_features"].setRange(512, 65536)
        w["max_features"].setSingleStep(1024)
        w["max_features"].setValue(int(s.get("max_features", 8192)))
        colmap_form.addRow("Max features:", w["max_features"])
        w["matcher"] = self._combo(
            ["Exhaustive", "Sequential", "Vocab Tree", "Spatial"],
            s.get("matcher", "Exhaustive"),
        )
        w["matcher"].setToolTip(
            "Exhaustive: checks all image pairs (best quality, slowest).\n"
            "Sequential: assumes images are taken in order (fast for video).\n"
            "Vocab Tree: approximate nearest-neighbour matching (large datasets).\n"
            "Spatial: uses navigation positions to limit pairs (needs nav)."
        )
        colmap_form.addRow("Matcher:", w["matcher"])
        w["single_camera"] = self._check(
            "Single shared camera (video frames)", bool(s.get("single_camera", True)))
        w["single_camera"].setToolTip(
            "Solve one shared intrinsic set for all frames — correct for video "
            "from one camera; more stable and faster.")
        colmap_form.addRow("", w["single_camera"])
        colmap_vbox.addWidget(colmap_group)

        # -- COLMAP: products --------------------------------------------------
        prod_group = QGroupBox("COLMAP — Products to generate")
        prod_form  = QVBoxLayout(prod_group)
        w["c_sparse"] = self._check("Sparse cloud (PLY)", True)
        w["c_sparse"].setChecked(True)
        w["c_sparse"].setEnabled(False)  # always produced
        w["c_sparse"].setToolTip("Always produced — the SfM result.")
        prod_form.addWidget(w["c_sparse"])
        w["c_traj"] = self._check(
            "Camera trajectory (poses + path)", bool(s.get("export_camera_trajectory", True)))
        prod_form.addWidget(w["c_traj"])
        w["run_mvs"] = self._check(
            "Dense cloud (MVS — needs CUDA GPU)", bool(s.get("run_mvs", True)))
        prod_form.addWidget(w["run_mvs"])
        w["c_undist"] = self._check(
            "Export undistorted frames", bool(s.get("export_undistorted", False)))
        prod_form.addWidget(w["c_undist"])
        w["c_depth"] = self._check(
            "Export depth + normal maps", bool(s.get("export_depth_maps", False)))
        prod_form.addWidget(w["c_depth"])
        w["c_poisson"] = self._check(
            "Poisson mesh (watertight, PLY)", bool(s.get("build_poisson_mesh", False)))
        prod_form.addWidget(w["c_poisson"])
        w["c_delaunay"] = self._check(
            "Delaunay mesh (detail-preserving, PLY)", bool(s.get("build_delaunay_mesh", False)))
        prod_form.addWidget(w["c_delaunay"])

        def _sync_colmap_dense():
            on = w["run_mvs"].isChecked()
            for key in ("c_undist", "c_depth", "c_poisson", "c_delaunay"):
                w[key].setEnabled(on)  # all dense-derived
        w["run_mvs"].toggled.connect(lambda _: _sync_colmap_dense())
        _sync_colmap_dense()
        colmap_vbox.addWidget(prod_group)

        # -- COLMAP: georeference ----------------------------------------------
        cgeo_group = QGroupBox("COLMAP — Georeference")
        cgeo_form  = QVBoxLayout(cgeo_group)
        w["c_georef"] = self._check(
            "Georeference to navigation (UTM E/N/-depth)",
            bool(s.get("georeference", True)))
        w["c_georef"].setToolTip(
            "Aligns the whole reconstruction to your nav track via model_aligner, "
            "so clouds/meshes come out in UTM and overlay the sensor 3-D products. "
            "Requires interp_full.csv / per-segment interp.csv.")
        cgeo_form.addWidget(w["c_georef"])
        note = QLabel("Orientation (heading/pitch/roll) cannot be used as a COLMAP "
                      "prior — only positions are used for alignment.")
        note.setStyleSheet("color: #888; font-size: 9px; font-style: italic;")
        note.setWordWrap(True)
        cgeo_form.addWidget(note)
        colmap_vbox.addWidget(cgeo_group)

        vbox.addWidget(self._colmap_widget)

        vbox.addStretch()

        # Show/hide engine-specific sections
        def _sync_engine():
            is_meta = w["engine"].currentText() == "Metashape"
            self._meta_widget.setVisible(is_meta)
            self._colmap_widget.setVisible(not is_meta)
        w["engine"].currentTextChanged.connect(lambda _: _sync_engine())
        _sync_engine()

        return scroll

    def _browse_frame_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Frame Directory")
        if d:
            self._widgets["frame_dir"].setText(d)

    # -----------------------------------------------------------------------
    # Accept
    # -----------------------------------------------------------------------

    def accept(self) -> None:
        """Write widget values back into the Task, then close."""
        task = self._task
        task.target = self._collect_target()

        if self._channel_list is not None:
            task.channels = [
                self._channel_list.item(i).text()
                for i in range(self._channel_list.count())
                if self._channel_list.item(i).checkState() == Qt.Checked
            ]

        s = task.settings
        w = self._widgets
        t = task.task_type

        if t == "sampling":
            s["mode"]       = "dynamic" if w["mode"].currentText() == "Dynamic spacing" else "fixed"
            s["frame_rate"] = w["frame_rate"].value()
            s["spacing_m"]  = w["spacing_m"].value()
            s["quality"]    = w["quality"].currentText()
            s["annotate"]   = w["annotate"].isChecked()
            s["rasters"]    = w["rasters"].isChecked()

        elif t == "nav_3d":
            s["cell_size"] = w["cell_size"].value()

        elif t == "nav_2d":
            s["cell_size"] = w["cell_size"].value()
            s["crs"]       = w["crs"].currentText()

        elif t == "sensor_3d":
            s["cell_size"]   = w["cell_size"].value()
            s["aggregation"] = w["aggregation"].currentText()
            s["fill"]        = w["fill"].currentText()
            s["zero_mask"]   = w["zero_mask"].value()

        elif t == "sensor_2d":
            s["cell_size"] = w["cell_size"].value()
            s["fill"]      = w["fill"].currentText()
            s["crs"]       = w["crs"].currentText()

        elif t == "depth_slice_geotiffs":
            s["altitude_step"] = w["altitude_step"].value()
            s["cell_size"]     = w["cell_size"].value()
            s["fill"]          = w["fill"].currentText()

        elif t == "sensor_slices":
            s["altitude_step"] = w["altitude_step"].value()
            s["ppc"]           = w["ppc"].value()
            s["color"]         = w["color"].currentText()
            s["log_scale"]     = w["log_scale"].isChecked()
            s["local_norm"]    = w["local_norm"].isChecked()
            s["manual_range"]  = w["manual_range"].isChecked()
            s["vmin"]          = w["vmin"].value()
            s["vmax"]          = w["vmax"].value()

        elif t == "photogrammetry":
            # Frame source
            if w["from_task_radio"].isChecked() and w["task_source_combo"].count() > 0:
                task.depends_on = w["task_source_combo"].currentData()
                s["frame_dir"]  = ""
            else:
                task.depends_on = None
                s["frame_dir"]  = w["frame_dir"].text().strip()

            s["engine"] = w["engine"].currentText()

            # Alignment
            s["align_accuracy"]     = w["align_accuracy"].currentText()
            s["key_point_limit"]    = w["key_point_limit"].value()
            s["tie_point_limit"]    = w["tie_point_limit"].value()
            s["generic_preselect"]  = w["generic_preselect"].isChecked()
            s["reference_preselect"] = w["reference_preselect"].isChecked()
            s["adaptive_fitting"]   = w["adaptive_fitting"].isChecked()
            s["reset_cameras"]      = w["reset_cameras"].isChecked()

            # Dense cloud
            s["build_dense"]   = w["build_dense"].isChecked()
            s["dense_quality"] = w["dense_quality"].currentText()
            s["depth_filter"]  = w["depth_filter"].currentText()
            s["reuse_depth"]   = w["reuse_depth"].isChecked()

            # Mesh
            s["build_mesh"]        = w["build_mesh"].isChecked()
            s["mesh_surface"]      = w["mesh_surface"].currentText()
            s["mesh_faces"]        = w["mesh_faces"].currentText()
            s["mesh_source"]       = w["mesh_source"].currentText()
            s["mesh_vertex_colors"] = w["mesh_vertex_colors"].isChecked()

            # Texture
            s["build_texture"]      = w["build_texture"].isChecked()
            s["texture_size"]       = int(w["texture_size"].currentText())
            s["texture_blending"]   = w["texture_blending"].currentText()
            s["texture_fill_holes"] = w["texture_fill_holes"].isChecked()

            # Export & project
            s["export_dense_ply"] = w["export_dense_ply"].isChecked()
            s["export_mesh_obj"]  = w["export_mesh_obj"].isChecked()
            s["save_project"]     = w["save_project"].isChecked()

            # Georeference
            s["use_nav_reference"] = w["use_nav_reference"].isChecked()
            s["nav_accuracy_h"]    = w["nav_accuracy_h"].value()
            s["nav_accuracy_v"]    = w["nav_accuracy_v"].value()

            # COLMAP — matching / SfM
            s["max_features"]  = w["max_features"].value()
            s["matcher"]       = w["matcher"].currentText()
            s["single_camera"] = w["single_camera"].isChecked()
            # COLMAP — products
            s["run_mvs"]                  = w["run_mvs"].isChecked()
            s["export_camera_trajectory"] = w["c_traj"].isChecked()
            s["export_undistorted"]       = w["c_undist"].isChecked()
            s["export_depth_maps"]        = w["c_depth"].isChecked()
            s["build_poisson_mesh"]       = w["c_poisson"].isChecked()
            s["build_delaunay_mesh"]      = w["c_delaunay"].isChecked()
            # COLMAP — georeference
            s["georeference"]             = w["c_georef"].isChecked()

        elif t == "qgis_project":
            s["project_name"] = w["project_name"].text().strip() or "EPR Survey"

        elif t == "sensor_netcdf":
            s["cell_size"]   = w["cell_size"].value()
            s["aggregation"] = w["aggregation"].currentText()
            s["fill"]        = w["fill"].currentText()

        elif t == "qc_report":
            s["max_gap_s"] = w["max_gap"].value()

        super().accept()

    def task(self) -> Task:
        return self._task
