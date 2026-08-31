"""
one_click_dialog.py — Guided builder for a complete processing pipeline.

One dialog → a fully wired Task Stack → an immediate run.  The user picks the
target (a job, the full dataset, or the anomaly windows) and ticks the products
they want; the dialog generates ordinary Task objects in dependency order:

    build_interp → anomaly_detect → job_interp → sampling → products → photogrammetry

Photogrammetry is linked to the generated sampling task via depends_on, so the
extracted frames flow straight into Metashape/COLMAP with no manual paths.
The generated tasks land in the normal Task Stack and stay individually
editable — One-Click is a task *generator*, not a separate execution path.

Anomalies appear in two independent places:

  * as a STEP  — an `anomaly_detect` task that (re)runs the MATLAB detector
    and/or rebuilds the site catalog, report, spreadsheets and QGIS bundle;
  * as a TARGET — "Anomaly windows", which turns the windows of an ALREADY
    BUILT catalog into a job right when the stack is generated, so sampling,
    products and photogrammetry all run on the anomalous stretches.

Those are deliberately separate because `_build_task_plan()` resolves every
task target BEFORE the stack runs: a task cannot target a job that a
later-running task in the same stack creates.  So the target list is built from
the catalog on disk, and a detector step queued in the same run refreshes that
catalog for NEXT time.
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
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from models import Task, TaskStack, TASK_INFO
from task_config_dialog import TaskConfigDialog


# Product task types offered in the "Data products" section, in generation
# order.  (label, task_type, per_channel) — labels mirror TASK_INFO.
# NOTE: sensor_3d (volumetric point cloud) and sensor_slices (its PNG child) are
# archived — see models.ARCHIVED_TASK_TYPES — so they are omitted here.  The
# depth_slice_geotiffs raster is the independent depth-band product that remains.
_PRODUCT_TYPES: list[tuple[str, str]] = [
    ("nav_3d",               "Nav Trackline PLY"),
    ("nav_2d",               "Nav Depth GeoTIFF"),
    ("sensor_2d",            "Sensor 2D GeoTIFF (per channel)"),
    ("depth_slice_geotiffs", "Depth-Slice GeoTIFFs (per channel)"),
    ("sensor_netcdf",        "Sensor NetCDF (CF, per channel)"),
    ("qc_report",            "Data QC Report"),
    ("qgis_project",         "QGIS Project (.qgs)"),
]


class OneClickPipelineDialog(QDialog):
    """Assemble target + products into a ready-to-run task stack."""

    def __init__(
        self,
        jobs: list[tuple],              # [(job_id, name), ...] with intervals
        channels: list[str],            # sensor channels from interp_full.csv
        availability: dict,             # from MainWindow._stack_data_availability
        interp_exists: bool,
        anomaly_info: Optional[dict] = None,  # {"tier_counts", "matlab_reason", "catalog_exists"}
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._channels = channels
        self._avail = availability or {}
        self._interp_exists = interp_exists
        self._anomaly = anomaly_info or {}

        # Prototype photogrammetry task: holds the full engine settings edited
        # via the existing TaskConfigDialog ("Configure…" button).  task_id 0 is
        # a placeholder — real ids are assigned at generation time.
        self._photo_proto = Task(task_id=0, task_type="photogrammetry")

        self.setWindowTitle("One-Click Pipeline")
        self.setMinimumSize(560, 640)
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        intro = QLabel(
            "Pick a target and the products you want.  OK generates a fully "
            "wired task stack (sampling feeds photogrammetry automatically) "
            "and runs it immediately.  The generated tasks remain editable in "
            "the Task Stack afterwards."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer.addWidget(scroll, stretch=1)
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setSpacing(8)
        scroll.setWidget(content)

        # ── Target ────────────────────────────────────────────────────────
        tgt_group = QGroupBox("Target")
        tgt_vbox = QVBoxLayout(tgt_group)
        self._tgt_full = QRadioButton("Full dataset")
        self._tgt_jobs = QRadioButton("Selected job(s)")
        tgt_vbox.addWidget(self._tgt_full)
        tgt_vbox.addWidget(self._tgt_jobs)
        self._job_list = QListWidget()
        self._job_list.setMaximumHeight(110)
        for job_id, name in self._jobs:
            item = QListWidgetItem(name or f"Job #{job_id}")
            item.setData(Qt.UserRole, job_id)
            item.setData(Qt.UserRole + 1, name)
            item.setCheckState(Qt.Unchecked)
            self._job_list.addItem(item)
        if self._jobs:
            self._tgt_jobs.setChecked(True)
            self._job_list.item(0).setCheckState(Qt.Checked)
        else:
            self._tgt_full.setChecked(True)
            self._tgt_jobs.setEnabled(False)
            note = QLabel("No jobs with intervals — assemble one on the Jobs tab "
                          "(or Import Intervals CSV) to target a job.")
            note.setStyleSheet("color: #888; font-size: 10px;")
            tgt_vbox.addWidget(note)
        tgt_vbox.addWidget(self._job_list)
        self._tgt_jobs.toggled.connect(self._job_list.setEnabled)
        self._job_list.setEnabled(self._tgt_jobs.isChecked())

        # ── Anomaly-window target ─────────────────────────────────────────
        # Built from the catalog ON DISK at generation time (see module docstring
        # for why it cannot come from a detector step in this same stack).
        counts = self._anomaly.get("tier_counts") or {}
        total = sum(counts.values())
        self._tgt_anomaly = QRadioButton("Anomaly windows (from the built catalog)")
        self._tgt_anomaly.setEnabled(total > 0)
        tgt_vbox.addWidget(self._tgt_anomaly)

        tier_row = QHBoxLayout()
        tier_row.addSpacing(18)
        self._anom_tier_checks: dict[str, QCheckBox] = {}
        for tier, default_on in (("HIGH", True), ("MODERATE", False), ("SCREEN", False)):
            cb = QCheckBox(f"{tier} ({counts.get(tier, 0)})")
            cb.setChecked(default_on)
            cb.setEnabled(total > 0)
            self._anom_tier_checks[tier] = cb
            tier_row.addWidget(cb)
        tier_row.addStretch(1)
        tgt_vbox.addLayout(tier_row)

        self._anom_target_note = QLabel()
        self._anom_target_note.setWordWrap(True)
        self._anom_target_note.setStyleSheet("color: #888; font-size: 10px;")
        if total > 0:
            self._anom_target_note.setText(
                f"{total} catalogued windows. Selected tiers become a new job "
                "(each window padded by 120 s of review context), which the "
                "steps below then run on."
            )
        else:
            self._anom_target_note.setText(
                "No anomaly catalog found. Tick “Run anomaly detection” below to "
                "build one; it can then be used as a target on the next run."
            )
        tgt_vbox.addWidget(self._anom_target_note)

        def _sync_tiers(on: bool) -> None:
            for cb in self._anom_tier_checks.values():
                cb.setEnabled(on and total > 0)
        self._tgt_anomaly.toggled.connect(_sync_tiers)
        _sync_tiers(self._tgt_anomaly.isChecked())
        vbox.addWidget(tgt_group)

        # ── Prepare ───────────────────────────────────────────────────────
        prep_group = QGroupBox("Prepare")
        prep_form = QFormLayout(prep_group)
        self._build_interp_check = QCheckBox("Build interp_full.csv (nav + sensor time grid)")
        can_build = self._avail.get("nav") and self._avail.get("sensor")
        self._build_interp_check.setChecked(bool(can_build and not self._interp_exists))
        self._build_interp_check.setEnabled(bool(can_build))
        if not can_build:
            self._build_interp_check.setToolTip(
                "Needs navigation + sensor sources configured on the Inputs tab.")
        prep_form.addRow("", self._build_interp_check)
        self._sample_hz = QDoubleSpinBox()
        self._sample_hz.setRange(0.001, 100.0)
        self._sample_hz.setValue(1.0)
        self._sample_hz.setSuffix(" Hz")
        self._sample_hz.setDecimals(3)
        prep_form.addRow("Interp sample rate:", self._sample_hz)
        self._job_interp_check = QCheckBox(
            "Per-interval interp.csv set (one CSV per interval of each target job)")
        self._job_interp_check.setChecked(bool(self._jobs))
        prep_form.addRow("", self._job_interp_check)
        vbox.addWidget(prep_group)

        # ── Anomaly detection ─────────────────────────────────────────────
        anom_group = QGroupBox("Anomaly detection")
        anom_form = QFormLayout(anom_group)
        matlab_reason = self._anomaly.get("matlab_reason")

        self._anom_run_check = QCheckBox(
            "Run anomaly detection (report, spreadsheets, GeoJSON, QGIS bundle)")
        self._anom_run_check.setChecked(False)
        anom_form.addRow("", self._anom_run_check)

        self._anom_detector_check = QCheckBox(
            "…including the MATLAB detector (long: recomputes the anomaly matrix)")
        self._anom_detector_check.setChecked(not matlab_reason)
        self._anom_detector_check.setEnabled(not matlab_reason)
        if matlab_reason:
            self._anom_detector_check.setToolTip(matlab_reason)
        anom_form.addRow("", self._anom_detector_check)

        self._anom_catalog_check = QCheckBox(
            "…including the site catalog + PDF report")
        self._anom_catalog_check.setChecked(True)
        anom_form.addRow("", self._anom_catalog_check)

        anom_note = QLabel(
            "⚠ " + matlab_reason if matlab_reason else
            "The catalog stage alone is fast; it reuses the existing detector "
            "event CSVs. Newly detected windows become available as a TARGET on "
            "the next One-Click run."
        )
        anom_note.setWordWrap(True)
        anom_note.setStyleSheet(
            "color: %s; font-size: 10px;" % ("#b26a00" if matlab_reason else "#888")
        )
        anom_form.addRow("", anom_note)

        def _sync_anom(on: bool) -> None:
            self._anom_detector_check.setEnabled(on and not matlab_reason)
            self._anom_catalog_check.setEnabled(on)
        self._anom_run_check.toggled.connect(_sync_anom)
        _sync_anom(self._anom_run_check.isChecked())
        vbox.addWidget(anom_group)

        # ── Sampling ──────────────────────────────────────────────────────
        samp_group = QGroupBox("Sampling (frame extraction)")
        samp_form = QFormLayout(samp_group)
        self._sampling_check = QCheckBox("Extract frames from video")
        self._sampling_check.setChecked(bool(self._avail.get("video")))
        self._sampling_check.setEnabled(bool(self._avail.get("video")))
        if not self._avail.get("video"):
            self._sampling_check.setToolTip("No videos loaded (Inputs tab).")
        samp_form.addRow("", self._sampling_check)
        self._samp_mode = QComboBox()
        self._samp_mode.addItems(["Fixed rate", "Dynamic spacing"])
        samp_form.addRow("Mode:", self._samp_mode)
        self._samp_rate = QDoubleSpinBox()
        self._samp_rate.setRange(0.01, 60.0)
        self._samp_rate.setValue(1.0)
        self._samp_rate.setSuffix(" Hz")
        samp_form.addRow("Frame rate:", self._samp_rate)
        self._samp_spacing = QDoubleSpinBox()
        self._samp_spacing.setRange(0.05, 100.0)
        self._samp_spacing.setValue(1.0)
        self._samp_spacing.setSuffix(" m")
        samp_form.addRow("Target spacing:", self._samp_spacing)
        self._samp_quality = QComboBox()
        self._samp_quality.addItems(["high", "medium", "low"])
        samp_form.addRow("Frame quality:", self._samp_quality)
        self._samp_annotate = QCheckBox("Annotate frames")
        samp_form.addRow("", self._samp_annotate)
        self._samp_rasters = QCheckBox("Generate sensor rasters")
        samp_form.addRow("", self._samp_rasters)

        def _sync_mode() -> None:
            dyn = self._samp_mode.currentText() == "Dynamic spacing"
            self._samp_spacing.setEnabled(dyn)
            self._samp_rate.setEnabled(not dyn)
        self._samp_mode.currentIndexChanged.connect(lambda _i: _sync_mode())
        _sync_mode()
        vbox.addWidget(samp_group)

        # ── Data products ─────────────────────────────────────────────────
        prod_group = QGroupBox("Data products")
        prod_vbox = QVBoxLayout(prod_group)
        self._product_checks: dict[str, QCheckBox] = {}
        interp_ok = self._avail.get("interp", False)
        for task_type, label in _PRODUCT_TYPES:
            cb = QCheckBox(label)
            needs = TASK_INFO.get(task_type, {}).get("requires", [])
            ok = all(self._avail.get(r, False) for r in needs)
            cb.setEnabled(ok)
            if not ok:
                cb.setToolTip("Missing inputs: "
                              + ", ".join(r for r in needs if not self._avail.get(r)))
            self._product_checks[task_type] = cb
            prod_vbox.addWidget(cb)
        # Sensible defaults: the core products on, exports off.
        for tt in ("nav_3d", "sensor_2d"):
            if self._product_checks[tt].isEnabled():
                self._product_checks[tt].setChecked(True)

        prod_vbox.addWidget(QLabel("Sensor channels (none checked = all):"))
        self._channel_list = QListWidget()
        self._channel_list.setMaximumHeight(100)
        for ch in self._channels:
            item = QListWidgetItem(ch)
            item.setCheckState(Qt.Checked)
            self._channel_list.addItem(item)
        if not self._channels:
            note = QLabel("Channels appear after interp_full.csv exists; per-channel "
                          "products will cover ALL channels found at run time.")
            note.setStyleSheet("color: #888; font-size: 10px;")
            prod_vbox.addWidget(note)
        prod_vbox.addWidget(self._channel_list)
        if not interp_ok:
            warn = QLabel("interp_full.csv missing and can't be built — output "
                          "products are disabled until nav + sensor are configured.")
            warn.setStyleSheet("color: #c0392b; font-size: 10px;")
            prod_vbox.addWidget(warn)
        vbox.addWidget(prod_group)

        # ── Photogrammetry ────────────────────────────────────────────────
        photo_group = QGroupBox("Photogrammetry")
        photo_form = QFormLayout(photo_group)
        self._photo_check = QCheckBox("Run photogrammetry on the extracted frames")
        self._photo_check.setEnabled(bool(self._avail.get("video")))
        photo_form.addRow("", self._photo_check)
        self._photo_engine = QComboBox()
        self._photo_engine.addItems(["Metashape", "COLMAP"])
        photo_form.addRow("Engine:", self._photo_engine)
        self._photo_summary = QLabel(self._photo_summary_text())
        self._photo_summary.setStyleSheet("color: #888; font-size: 10px;")
        self._photo_summary.setWordWrap(True)
        photo_form.addRow("", self._photo_summary)
        cfg_btn = QPushButton("Configure engine settings…")
        cfg_btn.clicked.connect(self._configure_photogrammetry)
        photo_form.addRow("", cfg_btn)
        vbox.addWidget(photo_group)
        vbox.addStretch()

        # ── Stack handling + buttons ──────────────────────────────────────
        self._replace_check = QCheckBox("Replace the current task stack (unchecked = append)")
        self._replace_check.setChecked(True)
        outer.addWidget(self._replace_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("▶  Generate && Run")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    # ------------------------------------------------------------------

    def _photo_summary_text(self) -> str:
        s = self._photo_proto.settings
        engine = s.get("engine", "Metashape")
        if engine == "COLMAP":
            bits = [f"matcher {s.get('matcher', 'Exhaustive')}"]
            if s.get("run_mvs", True):
                bits.append("dense MVS")
        else:
            bits = [f"align {s.get('align_accuracy', 'High')}",
                    f"chunks ≤{s.get('chunk_size', 250)} images"]
            if s.get("build_dense", True):
                bits.append(f"dense {s.get('dense_quality', 'Medium')}")
            if s.get("build_mesh", False):
                bits.append("mesh")
            if s.get("build_texture", False):
                bits.append("texture")
        return f"{engine}: " + ", ".join(bits) + ".  All settings editable via Configure."

    def _configure_photogrammetry(self) -> None:
        """Open the full TaskConfigDialog on the prototype photogrammetry task.

        The frame source and target chosen there are overridden at generation
        time (frames come from the generated sampling task; the target is this
        dialog's target) — only the engine settings are kept.
        """
        self._photo_proto.settings["engine"] = self._photo_engine.currentText()
        dlg = TaskConfigDialog(
            self._photo_proto,
            self._jobs,
            self._channels,
            [(0, "Sampling task (generated by One-Click)")],
            self,
        )
        if dlg.exec():
            engine = self._photo_proto.settings.get("engine", "Metashape")
            self._photo_engine.setCurrentText(engine)
            self._photo_summary.setText(self._photo_summary_text())

    # ------------------------------------------------------------------

    def accept(self) -> None:
        """Validate the combination before closing."""
        if self._tgt_jobs.isChecked() and not self._checked_jobs():
            QMessageBox.warning(self, "No job selected",
                                "Check at least one job, or switch to Full dataset.")
            return
        if self._photo_check.isChecked() and not self._sampling_check.isChecked():
            manual_dir = (self._photo_proto.settings.get("frame_dir") or "").strip()
            if not manual_dir:
                QMessageBox.warning(
                    self, "Photogrammetry needs frames",
                    "Photogrammetry is enabled but Sampling is off and no manual "
                    "frame directory is configured.  Enable Sampling, or set a "
                    "manual frame directory via 'Configure engine settings…'."
                )
                return
        if self._tgt_anomaly.isChecked() and not self.anomaly_tiers():
            QMessageBox.warning(
                self, "No tiers selected",
                "Targeting anomaly windows needs at least one confidence tier "
                "(HIGH / MODERATE / SCREEN)."
            )
            return
        if (self._anom_run_check.isChecked()
                and not (self._anom_detector_check.isChecked()
                         or self._anom_catalog_check.isChecked())):
            QMessageBox.warning(
                self, "Anomaly step does nothing",
                "Enable the MATLAB detector stage, the catalog stage, or both — "
                "or untick “Run anomaly detection”."
            )
            return
        anything = (self._build_interp_check.isChecked()
                    or self._job_interp_check.isChecked()
                    or self._sampling_check.isChecked()
                    or self._photo_check.isChecked()
                    or self._anom_run_check.isChecked()
                    or any(cb.isChecked() for cb in self._product_checks.values()))
        if not anything:
            QMessageBox.warning(self, "Nothing selected",
                                "Tick at least one product or step.")
            return
        super().accept()

    def _checked_jobs(self) -> list[dict]:
        jobs = []
        for i in range(self._job_list.count()):
            item = self._job_list.item(i)
            if item.checkState() == Qt.Checked:
                jobs.append({"job_id": item.data(Qt.UserRole),
                             "name":   item.data(Qt.UserRole + 1)})
        return jobs

    def _target(self) -> dict:
        if self._tgt_jobs.isChecked():
            jobs = self._checked_jobs()
            if jobs:
                return {"kind": "jobs", "jobs": jobs}
        return {"kind": "full"}

    # ---- anomaly accessors (read by MainWindow before build_tasks) --------

    def wants_anomaly_target(self) -> bool:
        """True when the generated tasks should run on the anomaly windows."""
        return self._tgt_anomaly.isChecked()

    def anomaly_tiers(self) -> list[str]:
        """Confidence tiers to turn into the anomaly job."""
        return [t for t, cb in self._anom_tier_checks.items() if cb.isChecked()]

    def wants_anomaly_step(self) -> bool:
        return self._anom_run_check.isChecked()

    def _selected_channels(self) -> list[str]:
        checked = [
            self._channel_list.item(i).text()
            for i in range(self._channel_list.count())
            if self._channel_list.item(i).checkState() == Qt.Checked
        ]
        # All checked == no restriction (empty list means "all" downstream).
        return [] if len(checked) == len(self._channels) else checked

    def replace_stack(self) -> bool:
        return self._replace_check.isChecked()

    # ------------------------------------------------------------------

    def build_tasks(self, stack: TaskStack,
                    anomaly_job: Optional[dict] = None) -> list[Task]:
        """Generate the wired Task list (ids allocated from `stack`).

        Args:
            stack: task stack used to allocate ids.
            anomaly_job: {"job_id", "name"} of the job MainWindow created from
                the anomaly windows.  Supplied when the user picked the
                "Anomaly windows" target; every generated task then runs on it.
        """
        if anomaly_job:
            target = {"kind": "jobs", "jobs": [dict(anomaly_job)]}
        else:
            target = self._target()
        channels = self._selected_channels()
        tasks: list[Task] = []

        if self._build_interp_check.isChecked():
            tasks.append(Task(
                task_id=stack.new_id(), task_type="build_interp",
                target={"kind": "full"},
                settings={"sample_hz": self._sample_hz.value()},
            ))

        # Anomaly detection runs early: it needs interp_full.csv and its outputs
        # (report / spreadsheets / QGIS) are independent of the product steps.
        if self._anom_run_check.isChecked():
            tasks.append(Task(
                task_id=stack.new_id(), task_type="anomaly_detect",
                target={"kind": "full"},          # workspace-level, never per-job
                settings={
                    "run_detector": self._anom_detector_check.isChecked(),
                    "run_catalog":  self._anom_catalog_check.isChecked(),
                },
            ))

        if self._job_interp_check.isChecked() and target["kind"] == "jobs":
            tasks.append(Task(
                task_id=stack.new_id(), task_type="job_interp",
                target=dict(target),
                settings={"annotate_video": True},
            ))

        sampling_task: Task | None = None
        if self._sampling_check.isChecked():
            mode = ("dynamic" if self._samp_mode.currentText() == "Dynamic spacing"
                    else "fixed")
            sampling_task = Task(
                task_id=stack.new_id(), task_type="sampling",
                target=dict(target),
                settings={
                    "mode":       mode,
                    "frame_rate": self._samp_rate.value(),
                    "spacing_m":  self._samp_spacing.value(),
                    "quality":    self._samp_quality.currentText(),
                    "annotate":   self._samp_annotate.isChecked(),
                    "rasters":    self._samp_rasters.isChecked(),
                },
            )
            tasks.append(sampling_task)

        for task_type, _label in _PRODUCT_TYPES:
            cb = self._product_checks[task_type]
            if not (cb.isEnabled() and cb.isChecked()):
                continue
            per_channel = TASK_INFO.get(task_type, {}).get("per_channel", False)
            tasks.append(Task(
                task_id=stack.new_id(), task_type=task_type,
                target=dict(target),
                settings={},                       # type defaults apply at plan time
                channels=list(channels) if per_channel else [],
            ))

        if self._photo_check.isChecked():
            settings = dict(self._photo_proto.settings)
            settings["engine"] = self._photo_engine.currentText()
            photo = Task(
                task_id=stack.new_id(), task_type="photogrammetry",
                target=dict(target),
                settings=settings,
            )
            if sampling_task is not None:
                photo.depends_on = sampling_task.task_id
                settings["frame_dir"] = ""
            tasks.append(photo)

        return tasks
