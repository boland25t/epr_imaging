"""
stack_panel.py — Task-stack dock widget (bottom-left).

Shows the ordered list of Task instances the user has created.  Provides:
  • Create Task  → a grouped menu of task types (greyed when data missing)
  • Edit / double-click → reopen the task's config dialog
  • Remove, Move Up, Move Down → manage the queue
  • Run Stack → emit run_requested()

The panel owns the create/edit dialogs but pulls live context (available jobs,
sensor channels, and data-availability flags) from provider callables injected
by MainWindow, so it stays decoupled from the rest of the app.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import preset_service
from models import Task, TaskStack, TASK_INFO, TASK_CATEGORIES, ARCHIVED_TASK_TYPES
from task_config_dialog import TaskConfigDialog


class StackPanel(QWidget):
    """Dockable task-stack editor + runner."""

    run_requested          = Signal()
    rerun_failed_requested = Signal()
    tasks_changed          = Signal()   # emitted whenever the stack is mutated
    one_click_requested    = Signal()   # open the One-Click Pipeline builder

    def __init__(self, stack: TaskStack, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._stack = stack
        # Provider callables, injected via set_providers().
        self._jobs_provider:    Callable[[], list[tuple]] = lambda: []
        self._channels_provider: Callable[[], list[str]]  = lambda: []
        self._availability_provider: Callable[[], dict]   = lambda: {}
        self._build_ui()
        self.refresh()

    def set_providers(
        self,
        jobs_provider: Callable[[], list[tuple]],
        channels_provider: Callable[[], list[str]],
        availability_provider: Callable[[], dict],
    ) -> None:
        self._jobs_provider = jobs_provider
        self._channels_provider = channels_provider
        self._availability_provider = availability_provider

    def set_running(self, running: bool) -> None:
        """Disable editing controls while a run is in progress."""
        self._run_btn.setEnabled(not running)
        self._create_btn.setEnabled(not running)
        self._tmpl_btn.setEnabled(not running)
        self._rerun_btn.setEnabled(not running)
        self._one_click_btn.setEnabled(not running)
        self._run_btn.setText("Running…" if running else "▶  Run Stack")

    def skip_existing(self) -> bool:
        """Whether to skip steps whose deterministic output already exists."""
        return self._skip_check.isChecked()

    def set_failed_count(self, n: int) -> None:
        """Show/hide the Re-run Failed button after a run, with the failure count."""
        self._rerun_btn.setVisible(n > 0)
        self._rerun_btn.setText(f"↻  Re-run Failed ({n})" if n else "↻  Re-run Failed")

    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        header = QLabel("Task Stack")
        f = QFont(); f.setBold(True)
        header.setFont(f)
        layout.addWidget(header)

        hint = QLabel("Tasks run top-to-bottom. Double-click to edit.")
        hint.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(hint)

        # One-Click Pipeline: generates a fully wired stack from one dialog
        # (target + products) and runs it immediately.
        self._one_click_btn = QPushButton("⚡ One-Click Pipeline…")
        self._one_click_btn.setStyleSheet("font-weight: bold; padding: 5px;")
        self._one_click_btn.setToolTip(
            "Pick a target job and the products you want (interp CSVs, point\n"
            "clouds, rasters, NetCDF, QC, Metashape/COLMAP) in one dialog —\n"
            "the tasks are generated, wired together, and run immediately."
        )
        self._one_click_btn.clicked.connect(self.one_click_requested)
        layout.addWidget(self._one_click_btn)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.itemDoubleClicked.connect(lambda _: self._edit_selected())
        layout.addWidget(self._list, stretch=1)

        # Row of edit controls
        edit_row = QHBoxLayout()
        self._create_btn = QPushButton("＋ Create Task")
        self._create_btn.setStyleSheet("font-weight: bold;")
        self._create_btn.clicked.connect(self._show_create_menu)
        edit_row.addWidget(self._create_btn)

        self._edit_btn = QPushButton("Edit")
        self._edit_btn.clicked.connect(self._edit_selected)
        edit_row.addWidget(self._edit_btn)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_selected)
        edit_row.addWidget(self._remove_btn)
        layout.addLayout(edit_row)

        move_row = QHBoxLayout()
        self._up_btn = QPushButton("▲ Up")
        self._up_btn.clicked.connect(lambda: self._move_selected(-1))
        move_row.addWidget(self._up_btn)
        self._down_btn = QPushButton("▼ Down")
        self._down_btn.clicked.connect(lambda: self._move_selected(1))
        move_row.addWidget(self._down_btn)
        self._dup_btn = QPushButton("Duplicate")
        self._dup_btn.clicked.connect(self._duplicate_selected)
        move_row.addWidget(self._dup_btn)
        layout.addLayout(move_row)

        # Templates: save/reuse tasks and whole stacks across surveys.
        self._tmpl_btn = QPushButton("⧉ Templates ▾")
        self._tmpl_btn.clicked.connect(self._show_templates_menu)
        layout.addWidget(self._tmpl_btn)

        # Failure recovery: skip steps whose deterministic output already exists.
        self._skip_check = QCheckBox("Skip steps already done (interp / frames)")
        self._skip_check.setToolTip(
            "When re-running a stack, skip Build interp_full.csv and Sampling steps "
            "whose output already exists on disk, so only the missing work runs.\n"
            "Versioned products (3D/2D/photogrammetry) always re-run."
        )
        layout.addWidget(self._skip_check)

        self._run_btn = QPushButton("▶  Run Stack")
        self._run_btn.setStyleSheet("font-weight: bold; padding: 6px; font-size: 13px;")
        self._run_btn.clicked.connect(self.run_requested)
        layout.addWidget(self._run_btn)

        # Re-run failed — hidden until a run finishes with failures.
        self._rerun_btn = QPushButton("↻  Re-run Failed")
        self._rerun_btn.setStyleSheet("font-weight: bold; padding: 4px; color: #b25000;")
        self._rerun_btn.clicked.connect(self.rerun_failed_requested)
        self._rerun_btn.setVisible(False)
        layout.addWidget(self._rerun_btn)

    # -----------------------------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the task list from the model, preserving selection by task_id."""
        sel_id = self._selected_task_id()
        self._list.clear()
        for task in self._stack.tasks:
            item = QListWidgetItem(task.display_label())
            item.setData(Qt.UserRole, task.task_id)
            item.setToolTip(self._task_tooltip(task))
            self._list.addItem(item)
            if task.task_id == sel_id:
                self._list.setCurrentItem(item)

    def _task_tooltip(self, task: Task) -> str:
        lines = [f"Type: {task.type_label}", f"Target: {task.target_label()}"]
        if task.per_channel:
            lines.append("Channels: " + (", ".join(task.channels) if task.channels else "all"))
        for k, v in task.settings.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def _selected_task_id(self) -> Optional[int]:
        item = self._list.currentItem()
        return item.data(Qt.UserRole) if item else None

    # -----------------------------------------------------------------------
    # Create / edit / remove / move
    # -----------------------------------------------------------------------

    def _show_create_menu(self) -> None:
        avail = self._availability_provider() or {}
        menu = QMenu(self)
        for category in TASK_CATEGORIES:
            types = [tt for tt, info in TASK_INFO.items()
                     if info.get("category") == category
                     and tt not in ARCHIVED_TASK_TYPES]   # hide archived (point-cloud) types
            if not types:
                continue
            menu.addSection(category)
            for tt in types:
                info = TASK_INFO[tt]
                missing = [r for r in info.get("requires", []) if not avail.get(r, False)]
                action = QAction(info["label"], self)
                if missing:
                    action.setEnabled(False)
                    action.setText(f"{info['label']}   (needs {', '.join(missing)})")
                action.triggered.connect(lambda _checked=False, t=tt: self._create_task(t))
                menu.addAction(action)
        menu.exec(self._create_btn.mapToGlobal(self._create_btn.rect().bottomLeft()))

    def _sampling_tasks_for_dialog(self, exclude_task_id: Optional[int] = None) -> list[tuple]:
        """Return [(task_id, display_label)] for all sampling tasks in the stack.

        Excludes exclude_task_id so a task can't depend on itself.
        """
        return [
            (t.task_id, t.display_label())
            for t in self._stack.tasks
            if t.task_type == "sampling" and t.task_id != exclude_task_id
        ]

    def _create_task(self, task_type: str) -> None:
        task = Task(task_id=self._stack.new_id(), task_type=task_type)
        if task.per_channel:
            task.channels = list(self._channels_provider() or [])
        sampling_tasks = self._sampling_tasks_for_dialog()
        dlg = TaskConfigDialog(
            task,
            self._jobs_provider() or [],
            self._channels_provider() or [],
            sampling_tasks,
            self,
        )
        if dlg.exec():
            self._stack.add(dlg.task())
            self.refresh()
            self.tasks_changed.emit()

    def _edit_selected(self) -> None:
        tid = self._selected_task_id()
        if tid is None:
            return
        task = self._stack.get(tid)
        if task is None:
            return
        sampling_tasks = self._sampling_tasks_for_dialog(exclude_task_id=tid)
        dlg = TaskConfigDialog(
            task,
            self._jobs_provider() or [],
            self._channels_provider() or [],
            sampling_tasks,
            self,
        )
        if dlg.exec():
            self.refresh()
            self.tasks_changed.emit()

    def _remove_selected(self) -> None:
        tid = self._selected_task_id()
        if tid is None:
            return
        self._stack.remove(tid)
        self.refresh()
        self.tasks_changed.emit()

    def _move_selected(self, delta: int) -> None:
        tid = self._selected_task_id()
        if tid is None:
            return
        self._stack.move(tid, delta)
        self.refresh()
        self.tasks_changed.emit()

    def _duplicate_selected(self) -> None:
        tid = self._selected_task_id()
        if tid is None:
            return
        src = self._stack.get(tid)
        if src is None:
            return
        clone = Task(
            task_id=self._stack.new_id(),
            task_type=src.task_type,
            target=dict(src.target),
            settings=dict(src.settings),
            channels=list(src.channels),
            depends_on=src.depends_on,
        )
        self._stack.add(clone)
        self.refresh()
        self.tasks_changed.emit()

    # -----------------------------------------------------------------------
    # Templates (presets.json — reusable across surveys)
    # -----------------------------------------------------------------------

    def _show_templates_menu(self) -> None:
        menu = QMenu(self)

        save_task = menu.addAction("Save selected task as template…")
        save_task.setEnabled(self._selected_task_id() is not None)
        save_task.triggered.connect(self._save_task_template)

        save_stack = menu.addAction("Save stack as template…")
        save_stack.setEnabled(bool(self._stack.tasks))
        save_stack.triggered.connect(self._save_stack_template)

        menu.addSeparator()

        # Insert task from template
        task_tmpls = preset_service.list_task_templates()
        insert_menu = menu.addMenu("Insert task from template")
        insert_menu.setEnabled(bool(task_tmpls))
        for tmpl in task_tmpls:
            label = f"{tmpl.get('name')}  ({TASK_INFO.get(tmpl.get('task_type'), {}).get('label', tmpl.get('task_type'))})"
            act = insert_menu.addAction(label)
            act.triggered.connect(lambda _c=False, t=tmpl: self._insert_task_from_template(t))

        # Load stack from template
        stack_tmpls = preset_service.list_stack_templates()
        load_menu = menu.addMenu("Load stack from template")
        load_menu.setEnabled(bool(stack_tmpls))
        for tmpl in stack_tmpls:
            act = load_menu.addAction(f"{tmpl.get('name')}  ({len(tmpl.get('tasks', []))} tasks)")
            act.triggered.connect(lambda _c=False, t=tmpl: self._load_stack_from_template(t))

        menu.addSeparator()
        manage = menu.addAction("Manage templates…")
        manage.setEnabled(bool(task_tmpls or stack_tmpls))
        manage.triggered.connect(self._manage_templates)

        menu.exec(self._tmpl_btn.mapToGlobal(self._tmpl_btn.rect().bottomLeft()))

    def _save_task_template(self) -> None:
        tid = self._selected_task_id()
        if tid is None:
            return
        task = self._stack.get(tid)
        if task is None:
            return
        name, ok = QInputDialog.getText(
            self, "Save Task Template", "Template name:",
            text=task.type_label,
        )
        name = (name or "").strip()
        if not ok or not name:
            return
        if self._name_exists(preset_service.list_task_templates(), name) and \
           not self._confirm_overwrite(name):
            return
        preset_service.add_task_template(name, task)

    def _save_stack_template(self) -> None:
        if not self._stack.tasks:
            return
        name, ok = QInputDialog.getText(self, "Save Stack Template", "Template name:")
        name = (name or "").strip()
        if not ok or not name:
            return
        if self._name_exists(preset_service.list_stack_templates(), name) and \
           not self._confirm_overwrite(name):
            return
        preset_service.add_stack_template(name, self._stack.tasks)

    def _insert_task_from_template(self, tmpl: dict) -> None:
        task = preset_service.template_to_task(tmpl, self._stack.new_id())
        self._stack.add(task)
        self.refresh()
        self.tasks_changed.emit()

    def _load_stack_from_template(self, tmpl: dict) -> None:
        bodies = tmpl.get("tasks", [])
        if not bodies:
            return
        replace = True
        if self._stack.tasks:
            box = QMessageBox(self)
            box.setWindowTitle("Load Stack Template")
            box.setText(f"Load template '{tmpl.get('name')}' ({len(bodies)} tasks)?")
            box.setInformativeText("Replace the current stack, or append to it?")
            replace_btn = box.addButton("Replace", QMessageBox.AcceptRole)
            append_btn  = box.addButton("Append", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is None or clicked not in (replace_btn, append_btn):
                return
            replace = clicked is replace_btn

        if replace:
            self._stack.tasks = []
        # Instantiate tasks, then remap positional depends_on_index → new task_id
        # so intra-stack links (photogrammetry → sampling) survive the load.
        new_tasks = [preset_service.template_to_task(b, self._stack.new_id()) for b in bodies]
        for body, task in zip(bodies, new_tasks):
            di = body.get("depends_on_index")
            if di is not None and 0 <= di < len(new_tasks):
                task.depends_on = new_tasks[di].task_id
        for task in new_tasks:
            self._stack.add(task)
        self.refresh()
        self.tasks_changed.emit()

    def _manage_templates(self) -> None:
        ManageTemplatesDialog(self).exec()

    @staticmethod
    def _name_exists(items: list[dict], name: str) -> bool:
        return any(i.get("name") == name for i in items)

    def _confirm_overwrite(self, name: str) -> bool:
        return QMessageBox.question(
            self, "Overwrite Template",
            f"A template named '{name}' already exists. Overwrite it?",
        ) == QMessageBox.Yes


# ---------------------------------------------------------------------------
# Manage Templates dialog — delete saved task / stack templates
# ---------------------------------------------------------------------------

class ManageTemplatesDialog(QDialog):
    """Lists saved task and stack templates with a Delete action for each."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage Templates")
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Task templates"))
        self._task_list = QListWidget()
        layout.addWidget(self._task_list)
        layout.addWidget(QLabel("Stack templates"))
        self._stack_list = QListWidget()
        layout.addWidget(self._stack_list)

        btn_row = QHBoxLayout()
        del_task = QPushButton("Delete selected task template")
        del_task.clicked.connect(self._delete_task)
        btn_row.addWidget(del_task)
        del_stack = QPushButton("Delete selected stack template")
        del_stack.clicked.connect(self._delete_stack)
        btn_row.addWidget(del_stack)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._reload()

    def _reload(self) -> None:
        self._task_list.clear()
        for t in preset_service.list_task_templates():
            self._task_list.addItem(f"{t.get('name')}  ({t.get('task_type')})")
        self._stack_list.clear()
        for s in preset_service.list_stack_templates():
            self._stack_list.addItem(f"{s.get('name')}  ({len(s.get('tasks', []))} tasks)")

    def _delete_task(self) -> None:
        names = preset_service.list_task_templates()
        row = self._task_list.currentRow()
        if 0 <= row < len(names):
            preset_service.delete_task_template(names[row].get("name"))
            self._reload()

    def _delete_stack(self) -> None:
        names = preset_service.list_stack_templates()
        row = self._stack_list.currentRow()
        if 0 <= row < len(names):
            preset_service.delete_stack_template(names[row].get("name"))
            self._reload()
