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
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from models import Task, TaskStack, TASK_INFO, TASK_CATEGORIES
from task_config_dialog import TaskConfigDialog


class StackPanel(QWidget):
    """Dockable task-stack editor + runner."""

    run_requested = Signal()
    tasks_changed = Signal()   # emitted whenever the stack is mutated

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
        self._run_btn.setText("Running…" if running else "▶  Run Stack")

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

        self._run_btn = QPushButton("▶  Run Stack")
        self._run_btn.setStyleSheet("font-weight: bold; padding: 6px; font-size: 13px;")
        self._run_btn.clicked.connect(self.run_requested)
        layout.addWidget(self._run_btn)

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
            types = [tt for tt, info in TASK_INFO.items() if info.get("category") == category]
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
