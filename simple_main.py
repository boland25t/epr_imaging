#!/usr/bin/env python3
"""simple_main.py — the simplified three-panel EPR Imaging UI.

    python3 simple_main.py [/path/to/<name>.eprproj]

Layout (QSplitter, horizontal):

    [ Products ~300px ] [ Trackline (stretch) ] [ Logger ~330px ]

Everything domain-specific lives behind ``product_catalog`` (see
docs/simple_ui_contract.md); this module owns only Qt.  The legacy
main_window.py is never imported.  Every single product_catalog call is wrapped
in try/except so the window still constructs (with an error line in the logger)
when the backend is missing, half-built or raising.

Workspace read/write goes through config_service.ConfigService: the loaded dict
is kept verbatim in ``self.ws`` and handed straight back to save_workspace(),
so keys this UI does not understand (task_stack, photo_*, simple_jobs, …)
round-trip untouched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import (QAbstractTableModel, QDateTime, QModelIndex, QObject,
                            Qt, QThread, QTimeZone, QUrl, Signal)
from PySide6.QtGui import QAction, QDesktopServices, QFont, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
                               QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout, QHBoxLayout, QHeaderView,
                               QInputDialog, QLabel, QLineEdit,
                               QMainWindow,
                               QMenu, QMessageBox, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QSplitter, QTableView, QTableWidget,
                               QTableWidgetItem, QTabWidget,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg          # noqa: E402
from matplotlib.figure import Figure                                     # noqa: E402
import numpy as np                                                       # noqa: E402

from config_service import ConfigService                                 # noqa: E402
from models import (NavigationConfig, SensorChannel, SensorFileConfig,    # noqa: E402
                    TimeValueSourceConfig)

# The backend lands in parallel — import it normally but never let its absence
# (or an exception inside it) stop the UI from coming up.
try:
    import product_catalog as pc
    PC_IMPORT_ERROR: str | None = None
except Exception as _exc:                                    # pragma: no cover
    pc = None                                                # type: ignore[assignment]
    PC_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

# --- palette (matches the pipeline's figure styling) -------------------------
BG, INK, MUT, GRIDC = "#0e1620", "#dbe4ec", "#9fb0bd", "#2a3846"
C_JOB, C_PEND, C_STAGED = "#f2a63b", "#4fd0ff", "#7ee787"
C_MARK = "#ff4fa3"                      # click markers (bright accent)
C_ERR, C_WARN, C_DIM = "#ff6b6b", "#ffc861", "#6f8291"

# Worker log-line conventions, mirrored from product_catalog (kept as literals
# so the UI still classifies correctly when the backend is unavailable).
FAIL_PREFIX = getattr(pc, "LOG_FAIL", "!! ") if pc else "!! "
DETAIL_PREFIX = getattr(pc, "LOG_DETAIL", "  · ") if pc else "  · "
WARN_PREFIX = getattr(pc, "LOG_WARN", "note: ") if pc else "note: "
CLICK_TOLERANCE_PX = 30.0               # a click further than this misses the track

ROLE = Qt.ItemDataRole.UserRole
SETTINGS_FILE = Path.home() / ".epr_simple_ui.json"
GEN_LABEL = "＋ Generate New…"
WHOLE = getattr(pc, "WHOLE_TRACKLINE", "__whole__") if pc else "__whole__"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _head_tail(path: str | Path) -> tuple[str, str]:
    """First and last non-blank lines of a text file (tail read from the end)."""
    first = ""
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if line.strip():
                first = line.strip()
                break
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - 65536))
        blk = fh.read().decode("utf-8", "replace").splitlines()
    tail = next((ln.strip() for ln in reversed(blk) if ln.strip()), first)
    return first, tail


def _parse_dt(date_tok: str, clock_tok: str) -> datetime | None:
    """Parse "M/D/YY"|"YYYY/MM/DD"|"YYYY-MM-DD" + "HH:MM:SS[.sss]"."""
    try:
        parts = [p for p in date_tok.strip().replace("-", "/").split("/") if p]
        if len(parts[0]) == 4:
            y, mo, dy = parts[0], parts[1], parts[2]
        else:
            mo, dy, y = parts[0], parts[1], parts[2]
        yi = int(y)
        yi += 2000 if yi < 100 else 0
        bits = (clock_tok.strip().split(":") + ["0", "0"])[:3]
        sec = float(bits[2] or 0)
        us = min(999999, int(round((sec % 1) * 1e6)))
        return datetime(yi, int(mo), int(dy), int(bits[0]), int(bits[1]), int(sec), us)
    except Exception:
        return None


def _csv_bounds(path: str | Path, date_i, clock_i) -> tuple[datetime | None, datetime | None]:
    """Time bounds of a headerless CSV from its first/last row (column indices)."""
    try:
        head, tail = _head_tail(path)
        hf, tf = head.split(","), tail.split(",")
        return (_parse_dt(hf[date_i], hf[clock_i]), _parse_dt(tf[date_i], tf[clock_i]))
    except Exception:
        return None, None


def _headered_bounds(path: str | Path, date_col: str,
                     time_col: str) -> tuple[datetime | None, datetime | None]:
    """Time bounds of a headered CSV using two named columns."""
    try:
        with open(path, "r", errors="replace") as fh:
            header = [c.strip() for c in fh.readline().split(",")]
        di, ti = header.index(date_col), header.index(time_col)
        with open(path, "r", errors="replace") as fh:
            fh.readline()
            rest = fh.readline()
        first = rest.strip()
        _, tail = _head_tail(path)
        ff, tf = first.split(","), tail.split(",")
        return (_parse_dt(ff[di], ff[ti]), _parse_dt(tf[di], tf[ti]))
    except Exception:
        return None, None


def _open_folder(path: str) -> bool:
    """Best-effort reveal of a file's directory (Linux, then WSL → Explorer)."""
    p = Path(path)
    d = str(p.parent if p.suffix else p)
    try:
        subprocess.Popen(["xdg-open", d], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return True
    except Exception:
        pass
    try:
        win = subprocess.run(["wslpath", "-w", d], capture_output=True,
                             text=True, timeout=5).stdout.strip() or d
        subprocess.Popen(["explorer.exe", win], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 3-D: bounded readers for point clouds and meshes
# ---------------------------------------------------------------------------
#
# The legacy app's 3-D viewer is viewer_widget.PointCloudViewer (PyVista/VTK).
# It is used when it can be: ``_pyvista_viewer`` below spawns exactly that
# window.  But PyVista is NOT installed on every machine (it is missing here),
# and a chunk's dense.ply is ~2 GB / 48 M vertices, which a WSL VM that OOMs
# must never load whole.  So the in-tab preview reads a BOUNDED sample instead:
# a memory-mapped PLY is sampled in 32 contiguous blocks spread through the
# file, which costs ~5 MB of I/O and ~0.2 s for that same 2 GB cloud while
# still spanning its whole bounding box.

MESH_EXTS = (".ply", ".obj", ".stl", ".glb", ".gltf")
MESH_SAMPLE = 120_000              # points drawn in the preview
_OBJ_MAX_BYTES = 96_000_000        # ASCII OBJ: how much text to walk at most

_PLY_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _ply_header(path: str) -> tuple[str, dict, list, int]:
    """(format, {element: count}, [(name, dtype)…] for vertex, data offset)."""
    fields: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    fmt, order, element, offset = "", "<", None, 0
    with open(path, "rb") as fh:
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError("not a PLY file (no end_header)")
            offset += len(raw)
            tok = raw.decode("ascii", "replace").split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
                order = ">" if "big" in fmt else "<"
            elif tok[0] == "element":
                element = tok[1]
                counts[element] = int(tok[2])
            elif tok[0] == "property" and element == "vertex":
                if tok[1] == "list":
                    raise ValueError("unsupported list property on vertex")
                fields.append((tok[-1], order + _PLY_DTYPES[tok[1]]))
            elif tok[0] == "end_header":
                return fmt, counts, fields, offset


def _sample_ply(path: str, cap: int = MESH_SAMPLE):
    """(xyz, rgb|None, n_vertices, n_faces) from a bounded read of a PLY."""
    fmt, counts, fields, offset = _ply_header(path)
    n = int(counts.get("vertex", 0))
    names = [f[0] for f in fields]
    if not n or not all(c in names for c in ("x", "y", "z")):
        raise ValueError("PLY has no xyz vertex data")
    if "binary" not in fmt:                       # ASCII: read line by line
        ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
        step, rows = max(1, n // cap), []
        with open(path, "r", errors="replace") as fh:
            fh.seek(offset)
            for i, line in enumerate(fh):
                if i >= n or len(rows) >= cap:
                    break
                if i % step:
                    continue
                part = line.split()
                if len(part) > max(ix, iy, iz):
                    rows.append((float(part[ix]), float(part[iy]), float(part[iz])))
        return np.asarray(rows, dtype="f8"), None, n, int(counts.get("face", 0))

    dt = np.dtype(fields)
    mm = np.memmap(path, mode="r", dtype=dt, offset=offset, shape=(n,))
    try:
        if n * dt.itemsize <= 64_000_000:
            sub = np.asarray(mm[::max(1, n // cap)][:cap])
        else:
            # Contiguous blocks spread end to end: bounded I/O, full extent.
            # Metashape writes vertices in depth-map order, so one block is one
            # small patch of seafloor — many small blocks read as a cloud,
            # a few big ones read as a handful of blobs.
            blocks = 256
            per = max(1, cap // blocks)
            sub = np.concatenate([
                np.asarray(mm[s:s + per]) for s in
                (min(n - per, int(b * (n - per) / (blocks - 1))) for b in range(blocks))])
    finally:
        del mm
    xyz = np.stack([sub["x"].astype("f8"), sub["y"].astype("f8"),
                    sub["z"].astype("f8")], axis=1)
    rgb = None
    if all(c in dt.names for c in ("red", "green", "blue")):
        rgb = np.stack([sub["red"], sub["green"], sub["blue"]],
                       axis=1).astype("f4") / 255.0
    return xyz, rgb, n, int(counts.get("face", 0))


def _sample_obj(path: str, cap: int = MESH_SAMPLE):
    """(xyz, None, n_vertices_seen, n_faces_seen) from a bounded OBJ walk.

    Wavefront OBJ is ASCII and Metashape writes vertices first, so the walk
    stops at the face section (or at a byte budget) rather than reading a
    300 MB file to its end.
    """
    lines: list[str] = []
    read = n_faces = 0
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            read += len(line)
            if line[:2] == "v ":
                lines.append(line)
            elif line[:2] == "f ":
                n_faces += 1
                if lines:
                    break
            if read > _OBJ_MAX_BYTES:
                break
    step = max(1, len(lines) // cap)
    pts = []
    for line in lines[::step][:cap]:
        part = line.split()
        if len(part) >= 4:
            try:
                pts.append((float(part[1]), float(part[2]), float(part[3])))
            except ValueError:
                pass
    return np.asarray(pts, dtype="f8"), None, len(lines), n_faces


def _sampling_dependent(type_key: str) -> bool:
    """Does a second sampling regime multiply this product type?

    Frame-consuming types (frames, photogrammetry, fauna) get one instance per
    sampling run; everything else is computed from the 1 Hz table and is listed
    once per job.  The backend owns the rule; this mirrors it safely.
    """
    if pc is None:
        return False
    try:
        return bool(pc.is_sampling_dependent(type_key))
    except Exception:
        return str(type_key) in ("frame_set", "photogrammetry", "fauna_detection")


def _catalog_builder():
    """``catalog_builder.build_catalog`` when the module exists, else None.

    The module is authored in parallel to this one, so every use is guarded:
    the app must come up — and the demo must run — whether or not it landed.
    Contract: ``build_catalog(workspace_dir, job_id="__whole__", log=print)``
    returns the path of a PDF under ``survey/catalog/``.
    """
    try:
        import catalog_builder
    except Exception:
        return None
    fn = getattr(catalog_builder, "build_catalog", None)
    return fn if callable(fn) else None


def _pyvista_viewer(path: str, parent=None):
    """Open the legacy PyVista/VTK viewer on a mesh, or return None.

    ``viewer_widget.PointCloudViewer`` is the repo's only real 3-D viewer; it
    is reused rather than reimplemented.  ``get_viewer()`` is deliberately NOT
    used — it is a process-wide singleton parented to its first caller.
    """
    try:
        from viewer_widget import PointCloudViewer, viewer_available
    except Exception:
        return None
    if not viewer_available():
        return None
    viewer = PointCloudViewer(parent)
    viewer.load_file(path, name=Path(path).name)
    viewer.show()
    return viewer


def _iv_bounds(iv) -> tuple[float, float]:
    """(t0, t1) from a catalog Interval or a plain 2-sequence."""
    try:
        return float(iv.t0), float(iv.t1)
    except Exception:
        return float(iv[0]), float(iv[1])


def _mk_interval(t0: float, t1: float):
    """A catalog Interval when the backend offers one, else a plain tuple."""
    try:
        return pc.Interval(t0=float(t0), t1=float(t1))       # type: ignore[union-attr]
    except Exception:
        return (float(t0), float(t1))


def _qdt(t: float) -> QDateTime:
    return QDateTime.fromSecsSinceEpoch(int(t)).toUTC()


def _utc(t: float) -> str:
    return _qdt(t).toString("yyyy-MM-dd HH:mm:ss")


def _dur(seconds: float) -> str:
    """A duration a human reads at a glance: "1 h 46 m", "6 m 12 s", "42 s"."""
    s = int(round(abs(float(seconds))))
    if s >= 3600:
        return f"{s // 3600} h {(s % 3600) // 60:02d} m"
    if s >= 60:
        return f"{s // 60} m {s % 60:02d} s"
    return f"{s} s"


def _browse_row(parent, edit: QLineEdit, caption: str, start_dir: str = "",
                directory: bool = False) -> QWidget:
    """A QLineEdit + "Browse…" button packed into one row widget."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    btn = QPushButton("Browse…")

    def pick() -> None:
        if directory:
            p = QFileDialog.getExistingDirectory(parent, caption,
                                                 edit.text() or start_dir)
        else:
            p, _ = QFileDialog.getOpenFileName(parent, caption,
                                               edit.text() or start_dir,
                                               "CSV files (*.csv);;All files (*)")
        if p:
            edit.setText(p)

    btn.clicked.connect(pick)
    lay.addWidget(edit)
    lay.addWidget(btn)
    return w


#: Threads detached by closing the window mid-task; held so Qt never destroys a
#: QThread (or its worker) while the callable inside it is still running.
_ORPHANS: set = set()


def _reap_orphans() -> None:
    """Release detached threads that have since finished.

    Called only when another one is detached, never from the thread's own
    ``finished`` signal: that fires while ``run()`` is still on the stack, so
    dropping the last reference there would destroy a live QThread.
    """
    for entry in [e for e in _ORPHANS if not e[0].isRunning()]:
        _ORPHANS.discard(entry)


class _StubJob:
    """Stand-in whole-trackline job used when load_jobs() is unavailable."""

    def __init__(self) -> None:
        self.job_id, self.name, self.intervals = WHOLE, "Whole trackline", []


# ---------------------------------------------------------------------------
# background worker
# ---------------------------------------------------------------------------

class Worker(QObject):
    """Runs one backend callable off the GUI thread; log lines come back as a Signal."""

    line = Signal(str)
    done = Signal(object)
    failed = Signal(str, str)          # (compact cause, full traceback)

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self._fn(self.line.emit))
        except Exception as exc:
            summary = (pc.one_line(exc) if pc is not None
                       else f"{type(exc).__name__}: {exc}")
            self.failed.emit(summary, traceback.format_exc())


# ---------------------------------------------------------------------------
# import dialogs
# ---------------------------------------------------------------------------

class NavImportDialog(QDialog):
    """Renav CSV (headerless) + optional DPA altitude CSV → NavigationConfig."""

    COLS = [("date", 0), ("clock", 1), ("latitude", 2), ("longitude", 3),
            ("depth", 4), ("heading", 5), ("pitch", 6), ("roll", 7)]

    def __init__(self, parent=None, start_dir: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Import navigation and orientation")
        self._start_dir = start_dir
        form = QFormLayout(self)

        self.nav_edit = QLineEdit()
        self.nav_edit.setMinimumWidth(380)
        self.nav_edit.textChanged.connect(self._load_preview)
        form.addRow("Renav CSV (headerless):",
                    _browse_row(self, self.nav_edit, "Renav CSV", start_dir))

        # The file has no header, so the indices below mean nothing without
        # seeing the file: show the first rows with their column numbers.
        self.preview = QTableWidget(0, 0)
        self.preview.setMinimumHeight(140)
        self.preview.verticalHeader().setVisible(False)
        self.preview.setAlternatingRowColors(True)
        self.preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.preview_note = QLabel("pick a CSV to preview its first rows")
        self.preview_note.setStyleSheet(f"color: {C_DIM};")
        form.addRow("First rows (column numbers as headers):", self.preview)
        form.addRow("", self.preview_note)

        self.spins: dict[str, QSpinBox] = {}
        for name, default in self.COLS:
            sp = QSpinBox()
            sp.setRange(0, 200)
            sp.setValue(default)
            sp.valueChanged.connect(self._highlight_columns)
            self.spins[name] = sp
            form.addRow(f"    column index — {name}:", sp)

        self.negate = QCheckBox("Negate depth (store as negative / Z-up)")
        form.addRow("", self.negate)

        self.alt_edit = QLineEdit()
        form.addRow("DPA altitude CSV (optional):",
                    _browse_row(self, self.alt_edit, "DPA altitude CSV", start_dir))
        self.alt_date = QLineEdit("DATE")
        self.alt_time = QLineEdit("TIME")
        self.alt_value = QLineEdit("ALTITUDE(m)")
        form.addRow("    DPA date column:", self.alt_date)
        form.addRow("    DPA time column:", self.alt_time)
        form.addRow("    DPA altitude column:", self.alt_value)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    # -- preview -------------------------------------------------------------
    def _load_preview(self, path: str, rows: int = 5) -> None:
        """First ``rows`` data rows of the CSV, headed by column index."""
        if not hasattr(self, "preview"):        # textChanged during construction
            return
        self.preview.clear()
        self.preview.setRowCount(0)
        self.preview.setColumnCount(0)
        if not (path and Path(path).is_file()):
            self.preview_note.setText("pick a CSV to preview its first rows")
            return
        data: list[list[str]] = []
        try:
            with open(path, "r", errors="replace") as fh:
                for line in fh:
                    if len(data) >= rows:
                        break
                    if line.strip():
                        data.append(line.rstrip("\n").split(","))
        except Exception as exc:
            self.preview_note.setText(f"could not read the file: {exc}")
            return
        if not data:
            self.preview_note.setText("the file is empty")
            return
        n_cols = max(len(r) for r in data)
        self.preview.setColumnCount(n_cols)
        self.preview.setRowCount(len(data))
        self.preview.setHorizontalHeaderLabels([str(c) for c in range(n_cols)])
        for r, row in enumerate(data):
            for c in range(n_cols):
                self.preview.setItem(r, c, QTableWidgetItem(
                    row[c].strip() if c < len(row) else ""))
        self.preview.resizeColumnsToContents()
        self.preview_note.setText(
            f"{len(data)} row(s) shown · {n_cols} columns (0–{n_cols - 1}) · "
            "the spin boxes below are these column numbers")
        self._highlight_columns()

    def _highlight_columns(self) -> None:
        """Name, in the header, the columns the spin boxes currently point at."""
        if not hasattr(self, "spins") or not self.preview.columnCount():
            return
        wanted = {sp.value(): name for name, sp in self.spins.items()}
        for c in range(self.preview.columnCount()):
            label = str(c) + (f"\n{wanted[c]}" if c in wanted else "")
            item = self.preview.horizontalHeaderItem(c)
            if item is None:
                self.preview.setHorizontalHeaderItem(c, QTableWidgetItem(label))
            else:
                item.setText(label)

    # -- result --------------------------------------------------------------
    def navigation_config(self) -> NavigationConfig | None:
        nav = self.nav_edit.text().strip()
        if not nav or not Path(nav).is_file():
            return None
        di, ci = self.spins["date"].value(), self.spins["clock"].value()
        t0, t1 = _csv_bounds(nav, di, ci)

        def src(col_name: str) -> TimeValueSourceConfig:
            return TimeValueSourceConfig(
                csv_path=Path(nav), timestamp_column=str(ci),
                value_column=str(self.spins[col_name].value()), date_column=str(di),
                start_time=t0, end_time=t1, no_header=True)

        alt = None
        alt_path = self.alt_edit.text().strip()
        if alt_path and Path(alt_path).is_file():
            a0, a1 = _headered_bounds(alt_path, self.alt_date.text().strip(),
                                      self.alt_time.text().strip())
            alt = TimeValueSourceConfig(
                csv_path=Path(alt_path), timestamp_column=self.alt_time.text().strip(),
                value_column=self.alt_value.text().strip(),
                date_column=self.alt_date.text().strip(),
                start_time=a0, end_time=a1, no_header=False)

        return NavigationConfig(
            latitude_source=src("latitude"), longitude_source=src("longitude"),
            altitude_source=alt, depth_source=src("depth"),
            negate_depth=self.negate.isChecked(), heading_source=src("heading"),
            pitch_source=src("pitch"), roll_source=src("roll"))


class VideoImportDialog(QDialog):
    """Video directory + filename time format."""

    def __init__(self, parent=None, directory: str = "", fmt: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Import video")
        form = QFormLayout(self)
        self.dir_edit = QLineEdit(directory)
        self.dir_edit.setMinimumWidth(380)
        form.addRow("Video directory:",
                    _browse_row(self, self.dir_edit, "Video directory",
                                directory, directory=True))
        self.fmt_edit = QLineEdit(fmt or "%Y_%m_%dT%H_%M_%S")
        form.addRow("Filename time format:", self.fmt_edit)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def values(self) -> tuple[str, str]:
        return self.dir_edit.text().strip(), self.fmt_edit.text().strip()


class SensorChannelDialog(QDialog):
    """CSV + timestamp column + one channel (source column, name, units, delay)."""

    def __init__(self, parent=None, start_dir: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Import new sensor channel")
        self._start_dir = start_dir
        form = QFormLayout(self)
        self.csv_edit = QLineEdit()
        self.csv_edit.setMinimumWidth(380)
        self.csv_edit.textChanged.connect(self._load_columns)
        form.addRow("Sensor CSV:",
                    _browse_row(self, self.csv_edit, "Sensor CSV", start_dir))

        self.ts_combo = QComboBox()
        self.ts_combo.setEditable(True)
        self.col_combo = QComboBox()
        self.col_combo.setEditable(True)
        self.col_combo.currentTextChanged.connect(self._sync_name)
        self.name_edit = QLineEdit()
        self.units_edit = QLineEdit()
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(-3600.0, 3600.0)
        self.delay_spin.setDecimals(2)
        self.delay_spin.setValue(0.0)
        form.addRow("Timestamp column:", self.ts_combo)
        form.addRow("Channel source column:", self.col_combo)
        form.addRow("Display name:", self.name_edit)
        form.addRow("Units:", self.units_edit)
        form.addRow("Time delay (s):", self.delay_spin)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def _load_columns(self, p: str) -> None:
        """Populate the two column combos from the CSV's header row."""
        if not (p and Path(p).is_file()):
            return
        cols: list[str] = []
        try:
            with open(p, "r", errors="replace") as fh:
                cols = [c.strip() for c in fh.readline().split(",") if c.strip()]
        except Exception:
            pass
        for cb in (self.ts_combo, self.col_combo):
            cb.clear()
            cb.addItems(cols)
        guess = next((c for c in cols if any(k in c.lower()
                                             for k in ("datetime", "time", "date"))), "")
        if guess:
            self.ts_combo.setCurrentText(guess)

    def _sync_name(self, text: str) -> None:
        if not self.name_edit.text().strip():
            self.name_edit.setText(text)

    def values(self) -> tuple[str, str, SensorChannel] | None:
        csv_path = self.csv_edit.text().strip()
        col = self.col_combo.currentText().strip()
        ts = self.ts_combo.currentText().strip()
        if not (csv_path and col and ts):
            return None
        ch = SensorChannel(source_column=col,
                           display_name=self.name_edit.text().strip() or col,
                           units=self.units_edit.text().strip(), use_header_name=False,
                           time_delay_s=float(self.delay_spin.value()))
        return csv_path, ts, ch


class GenerateDialog(QDialog):
    """Settings form from ProductType.settings_schema, with two run buttons."""

    def __init__(self, parent, type_label: str, schema, busy: bool = False,
                 scope: str = "") -> None:
        super().__init__(parent)
        # The scope belongs in the title AND in the body: a long run against the
        # wrong job is the expensive mistake this dialog can cause.
        self.setWindowTitle(f"Generate — {type_label}"
                            + (f" — {scope}" if scope else ""))
        self.mode: str = ""
        self._widgets: dict[str, tuple[str, QWidget]] = {}
        self._defaults: dict = {}
        outer = QVBoxLayout(self)
        if scope:
            head = QLabel(f"<b>{type_label}</b><br>scope: <b>{scope}</b>")
            head.setTextFormat(Qt.TextFormat.RichText)
            outer.addWidget(head)
        form = QFormLayout()
        outer.addLayout(form)

        for entry in list(schema or []):
            entry = list(entry) + [None] * (5 - len(list(entry)))
            key, label, kind, default, extra = entry[:5]
            kind = (kind or "str").lower()
            if kind == "float":
                w = QDoubleSpinBox()
                w.setDecimals(4)
                w.setRange(-1e9, 1e9)
                w.setValue(float(default or 0.0))
            elif kind == "int":
                w = QSpinBox()
                w.setRange(-10 ** 9, 10 ** 9)
                w.setValue(int(default or 0))
            elif kind == "choice":
                w = QComboBox()
                opts = [str(o) for o in (extra or [])]
                w.addItems(opts)
                if default is not None and str(default) in opts:
                    w.setCurrentText(str(default))
            elif kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(default))
            else:
                w = QLineEdit("" if default is None else str(default))
            self._widgets[str(key)] = (kind, w)
            self._defaults[str(key)] = default
            form.addRow(str(label or key) + ":", w)

        if not self._widgets:
            form.addRow(QLabel("This product takes no settings."))

        btns = QHBoxLayout()
        self.btn_default = QPushButton("Default Run")
        self.btn_custom = QPushButton("Run with these settings")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_default.setToolTip("Ignore the form above and run the settled "
                                    "defaults for this product.")
        self.btn_custom.setToolTip("Run with exactly the values shown above.")
        for b in (self.btn_default, self.btn_custom):
            b.setEnabled(not busy)
        self.btn_default.clicked.connect(lambda: self._go("default"))
        self.btn_custom.clicked.connect(lambda: self._go("custom"))
        self.btn_cancel.clicked.connect(self.reject)
        btns.addStretch(1)
        for b in (self.btn_default, self.btn_custom, self.btn_cancel):
            btns.addWidget(b)
        outer.addLayout(btns)

    def _go(self, mode: str) -> None:
        self.mode = mode
        self.accept()

    def defaults(self) -> dict:
        """Schema defaults, ignoring whatever the user typed into the form."""
        return {k: v for k, v in self._defaults.items() if v is not None}

    def values(self) -> dict:
        out: dict = {}
        for key, (kind, w) in self._widgets.items():
            if kind in ("float", "int"):
                out[key] = w.value()
            elif kind == "choice":
                out[key] = w.currentText()
            elif kind == "bool":
                out[key] = w.isChecked()
            else:
                out[key] = w.text()
        return out


# ---------------------------------------------------------------------------
# trackline panel
# ---------------------------------------------------------------------------

class TracklinePanel(QWidget):
    """Track map with click-to-pick intervals, staging and job creation.

    Navigation: wheel zooms about the cursor, middle- or right-drag pans, and
    "Reset view" returns to the whole track.  Plain LEFT clicks are never
    navigation — they are the two interval picks — so panning cannot steal them.
    """

    createJob = Signal(list)          # list[(t0, t1)]
    message = Signal(str)

    ZOOM_STEP = 1.3

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.track: np.ndarray | None = None
        self.job_intervals: list[tuple[float, float]] = []
        self.pending: list[tuple[float, float]] = []
        self.staged: list[tuple[float, float]] = []
        self.marks: list[tuple[float, float]] = []   # easting/northing of clicks
        self._first: float | None = None
        self._zoomed = False              # user has moved the view off "home"
        self._pan: tuple | None = None     # (px, py, xlim, ylim, sx, sy)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.fig = Figure(figsize=(6, 5), facecolor=BG)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        lay.addWidget(self.canvas, 1)

        # Live readout: the time under the cursor, and the state of a pick in
        # progress.  A dangling first click is only safe if it is visible.
        self.readout = QLabel("")
        self.readout.setFont(QFont("monospace", 8))
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight
                                  | Qt.AlignmentFlag.AlignVCenter)
        self.readout.setStyleSheet(f"color: {MUT};")
        lay.addWidget(self.readout)

        typed = QHBoxLayout()
        self.dt_from, self.dt_to = QDateTimeEdit(), QDateTimeEdit()
        for e in (self.dt_from, self.dt_to):
            try:                                     # Qt >= 6.7
                e.setTimeZone(QTimeZone(QTimeZone.Initialization.UTC))
            except Exception:                        # pragma: no cover
                e.setTimeSpec(Qt.TimeSpec.UTC)
            e.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            e.setCalendarPopup(True)
        btn_typed = QPushButton("Add typed interval")
        btn_typed.clicked.connect(self._add_typed)
        typed.addWidget(QLabel("From (UTC):"))
        typed.addWidget(self.dt_from)
        typed.addWidget(QLabel("To (UTC):"))
        typed.addWidget(self.dt_to)
        typed.addWidget(btn_typed)
        typed.addStretch(1)
        lay.addLayout(typed)

        row = QHBoxLayout()
        self.btn_stage = QPushButton("Add interval to job")
        self.btn_create = QPushButton("Create job")
        self.btn_clear = QPushButton("Clear picks && staged")
        self.btn_clear.setToolTip("Forget the pending clicks, their markers and "
                                  "everything staged.")
        self.btn_reset = QPushButton("Reset view")
        self.btn_reset.setToolTip("Zoom: mouse wheel · Pan: middle- or right-drag")
        self.counter = QLabel("staged: 0")
        self.btn_stage.clicked.connect(self._stage)
        self.btn_create.clicked.connect(self._create)
        self.btn_clear.clicked.connect(self._clear)
        self.btn_reset.clicked.connect(self.reset_view)
        for w in (self.btn_stage, self.btn_create, self.btn_clear, self.btn_reset):
            row.addWidget(w)
        row.addWidget(self.counter)
        row.addStretch(1)
        hint = QLabel("wheel = zoom · right-drag = pan · left click ×2 = interval")
        hint.setStyleSheet(f"color: {MUT};")
        row.addWidget(hint)
        lay.addLayout(row)
        self.canvas.setToolTip("Click two track points to define an interval.\n"
                               "Mouse wheel zooms, middle- or right-drag pans.")
        self._redraw()

    # -- data ---------------------------------------------------------------
    def set_track(self, arr) -> None:
        try:
            a = np.asarray(arr, dtype=float)
            self.track = a if (a.ndim == 2 and a.shape[1] >= 3 and len(a)) else None
        except Exception:
            self.track = None
        self._zoomed = False                # new data: show all of it
        if self.track is not None:
            t0, t1 = float(self.track[0, 0]), float(self.track[-1, 0])
            self.dt_from.setDateTime(_qdt(t0))
            self.dt_to.setDateTime(_qdt(t1))
            self.dt_from.setToolTip(f"dive spans {_utc(t0)} → {_utc(t1)} UTC")
            self.dt_to.setToolTip(self.dt_from.toolTip())
        self._refresh_readout()
        self._redraw()

    def set_job_intervals(self, intervals) -> None:
        self.job_intervals = [_iv_bounds(iv) for iv in (intervals or [])]
        self.cancel_pick(quiet=True)    # a half-made pick must not outlive a job switch
        self._redraw()

    # -- pick state ----------------------------------------------------------
    def dive_bounds(self) -> tuple[float, float] | None:
        if self.track is None or not len(self.track):
            return None
        return float(self.track[0, 0]), float(self.track[-1, 0])

    def cancel_pick(self, quiet: bool = False) -> bool:
        """Drop a half-finished click pair (and its markers).  True if there was one."""
        had = self._first is not None or bool(self.marks)
        self._first = None
        if self.marks:
            self.marks.clear()
        if had and not quiet:
            self.message.emit("interval pick cancelled")
        self._refresh_readout()
        if had:
            self._redraw()
        return had

    def _refresh_readout(self, cursor: str = "") -> None:
        bits = []
        if self._first is not None:
            bits.append("pick open: start "
                        + _qdt(self._first).toString("HH:mm:ss")
                        + " — click a 2nd point (Esc cancels)")
        elif self.pending:
            t0, t1 = self.pending[-1]
            bits.append(f"pending {_qdt(t0).toString('HH:mm:ss')} → "
                        f"{_qdt(t1).toString('HH:mm:ss')} ({_dur(t1 - t0)})")
        if cursor:
            bits.append(cursor)
        self.readout.setText("   ·   ".join(bits))
        self.readout.setStyleSheet(
            f"color: {C_MARK if self._first is not None else MUT};")

    def set_busy(self, busy: bool) -> None:
        self.btn_create.setEnabled(not busy)

    # -- view navigation ------------------------------------------------------
    def reset_view(self) -> None:
        """Back to the whole track."""
        self._zoomed = False
        self._pan = None
        self._redraw()

    def _on_scroll(self, event) -> None:
        """Wheel zoom centred on the cursor (equal aspect is preserved)."""
        if self.track is None or event.inaxes is not self.ax:
            return
        factor = (1.0 / self.ZOOM_STEP) if event.button == "up" else self.ZOOM_STEP
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = event.xdata if event.xdata is not None else (x0 + x1) / 2.0
        cy = event.ydata if event.ydata is not None else (y0 + y1) / 2.0
        self.ax.set_xlim(cx + (x0 - cx) * factor, cx + (x1 - cx) * factor)
        self.ax.set_ylim(cy + (y0 - cy) * factor, cy + (y1 - cy) * factor)
        self._zoomed = True
        self.canvas.draw_idle()

    def _on_press(self, event) -> None:
        """Left = interval pick; middle/right = start a pan."""
        if event.button == 1:
            self._on_click(event)
        elif event.button in (2, 3) and event.inaxes is self.ax:
            box = self.ax.get_window_extent()
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            # Pixel->data scale frozen at press: the limits move during the drag.
            self._pan = (event.x, event.y, (x0, x1), (y0, y1),
                         (x1 - x0) / max(1.0, box.width),
                         (y1 - y0) / max(1.0, box.height))

    def _on_motion(self, event) -> None:
        if self._pan and event.x is not None and event.y is not None:
            px, py, (x0, x1), (y0, y1), sx, sy = self._pan
            dx = (event.x - px) * sx
            dy = (event.y - py) * sy
            self.ax.set_xlim(x0 - dx, x1 - dx)
            self.ax.set_ylim(y0 - dy, y1 - dy)
            self._zoomed = True
            self.canvas.draw_idle()
            return
        self._hover(event)

    def _hover(self, event) -> None:
        """Time under the cursor — a QLabel, so no canvas redraw per mouse move."""
        hit = self._nearest(event)
        if hit is None:
            self._refresh_readout()
            return
        i, distance = hit
        t = float(self.track[i, 0])
        near = distance <= CLICK_TOLERANCE_PX
        self._refresh_readout(
            f"cursor {_utc(t)} UTC ({distance:.0f} px from track"
            + ("" if near else " — too far to pick") + ")")

    def _nearest(self, event) -> tuple[int, float] | None:
        """(index, pixel distance) of the track vertex nearest the event."""
        if self.track is None or event.inaxes is not self.ax:
            return None
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return None
        d2 = ((self.track[:, 1] - event.xdata) ** 2
              + (self.track[:, 2] - event.ydata) ** 2)
        i = int(np.argmin(d2))
        try:                       # measure in PIXELS: metres mean nothing zoomed
            (vx, vy), (ex, ey) = self.ax.transData.transform([
                (self.track[i, 1], self.track[i, 2]), (event.xdata, event.ydata)])
            distance = float(np.hypot(vx - ex, vy - ey))
        except Exception:                                   # pragma: no cover
            distance = 0.0
        return i, distance

    def _on_release(self, event) -> None:
        if self._pan and event.button in (2, 3):
            self._pan = None

    # -- interval building ---------------------------------------------------
    def _on_click(self, event) -> None:
        hit = self._nearest(event)
        if hit is None:
            return
        i, distance = hit
        if distance > CLICK_TOLERANCE_PX:
            # Without this, a click in an empty corner silently snaps to
            # whatever vertex happens to be nearest — hours from what was meant.
            self.message.emit(
                f"ignored: that click is {distance:.0f} px from the track "
                f"(limit {CLICK_TOLERANCE_PX:.0f} px) — click on the line, or "
                "zoom in first")
            self._refresh_readout()
            return
        t = float(self.track[i, 0])
        # Mark the picked track point at once, so the user sees WHERE the click
        # landed (the nearest vertex, not the cursor) before the second click.
        self.marks.append((float(self.track[i, 1]), float(self.track[i, 2])))
        if self._first is None:
            self._first = t
            self.message.emit(f"interval start {_utc(t)} UTC "
                              "— click a second point (Esc cancels)")
            self._refresh_readout()
            self._redraw()
        else:
            a, b = sorted((self._first, t))
            self._first = None
            if b <= a:                 # rejected below: drop this pair's markers
                del self.marks[-2:]
            self._add(a, b)

    def _add_typed(self) -> None:
        a = float(self.dt_from.dateTime().toSecsSinceEpoch())
        b = float(self.dt_to.dateTime().toSecsSinceEpoch())
        if b < a:
            # Silently swapping hides a typo; say so and put the fields right.
            self.message.emit(f"note: From/To were reversed — swapped to "
                              f"{_utc(b)} → {_utc(a)} UTC")
            a, b = b, a
            self.dt_from.setDateTime(_qdt(a))
            self.dt_to.setDateTime(_qdt(b))
        self._add(a, b)

    def _add(self, t0: float, t1: float) -> None:
        if t1 <= t0:
            self.message.emit("ignored zero-length interval")
            self._refresh_readout()
            self._redraw()
            return
        bounds = self.dive_bounds()
        if bounds is not None:
            d0, d1 = bounds
            if t1 < d0 or t0 > d1:
                # An interval outside the dive can only ever make empty products.
                self.message.emit(
                    f"rejected {_utc(t0)} → {_utc(t1)}: outside this dive "
                    f"({_utc(d0)} → {_utc(d1)} UTC)")
                self._refresh_readout()
                return
            if t0 < d0 or t1 > d1:
                t0, t1 = max(t0, d0), min(t1, d1)
                self.message.emit(f"note: interval clipped to the dive — "
                                  f"{_utc(t0)} → {_utc(t1)} UTC")
                if t1 <= t0:
                    self.message.emit("nothing left of that interval inside the dive")
                    self._refresh_readout()
                    return
        self.pending.append((t0, t1))
        self.message.emit(f"pending interval {_utc(t0)} → {_utc(t1)} UTC "
                          f"({_dur(t1 - t0)}, {len(self.pending)} pending)")
        self._refresh_readout()
        self._redraw()

    def _stage(self) -> None:
        if not self.pending:
            self.message.emit("no pending interval to add — click two track points")
            return
        self.staged.extend(self.pending)
        self.pending.clear()
        self.marks.clear()          # the pending pair is now a staged span
        self._first = None
        self._refresh_counter()
        self._refresh_readout()
        self._redraw()

    def _create(self) -> None:
        if self.pending:            # never drop an un-staged interval on create
            self._stage()
        if not self.staged:
            self.message.emit("nothing staged — add at least one interval")
            return
        self.createJob.emit(list(self.staged))

    def _clear(self) -> None:
        self.staged.clear()
        self.pending.clear()
        self.marks.clear()
        self._first = None
        self._refresh_counter()
        self._refresh_readout()
        self._redraw()

    def job_created(self) -> None:
        """Called by the window once create_job() succeeded."""
        self._clear()

    def _refresh_counter(self) -> None:
        self.counter.setText(f"staged: {len(self.staged)}")

    # -- drawing -------------------------------------------------------------
    def _span(self, t0: float, t1: float, color: str, lw: float) -> None:
        if self.track is None:
            return
        m = (self.track[:, 0] >= t0) & (self.track[:, 0] <= t1)
        if m.sum() >= 2:
            self.ax.plot(self.track[m, 1], self.track[m, 2], "-", color=color, lw=lw,
                         solid_capstyle="round", zorder=3)

    def _redraw(self) -> None:
        ax = self.ax
        # ax.clear() autoscales; keep whatever the user zoomed/panned to.
        keep = (ax.get_xlim(), ax.get_ylim()) if self._zoomed else None
        ax.clear()
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(GRIDC)
        ax.tick_params(colors=MUT, labelsize=8)
        ax.grid(True, color=GRIDC, lw=0.4, alpha=0.7)
        ax.set_xlabel("easting (m)", color=MUT, fontsize=8)
        ax.set_ylabel("northing (m)", color=MUT, fontsize=8)
        if self.track is None:
            ax.text(0.5, 0.5, "no trackline available", color=MUT, ha="center",
                    va="center", transform=ax.transAxes, fontsize=10)
        else:
            ax.plot(self.track[:, 1], self.track[:, 2], "-", color=MUT, lw=0.7,
                    alpha=0.85, zorder=2)
            for t0, t1 in self.job_intervals:
                self._span(t0, t1, C_JOB, 4.5)
            for t0, t1 in self.staged:
                self._span(t0, t1, C_STAGED, 3.0)
            for t0, t1 in self.pending:
                self._span(t0, t1, C_PEND, 3.0)
            if self.marks:
                ax.plot([e for e, _ in self.marks], [n for _, n in self.marks],
                        marker="*", ls="none", ms=15, mfc=C_MARK, mec=BG, mew=0.8,
                        zorder=6)
            ax.set_aspect("equal", adjustable="datalim")
            if keep is not None:
                ax.set_xlim(*keep[0])
                ax.set_ylim(*keep[1])
        self.fig.tight_layout()
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# product viewer
# ---------------------------------------------------------------------------

class _DFModel(QAbstractTableModel):
    def __init__(self, df) -> None:
        super().__init__()
        self._df = df

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return str(self._df.iat[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


class ProductViewer(QMainWindow):
    """Views one ProductInstance's view_paths — one tab per path.

    Tabs are built LAZILY, on first selection.  A 28-chunk photogrammetry run
    offers 80+ views, and decoding every orthomosaic up front would stall the
    window for minutes and exhaust a WSL VM; only the tab being looked at is
    ever materialised.
    """

    MAX_PX = 1400

    def __init__(self, instance, track=None, parent=None) -> None:
        super().__init__(parent)
        label = getattr(instance, "label", "product")
        self.setWindowTitle(f"View — {label}")
        self.resize(1000, 760)
        self._track = track
        self._keep: list = []                       # keep QImage buffers alive
        self._children: list = []                   # spawned 3-D viewer windows
        self._pending: dict[int, str] = {}           # tab index -> path, unbuilt
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.TextElideMode.ElideMiddle)
        self.setCentralWidget(self.tabs)

        paths = list(getattr(instance, "view_paths", None)
                     or ([getattr(instance, "path", "")] if getattr(instance, "path", "")
                         else []))
        if not paths:
            self.tabs.addTab(self._msg("This product has no viewable files."), "empty")
            return
        for p in paths:
            name = Path(str(p)).name[:28] or "file"
            # Each tab is a container whose contents are swapped in on demand:
            # removing and re-inserting tabs instead would renumber every other
            # tab and scramble the pending map.
            host = QWidget()
            box = QVBoxLayout(host)
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(self._msg(f"loading {name} …"))
            index = self.tabs.addTab(host, name)
            self.tabs.setTabToolTip(index, str(p))
            self._pending[index] = str(p)
        self.tabs.currentChanged.connect(self._ensure_tab)
        self._ensure_tab(self.tabs.currentIndex())

    def _ensure_tab(self, index: int) -> None:
        """Materialise one tab's real widget the first time it is shown."""
        path = self._pending.pop(int(index), None)
        if path is None:
            return
        try:
            widget = self._widget_for(path)
        except Exception as exc:
            widget = self._msg(f"Could not open {path}\n\n{type(exc).__name__}: {exc}")
        host = self.tabs.widget(int(index))
        layout = host.layout()
        while layout.count():
            item = layout.takeAt(0)
            stale = item.widget()
            if stale is not None:
                # takeAt only drops the LAYOUT item: the placeholder would stay
                # parented to the host and keep painting over the real widget
                # until deleteLater is processed.  Unparent it now.
                stale.setParent(None)
                stale.deleteLater()
        layout.addWidget(widget)

    # -- per-type widgets ----------------------------------------------------
    def _msg(self, text: str) -> QWidget:
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lab.setMargin(24)
        return lab

    def _scroll(self, w: QWidget) -> QWidget:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(w)
        return sa

    def _widget_for(self, path: str) -> QWidget:
        ext = Path(path).suffix.lower()
        if not Path(path).exists():
            return self._msg(f"missing file:\n{path}")
        if Path(path).is_dir():
            return self._scroll(self._directory_grid(path))
        if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            pm = QPixmap(path)
            if pm.isNull():
                return self._msg(f"unreadable image:\n{path}")
            if max(pm.width(), pm.height()) > self.MAX_PX:
                pm = pm.scaled(self.MAX_PX, self.MAX_PX,
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
            lab = QLabel()
            lab.setPixmap(pm)
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return self._scroll(lab)
        if ext in (".tif", ".tiff"):
            return self._scroll(self._raster_label(path))
        if ext in (".pdf", ".html", ".htm"):
            # Qt has no viewer for these; hand them to the desktop ON REQUEST —
            # never at tab-construction time, or one product with three PDFs
            # would launch three external viewers the moment it is opened.
            btn = QPushButton("Open in external viewer")
            btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(path)))
            folder = QPushButton("Open containing folder")
            folder.clicked.connect(lambda: _open_folder(path))
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.addWidget(self._msg(f"{ext.lstrip('.').upper()} document:\n{path}"))
            lay.addWidget(btn)
            lay.addWidget(folder)
            lay.addStretch(1)
            return w
        if ext == ".csv":
            import pandas as pd
            df = pd.read_csv(path, nrows=500)
            view = QTableView()
            model = _DFModel(df)
            self._keep.append(model)
            view.setModel(model)
            view.resizeColumnsToContents()
            return view
        if ext in (".geojson", ".json"):
            return self._geojson_canvas(path)
        if ext in MESH_EXTS:
            return self._mesh_widget(path)
        # Anything else (e.g. a Metashape .psx): say so and offer the folder.
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(self._msg(f"No built-in viewer for '{ext}' files:\n{path}"))
        btn = QPushButton("Open containing folder")
        btn.clicked.connect(lambda: _open_folder(path))
        lay.addWidget(btn)
        lay.addStretch(1)
        return w

    def _directory_grid(self, path: str, limit: int = 12) -> QWidget:
        """Frame-set style view_path: a contact sheet of the first images."""
        files = sorted(p for p in Path(path).iterdir() if p.is_file())
        imgs = [p for p in files
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")]
        w = QWidget()
        lay = QVBoxLayout(w)
        head = QLabel(f"{path}\n{len(files)} file(s), {len(imgs)} image(s) — "
                      f"showing first {min(limit, len(imgs))}")
        head.setWordWrap(True)
        lay.addWidget(head)
        btn = QPushButton("Open containing folder")
        btn.clicked.connect(lambda: _open_folder(path))
        lay.addWidget(btn)
        grid = QGridLayout()
        for i, p in enumerate(imgs[:limit]):
            pm = QPixmap(str(p))
            if pm.isNull():
                continue
            cell = QLabel()
            cell.setPixmap(pm.scaled(240, 240, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))
            cell.setToolTip(p.name)
            grid.addWidget(cell, i // 4, i % 4)
        lay.addLayout(grid)
        lay.addStretch(1)
        return w

    def _mesh_widget(self, path: str) -> QWidget:
        """A 3-D product: full PyVista viewer when available, preview always.

        The preview is a rotatable matplotlib 3-D scatter of a bounded sample
        (see ``_sample_ply``), so a 2 GB dense cloud opens in a fraction of a
        second and never loads whole.  The exact sample size is stated on the
        figure: this is a preview, not the product.
        """
        ext = Path(path).suffix.lower()
        if ext == ".ply":
            xyz, rgb, n_vertices, n_faces = _sample_ply(path)
            kind = "point cloud" if not n_faces else "mesh"
        elif ext == ".obj":
            xyz, rgb, n_vertices, n_faces = _sample_obj(path)
            kind = "mesh"
        else:
            xyz, rgb, n_vertices, n_faces = np.zeros((0, 3)), None, 0, 0
            kind = ext.lstrip(".").upper()

        w = QWidget()
        lay = QVBoxLayout(w)
        size_mb = Path(path).stat().st_size / 1e6
        head = QLabel(f"<b>{Path(path).name}</b> — {kind}, {size_mb:,.0f} MB"
                      + (f", {n_vertices:,} vertices" if n_vertices else "")
                      + (f", {n_faces:,}+ faces" if n_faces else ""))
        head.setTextFormat(Qt.TextFormat.RichText)
        head.setWordWrap(True)
        lay.addWidget(head)

        row = QHBoxLayout()
        btn3d = QPushButton("Open in 3-D viewer")
        have3d = False
        try:
            from viewer_widget import viewer_available
            have3d = bool(viewer_available())
        except Exception:
            have3d = False
        btn3d.setEnabled(have3d)
        btn3d.setToolTip("viewer_widget.PointCloudViewer (PyVista/VTK)" if have3d
                         else "PyVista is not installed in this environment — "
                              "the preview below and the external viewer remain "
                              "available")
        btn3d.clicked.connect(lambda: self._open_3d(path))
        ext_btn = QPushButton("Open in external viewer")
        ext_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(path)))
        folder = QPushButton("Open containing folder")
        folder.clicked.connect(lambda: _open_folder(path))
        for b in (btn3d, ext_btn, folder):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        if not len(xyz):
            lay.addWidget(self._msg(
                f"No built-in preview for '{ext}' — use the buttons above."))
            lay.addStretch(1)
            return w

        fig = Figure(facecolor=BG)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(BG)
        if rgb is not None:
            # Seafloor RGB at 2,500 m is near-black; the same 2-98 percentile
            # stretch the GeoTIFF tab uses makes it legible without inventing
            # colour that is not in the file.
            flat = rgb.reshape(-1)
            lo, hi = float(np.percentile(flat, 2)), float(np.percentile(flat, 98))
            colour = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
            cmap = None
        else:
            colour, cmap = xyz[:, 2], "viridis"
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colour, s=2.0, lw=0,
                   cmap=cmap, depthshade=False)
        span = float(max(np.ptp(xyz, axis=0).max(), 1e-6))
        mid = xyz.min(axis=0) + np.ptp(xyz, axis=0) / 2
        for setter, m in ((ax.set_xlim, mid[0]), (ax.set_ylim, mid[1]),
                          (ax.set_zlim, mid[2])):
            setter(m - span / 2, m + span / 2)      # equal aspect, no distortion
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.055, 0.086, 0.125, 1.0))
            axis.line.set_color(GRIDC)
            axis.label.set_color(MUT)
            axis._axinfo["grid"]["color"] = GRIDC
        ax.tick_params(colors=MUT, labelsize=7)
        ax.set_xlabel("easting (m)", fontsize=8)
        ax.set_ylabel("northing (m)", fontsize=8)
        ax.set_zlabel("z (m)", fontsize=8)
        ax.set_title(f"preview — {len(xyz):,} of {n_vertices:,} vertices "
                     f"(drag to rotate)", color=INK, fontsize=9)
        fig.tight_layout()
        lay.addWidget(FigureCanvasQTAgg(fig), 1)
        return w

    def _open_3d(self, path: str) -> None:
        viewer = _pyvista_viewer(path, self)
        if viewer is None:
            QMessageBox.information(
                self, "3-D viewer unavailable",
                "PyVista/pyvistaqt is not installed in this environment, so the "
                "full 3-D viewer cannot start.\n\nThe preview below is a bounded "
                "sample of the same file; “Open in external viewer” hands it to "
                "the desktop.")
            return
        self._children.append(viewer)

    def _raster_label(self, path: str) -> QWidget:
        import rasterio
        with rasterio.open(path) as ds:
            # ceil, not trunc: a 2.8x oversized raster must still be decimated.
            step = max(1, -(-max(ds.width, ds.height) // self.MAX_PX))
            h, w = max(1, ds.height // step), max(1, ds.width // step)
            n = 3 if ds.count >= 3 else 1
            arr = ds.read(list(range(1, n + 1)), out_shape=(n, h, w),
                          masked=True).astype("float32").filled(np.nan)
        bands = []
        for b in arr:
            fin = b[np.isfinite(b)]
            lo, hi = (np.percentile(fin, 2), np.percentile(fin, 98)) if fin.size else (0, 1)
            if hi <= lo:
                hi = lo + 1.0
            scaled = np.clip((np.nan_to_num(b, nan=lo) - lo) / (hi - lo), 0, 1) * 255
            bands.append(scaled.astype(np.uint8))
        rgb = np.dstack(bands if len(bands) == 3 else bands * 3)
        buf = np.ascontiguousarray(rgb)
        img = QImage(buf.data, buf.shape[1], buf.shape[0], buf.strides[0],
                     QImage.Format.Format_RGB888).copy()
        self._keep.append(buf)
        lab = QLabel()
        lab.setPixmap(QPixmap.fromImage(img))
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return lab

    def _geojson_canvas(self, path: str) -> QWidget:
        obj = json.loads(Path(path).read_text())
        pts: list[tuple[float, float]] = []

        def walk(c) -> None:
            if isinstance(c, (list, tuple)):
                if len(c) >= 2 and all(isinstance(v, (int, float)) for v in c[:2]):
                    pts.append((float(c[0]), float(c[1])))
                else:
                    for sub in c:
                        walk(sub)
        feats = obj.get("features", [obj]) if isinstance(obj, dict) else []
        for f in feats:
            geo = (f or {}).get("geometry") or f
            walk((geo or {}).get("coordinates"))

        fig = Figure(facecolor=BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRIDC)
        ax.grid(True, color=GRIDC, lw=0.4, alpha=0.7)
        if pts:
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            # Overlay the trackline only when the coordinate systems agree.
            if self._track is not None and len(self._track):
                tx, ty = self._track[:, 1], self._track[:, 2]
                if (tx.min() - 5000 <= xs.mean() <= tx.max() + 5000
                        and ty.min() - 5000 <= ys.mean() <= ty.max() + 5000):
                    ax.plot(tx, ty, "-", color=MUT, lw=0.7, alpha=0.8)
            ax.plot(xs, ys, ".", color=C_JOB, ms=4)
            ax.set_aspect("equal", adjustable="datalim")
        else:
            ax.text(0.5, 0.5, "no coordinates in file", color=MUT, ha="center",
                    va="center", transform=ax.transAxes)
        ax.set_title(Path(path).name, color=INK, fontsize=9)
        fig.tight_layout()
        return FigureCanvasQTAgg(fig)


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """The three-panel simplified UI."""

    def __init__(self, workspace_dir: str | Path) -> None:
        super().__init__()
        self.ws_dir = Path(workspace_dir).resolve()
        self.ws_path = str(self.ws_dir)
        self.ws_json = self.ws_dir / "workspace.json"
        self.ws: dict = {}
        self.ws_loaded = True           # False => workspace.json is unreadable
        self.jobs: list = []
        self.track: np.ndarray | None = None
        self._busy = False
        self._task_title = ""
        self._seen_errors: set[str] = set()
        self._details: list[str] = []       # collapsed traceback lines
        self._last_details = ""
        self._fails = 0                     # red lines during the current task
        self._last_fail = ""                # the most recent red line's text
        self._tree_col_w = 0                # widest the product column has been
        self._thread: QThread | None = None
        self._worker: Worker | None = None
        self._viewers: list[ProductViewer] = []

        self.setWindowTitle(f"EPR Imaging — {self.ws_dir.name}")
        self.resize(1620, 940)
        self._build_ui()
        self._build_menus()
        self.statusBar().showMessage("ready")

        self.log(f"workspace {self.ws_dir}")
        self.log("ready — expand a product type and double-click ＋ Generate New…, "
                 "or Run ▸ Default run on all")
        if PC_IMPORT_ERROR:
            self.log_error(f"product_catalog unavailable — {PC_IMPORT_ERROR}")
        self._load_workspace()
        self.refresh_jobs()
        self._load_track()

    # -- construction --------------------------------------------------------
    DEFAULT_SIZES = [320, 950, 350]

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter = splitter
        # One careless drag used to collapse the job selector and product tree
        # to zero width, with no way back.
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        self.setCentralWidget(splitter)

        # products
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        self.job_combo = QComboBox()
        self.job_combo.currentIndexChanged.connect(self._on_job_changed)
        self.job_combo.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.job_combo.customContextMenuRequested.connect(self._job_menu)
        self.job_combo.setToolTip("Right-click to rename or delete a job.")
        job_head = QHBoxLayout()
        job_head.setContentsMargins(0, 0, 0, 0)
        job_head.addWidget(QLabel("Job"))
        job_head.addStretch(1)
        self.btn_job_manage = QPushButton("⋯")
        self.btn_job_manage.setFixedWidth(26)
        self.btn_job_manage.setToolTip("Rename or delete the selected job")
        self.btn_job_manage.clicked.connect(
            lambda: self._job_menu(self.btn_job_manage.geometry().bottomLeft()))
        job_head.addWidget(self.btn_job_manage)
        lv.addLayout(job_head)
        lv.addWidget(self.job_combo)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Products"])
        # Size the column to its contents (never below the widest it has had),
        # and elide what still does not fit so there is always a visible "…".
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.tree.itemExpanded.connect(self._on_expanded)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        lv.addWidget(self.tree, 1)
        left.setMinimumWidth(270)
        splitter.addWidget(left)

        # trackline
        self.trackline = TracklinePanel()
        self.trackline.setMinimumWidth(360)
        self.trackline.message.connect(self.log)
        self.trackline.createJob.connect(self._create_job)
        splitter.addWidget(self.trackline)

        # logger
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(QLabel("Log"))
        head.addStretch(1)
        self.chk_wrap = QCheckBox("wrap")
        self.chk_wrap.setChecked(True)
        self.chk_wrap.setToolTip("Wrap long lines (off = horizontal scrollbar)")
        self.chk_wrap.toggled.connect(self._set_wrap)
        head.addWidget(self.chk_wrap)
        self.btn_details = QPushButton("details")
        self.btn_details.setToolTip("Show the collapsed traceback of the last failure")
        self.btn_details.setEnabled(False)
        self.btn_details.clicked.connect(self._show_details)
        head.addWidget(self.btn_details)
        rv.addLayout(head)
        self.logger = QPlainTextEdit()
        self.logger.setReadOnly(True)
        self.logger.setMaximumBlockCount(20000)
        self.logger.setFont(QFont("monospace", 8))
        # Wrapped by default: this panel is the app's only feedback channel and
        # a clipped path is worse than a wrapped one.
        self.logger.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.logger.setPlaceholderText(
            "Pick a job (or stay on Whole trackline), expand a product type and "
            "double-click ＋ Generate New… — or Run ▸ Default run on all.\n\n"
            "Everything the app and the pipelines do is reported here.")
        rv.addWidget(self.logger, 1)
        right.setMinimumWidth(300)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes(list(self.DEFAULT_SIZES))

    def _set_wrap(self, wrap: bool) -> None:
        self.logger.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth if wrap
                                    else QPlainTextEdit.LineWrapMode.NoWrap)

    def reset_layout(self) -> None:
        self.splitter.setSizes(list(self.DEFAULT_SIZES))
        self._tree_col_w = 0
        self._size_tree_column()
        self.log("layout reset (View ▸ Reset map view resets the map separately)")

    def _build_menus(self) -> None:
        bar = self.menuBar()
        fm = bar.addMenu("&File")
        self.act_save = QAction("Save workspace", self, triggered=self._save)
        self.act_save_as = QAction("Save workspace as…", self, triggered=self._save_as)
        fm.addAction(self.act_save)
        fm.addAction(self.act_save_as)
        fm.addSeparator()
        fm.addAction(QAction("Import navigation and orientation…", self,
                             triggered=self._import_nav))
        fm.addAction(QAction("Import video…", self, triggered=self._import_video))
        fm.addAction(QAction("Import new sensor channel…", self,
                             triggered=self._import_sensor))

        vm = bar.addMenu("&View")
        vm.addAction(QAction("Reset layout", self, triggered=self.reset_layout))
        vm.addAction(QAction("Reset map view", self,
                             triggered=self.trackline.reset_view))

        rm = bar.addMenu("&Run")
        self.act_default_all = QAction("Default run on all", self,
                                       triggered=self._default_run_all)
        f = self.act_default_all.font()
        f.setBold(True)
        self.act_default_all.setFont(f)
        rm.addAction(self.act_default_all)
        rm.addSeparator()
        # A POST-RUN step, never part of the default suite: it gathers what is
        # already on disk into one PDF.  The module is built in parallel, so the
        # action exists only when it can actually be imported.
        self.act_catalog = QAction("Build product catalog (PDF)", self,
                                   triggered=self._build_catalog)
        self.act_catalog.setEnabled(_catalog_builder() is not None)
        self.act_catalog.setToolTip(
            "Collect this job's existing products into survey/catalog/*.pdf"
            if self.act_catalog.isEnabled()
            else "catalog_builder.py is not available in this checkout")
        rm.addAction(self.act_catalog)

        hm = bar.addMenu("&Help")
        hm.addAction(QAction("About", self, triggered=self._about))

    # -- logging -------------------------------------------------------------
    def log(self, text: str) -> None:
        """Every log line, classified.

        Worker output arrives here, so this is where a failure becomes visible:
        a line starting with ``!!`` is a failure (red, counted), a line starting
        with the detail prefix is traceback noise (collapsed into the details
        buffer, replaced by one dim summary line).
        """
        for line in str(text).rstrip("\n").splitlines() or [""]:
            if line.startswith(DETAIL_PREFIX):
                self._details.append(line[len(DETAIL_PREFIX):])
                continue
            self._flush_details()
            if line.lstrip().startswith(FAIL_PREFIX.strip()):
                self._fails += 1
                self._last_fail = line.lstrip().lstrip("! ").strip()
                self.log_error(line.lstrip())
            elif line.lstrip().startswith(WARN_PREFIX):
                self._log_html(line, C_WARN)
            else:
                self.logger.appendPlainText(f"[{_stamp()}] {line}")

    def _flush_details(self) -> None:
        """Replace a buffered traceback with one dim, clickable-by-button line."""
        if not self._details:
            return
        self._last_details = "\n".join(self._details)
        n = len(self._details)
        self._details = []
        self.btn_details.setEnabled(True)
        self._log_html(f"   … {n} line(s) of detail collapsed — press "
                       "“details” above to read them", C_DIM)

    def _show_details(self) -> None:
        self._flush_details()
        if not self._last_details:
            self.log("no error details recorded")
            return
        self._log_html("--- error details ---", C_DIM)
        for line in self._last_details.splitlines():
            self._log_html("  " + line, C_DIM)
        self._log_html("--- end of details ---", C_DIM)

    def _log_html(self, text: str, colour: str) -> None:
        safe = (str(text).replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace("\n", "<br>"))
        self.logger.appendHtml(
            f'<span style="color:{colour}">[{_stamp()}] {safe}</span>')

    def log_error(self, text: str) -> None:
        self._flush_details()
        self._log_html(text, C_ERR)

    # -- workspace -----------------------------------------------------------
    def _load_workspace(self) -> None:
        try:
            self.ws = ConfigService.load_workspace(self.ws_json)
            self.ws_loaded = True
            self.log(f"loaded {self.ws_json.name} "
                     f"({len(self.ws.get('sensor_files') or [])} sensor file(s))")
        except Exception as exc:
            self.ws = {}
            self.ws_loaded = not self.ws_json.exists()   # absent is fine; broken is not
            self.log_error(f"could not load workspace: {exc}")

    # Keys this UI owns (imports) and keys load_workspace hands back as typed
    # objects — everything else on disk is owned by the backend / legacy app and
    # is refreshed from the file before saving so a save never clobbers it.
    _UI_KEYS = frozenset({"video_directory", "filename_datetime_format",
                          "workspace_path"})
    _TYPED_KEYS = frozenset({"navigation_file", "sensor_files", "pending_job",
                             "segment_history", "job_history", "threshold_history",
                             "annotation_config", "depth_source", "speed_source"})

    def _write_workspace(self, path: Path) -> None:
        data = dict(self.ws)
        # product_catalog writes the job store ("simple_jobs") straight into
        # workspace.json while this window is open; re-read those keys so
        # Save does not write back the stale copy loaded at startup.
        try:
            disk = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            disk = {}
        for key, value in disk.items():
            if key not in self._UI_KEYS and key not in self._TYPED_KEYS:
                data[key] = value
        # save_workspace()'s required keyword args, for a workspace that failed
        # to load or predates one of them.  Everything else passes through.
        for key, value in (("video_directory", ""), ("filename_datetime_format", ""),
                           ("navigation_file", None), ("sensor_files", []),
                           ("pending_job", None), ("next_job_id", 1),
                           ("segment_history", []), ("frame_rate", 1.0),
                           ("generate_sensor_tiffs", True), ("annotate_frames", False)):
            data.setdefault(key, value)
        # workspace_path names the .eprproj DIRECTORY (the current format that
        # main_window._restore_last_session and dive_setup both write); writing
        # the workspace.json file path here left a mixed-format field on disk.
        data["workspace_path"] = str(Path(path).parent)
        ConfigService.save_workspace(path, **data)

    def _save(self) -> None:
        if not getattr(self, "ws_loaded", True):
            # Saving now would replace an unreadable workspace.json with this
            # session's empty skeleton, destroying whatever is still in it.
            self.log_error(f"refusing to save over {self.ws_json.name}: it could not "
                           "be read at startup. Fix or move the file, or use "
                           "File ▸ Save workspace as… to write elsewhere.")
            return
        try:
            self._write_workspace(self.ws_json)
            self.log(f"saved {self.ws_json}")
        except Exception as exc:
            self.log_error(f"save failed: {exc}")

    def _save_as(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "Save workspace as",
                                           str(self.ws_dir.parent / self.ws_dir.name),
                                           "EPR project bundle (*.eprproj)")
        if not p:
            return
        try:
            root = Path(p)
            if root.suffix != ".eprproj":
                root = root.with_suffix(".eprproj")
            root.mkdir(parents=True, exist_ok=True)
            self.ws_dir, self.ws_path = root, str(root)
            self.ws_json = root / "workspace.json"
            self._write_workspace(self.ws_json)
            self.ws_loaded = True       # this file is ours now, and it is valid
            self.setWindowTitle(f"EPR Imaging — {root.name}")
            self.log(f"saved workspace as {self.ws_json}")
            _remember(str(root))
        except Exception as exc:
            self.log_error(f"save-as failed: {exc}")

    # -- imports -------------------------------------------------------------
    def _import_nav(self) -> None:
        dlg = NavImportDialog(self, str(self.ws_dir))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        nav = dlg.navigation_config()
        if nav is None:
            self.log_error("navigation import cancelled — no readable Renav CSV")
            return
        self.ws["navigation_file"] = nav
        span = (f"{nav.start_time} → {nav.end_time}" if nav.start_time else "unknown span")
        self.log(f"navigation imported: {Path(nav.latitude_source.csv_path).name} ({span})")
        if nav.altitude_source:
            self.log(f"altitude source: {Path(nav.altitude_source.csv_path).name}")
        self.log("remember to File ▸ Save workspace")

    def _import_video(self) -> None:
        dlg = VideoImportDialog(self, str(self.ws.get("video_directory") or ""),
                                str(self.ws.get("filename_datetime_format") or ""))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        directory, fmt = dlg.values()
        if not directory:
            self.log_error("video import cancelled — no directory")
            return
        self.ws["video_directory"] = directory
        self.ws["filename_datetime_format"] = fmt
        try:
            n = len([p for p in Path(directory).iterdir()
                     if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".m4v")])
        except Exception:
            n = 0
        self.log(f"video directory set: {directory} ({n} video file(s)), format '{fmt}'")

    def _import_sensor(self) -> None:
        dlg = SensorChannelDialog(self, str(self.ws_dir))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        vals = dlg.values()
        if not vals:
            self.log_error("sensor import cancelled — need CSV, timestamp and channel column")
            return
        csv_path, ts_col, channel = vals
        files = list(self.ws.get("sensor_files") or [])
        target = next((sf for sf in files
                       if str(getattr(sf, "csv_path", "")) == csv_path), None)
        if target is None:
            t0, t1 = self._ts_bounds(csv_path, ts_col)
            target = SensorFileConfig(csv_path=Path(csv_path), timestamp_column=ts_col,
                                      date_column=None, channels=[], start_time=t0,
                                      end_time=t1, no_header=False)
            files.append(target)
            self.log(f"new sensor file {Path(csv_path).name} ({t0} → {t1})")
        # Re-importing a channel (same source column) replaces the existing
        # entry instead of appending a second copy: two channels on one column
        # give the interp table two identically-named series and every
        # downstream product then reads a DataFrame where it wants a Series.
        dup = next((c for c in target.channels
                    if c.source_column == channel.source_column), None)
        if dup is not None:
            target.channels[target.channels.index(dup)] = channel
            self.log(f"channel on column '{channel.source_column}' re-imported — "
                     f"replaced the previous '{dup.display_name}' entry")
        else:
            target.channels.append(channel)
        self.ws["sensor_files"] = files
        self.log(f"channel '{channel.display_name}' [{channel.units}] ← column "
                 f"'{channel.source_column}', delay {channel.time_delay_s:g} s")

    def _ts_bounds(self, path: str, col: str):
        try:
            import pandas as pd
            s = pd.to_datetime(pd.read_csv(path, usecols=[col])[col],
                               format="mixed", errors="coerce")
            s = s.dropna()
            if not len(s):
                return None, None
            return s.min().to_pydatetime(), s.max().to_pydatetime()
        except Exception as exc:
            self.log_error(f"could not read timestamps from {Path(path).name}: {exc}")
            return None, None

    def _about(self) -> None:
        QMessageBox.about(
            self, "About EPR Imaging",
            "<b>EPR Imaging — simplified UI</b><br><br>"
            "Three panels: products, trackline, log.<br>"
            "Pick a job, expand a product type, generate; or Run ▸ Default run on all."
            f"<br><br>Workspace: {self.ws_dir}")

    # -- catalog access (all guarded) ---------------------------------------
    def _types(self) -> list:
        if pc is None:
            return []
        try:
            return list(pc.PRODUCT_TYPES)
        except Exception as exc:
            self.log_error(f"PRODUCT_TYPES unavailable: {exc}")
            return []

    def refresh_jobs(self) -> None:
        jobs: list = []
        if pc is not None:
            try:
                jobs = list(pc.load_jobs(self.ws_path))
            except Exception as exc:
                self.log_error(f"load_jobs failed: {exc}")
        if not jobs:
            jobs = [_StubJob()]
        self.jobs = jobs
        prev = self.job_combo.currentData()
        self.job_combo.blockSignals(True)
        self.job_combo.clear()
        for job in jobs:
            jid = getattr(job, "job_id", WHOLE)
            name = "Whole trackline" if jid == WHOLE else (getattr(job, "name", "") or str(jid))
            self.job_combo.addItem(name, jid)
        idx = self.job_combo.findData(prev)
        self.job_combo.setCurrentIndex(max(0, idx))
        self.job_combo.blockSignals(False)
        self._on_job_changed()

    def current_job(self):
        jid = self.job_combo.currentData()
        for job in self.jobs:
            if getattr(job, "job_id", None) == jid:
                return job
        return self.jobs[0] if self.jobs else _StubJob()

    def _job_menu(self, pos) -> None:
        """Rename / delete the selected job (never the whole trackline)."""
        job = self.current_job()
        jid = getattr(job, "job_id", WHOLE)
        menu = QMenu(self)
        if jid == WHOLE:
            menu.addAction("(select a job to rename or delete)").setEnabled(False)
        else:
            menu.addAction("Rename job…").triggered.connect(self._rename_job)
            menu.addAction("Delete job…").triggered.connect(self._delete_job)
        menu.addSeparator()
        catalog = menu.addAction("Build product catalog (PDF)")
        catalog.setEnabled(_catalog_builder() is not None and not self._busy)
        catalog.triggered.connect(self._build_catalog)
        menu.exec(self.job_combo.mapToGlobal(pos))

    def _rename_job(self) -> None:
        job = self.current_job()
        jid = getattr(job, "job_id", WHOLE)
        if pc is None or jid == WHOLE:
            return
        name, ok = QInputDialog.getText(self, "Rename job", "Job name:",
                                        text=str(getattr(job, "name", jid)))
        if not ok:
            return
        try:
            renamed = pc.rename_job(self.ws_path, jid, name)
        except Exception as exc:
            self.log_error(f"rename failed: {exc}")
            return
        self.log(f"renamed {jid} to '{getattr(renamed, 'name', name)}'")
        self.refresh_jobs()
        idx = self.job_combo.findData(jid)
        if idx >= 0:
            self.job_combo.setCurrentIndex(idx)

    def _delete_job(self) -> None:
        job = self.current_job()
        jid = getattr(job, "job_id", WHOLE)
        if pc is None or jid == WHOLE:
            return
        label = str(getattr(job, "name", jid))
        if QMessageBox.question(
                self, "Delete job",
                f"Delete '{label}'?\n\nIts products are never deleted — if it "
                "has any, the job is kept and you can rename it instead.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            pc.delete_job(self.ws_path, jid)
        except Exception as exc:
            self.log_error(f"delete refused: {exc}")
            return
        self.log(f"deleted {label}")
        self.job_combo.setCurrentIndex(0)
        self.refresh_jobs()

    def _on_job_changed(self, *_a) -> None:
        job = self.current_job()
        self.trackline.set_job_intervals(getattr(job, "intervals", []) or [])
        self._refresh_tree()

    def _interp_ready(self) -> bool:
        """True when interp_full.csv already exists for this workspace.

        track_polyline() builds it through the whole pipeline when it is
        missing — minutes of work whose log goes to stdout.  Never let that
        happen on the GUI thread; the user builds it from Run ▸ Default run on
        all, which reports into the logger on a worker thread.
        """
        try:
            path = Path(pc.interp_path(self.ws_path))               # type: ignore[union-attr]
        except Exception:
            return True                     # unknown: let track_polyline decide
        if path.is_file() and path.stat().st_size > 0:
            return True
        self.log_error(f"no interp_full.csv yet ({path}) — import navigation and "
                       "sensors, then Run ▸ Default run on all to build it")
        return False

    def _load_track(self) -> None:
        arr = None
        if pc is not None and self._interp_ready():
            try:
                arr = pc.track_polyline(self.ws_path)
            except Exception as exc:
                self.log_error(f"track_polyline failed: {exc}")
        self.track = np.asarray(arr, dtype=float) if arr is not None and len(arr) else None
        self.trackline.set_track(self.track)
        if self.track is not None:
            self.log(f"trackline: {len(self.track):,} vertices")

    # -- product tree --------------------------------------------------------
    def _refresh_tree(self) -> None:
        expanded = {it.data(0, ROLE)[1] for it in self._top_items() if it.isExpanded()}
        self.tree.clear()
        for t in self._types():
            key = str(getattr(t, "key", ""))
            item = QTreeWidgetItem([str(getattr(t, "label", key) or key)])
            item.setData(0, ROLE, ("type", key, t))
            item.setToolTip(0, (
                "One instance per sampling run: this product is built from the "
                "frames, so a different sampling technique gives a different "
                "product. Instances are labelled by their sampling technique."
                if _sampling_dependent(key) else
                "Computed once per job from the 1 Hz nav/sensor table. It does "
                "not depend on the sampling technique, so it is listed once no "
                "matter how many sampling runs this job has."))
            self.tree.addTopLevelItem(item)
            item.addChild(QTreeWidgetItem(["…"]))
            if key in expanded:
                item.setExpanded(True)          # triggers _on_expanded → populate
        if not self.tree.topLevelItemCount():
            self.tree.addTopLevelItem(QTreeWidgetItem(["(no product types available)"]))
        elif not self._job_is_whole():
            # Under a job, every type can read "(none yet)" while the dive's
            # products sit one combo entry away.  Say where they are, once.
            hint = QTreeWidgetItem(
                ["ⓘ whole-dive products live under “Whole trackline”"])
            hint.setDisabled(True)
            hint.setToolTip(0, "This job only lists products generated for its "
                               "own intervals. Switch the Job selector to "
                               "“Whole trackline” to see the dive-wide ones.")
            self.tree.addTopLevelItem(hint)
        self._size_tree_column()

    def _job_is_whole(self) -> bool:
        return getattr(self.current_job(), "job_id", WHOLE) == WHOLE

    def _size_tree_column(self) -> None:
        """Fit column 0 to its contents and never let it shrink again.

        ``resizeColumnToContents`` alone made the column flap between 259 and
        173 px as jobs were switched, clipping the type names on launch.
        """
        try:
            needed = int(self.tree.sizeHintForColumn(0)) + 12
        except Exception:                                       # pragma: no cover
            return
        self._tree_col_w = max(self._tree_col_w, needed, 180)
        self.tree.setColumnWidth(0, self._tree_col_w)

    def _top_items(self) -> list[QTreeWidgetItem]:
        out = []
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if isinstance(it.data(0, ROLE), tuple):
                out.append(it)
        return out

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        data = item.data(0, ROLE)
        if not (isinstance(data, tuple) and data[0] == "type"):
            return
        if item.childCount() == 1 and item.child(0).data(0, ROLE) is None:
            self._populate(item, data[2])

    def _populate(self, item: QTreeWidgetItem, ptype) -> None:
        item.takeChildren()
        gen = QTreeWidgetItem([GEN_LABEL])
        gen.setData(0, ROLE, ("gen", str(getattr(ptype, "key", "")), ptype))
        item.addChild(gen)
        job = self.current_job()
        try:
            instances = list(ptype.discover(self.ws_path, job) or [])
        except Exception as exc:
            child = QTreeWidgetItem([f"(discover failed: {exc})"])
            child.setDisabled(True)
            item.addChild(child)
            msg = f"{getattr(ptype, 'key', '?')}.discover failed: {exc}"
            if msg not in self._seen_errors:      # every refresh would repeat it
                self._seen_errors.add(msg)
                self.log_error(msg)
            return
        # ProductType.discover() swallows its own exceptions and reports them on
        # .last_error; without this a broken discover looks like "no products".
        problem = getattr(ptype, "last_error", None)
        if problem:
            broken = QTreeWidgetItem([f"(discovery error: {problem})"])
            broken.setDisabled(True)
            item.addChild(broken)
            msg = f"{getattr(ptype, 'key', '?')}.discover error: {problem}"
            if msg not in self._seen_errors:
                self._seen_errors.add(msg)
                self.log_error(msg)
        for inst in instances:
            child = QTreeWidgetItem([str(getattr(inst, "label", getattr(inst, "path", "?")))])
            child.setData(0, ROLE, ("inst", inst, ptype))
            child.setToolTip(0, f"{child.text(0)}\n{getattr(inst, 'path', '')}")
            item.addChild(child)
        if not instances:
            none_it = QTreeWidgetItem(
                ["(none yet for this job — see Whole trackline)"
                 if not self._job_is_whole() else "(none yet)"])
            none_it.setDisabled(True)
            item.addChild(none_it)
        self._size_tree_column()

    def _tree_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        data = item.data(0, ROLE) if item else None
        if not isinstance(data, tuple):
            return
        menu = QMenu(self)
        if data[0] == "inst":
            inst = data[1]
            menu.addAction("View").triggered.connect(lambda: self._view(inst))
            menu.addAction("Open containing folder").triggered.connect(
                lambda: self._reveal(inst))
        elif data[0] == "gen":
            ptype = data[2]
            menu.addAction("Generate New…").triggered.connect(
                lambda: self._generate(ptype))
        else:
            return
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _on_double_click(self, item: QTreeWidgetItem, _col: int) -> None:
        data = item.data(0, ROLE)
        if not isinstance(data, tuple):
            return
        if data[0] == "inst":
            self._view(data[1])
        elif data[0] == "gen":
            self._generate(data[2])

    def _view(self, inst) -> None:
        try:
            viewer = ProductViewer(inst, self.track, self)
            viewer.show()
            # Drop viewers the user has closed; each holds decoded rasters.
            for stale in [v for v in self._viewers if not v.isVisible()]:
                self._viewers.remove(stale)
                stale.deleteLater()
            self._viewers.append(viewer)
            self.log(f"viewing {getattr(inst, 'label', '')}")
        except Exception as exc:
            self.log_error(f"viewer failed: {exc}")

    def _reveal(self, inst) -> None:
        path = str(getattr(inst, "path", "") or "")
        if path and _open_folder(path):
            self.log(f"opened folder for {Path(path).name}")
        else:
            self.log_error(f"could not open containing folder for '{path}'")

    # -- runs ----------------------------------------------------------------
    def _generate(self, ptype) -> None:
        if self._busy:
            self.log_error("a task is already running")
            return
        label = str(getattr(ptype, "label", getattr(ptype, "key", "product")))
        schema: list = []
        try:                    # schema(ws) fills workspace-dependent choices
            schema = list(ptype.schema(self.ws_path))
        except Exception:
            try:
                schema = list(getattr(ptype, "settings_schema", []) or [])
            except Exception as exc:
                self.log_error(f"settings_schema unavailable for {label}: {exc}")
        job = self.current_job()
        scope = ("Whole trackline" if getattr(job, "job_id", WHOLE) == WHOLE
                 else str(getattr(job, "name", "")))
        dlg = GenerateDialog(self, label, schema, busy=self._busy, scope=scope)
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.mode:
            return
        if dlg.mode == "default":
            try:
                settings = dict(ptype.defaults(self.ws_path))
            except Exception:
                settings = dlg.defaults()
        else:
            settings = dlg.values()
        how = ("defaults" if dlg.mode == "default"
               else json.dumps(settings, default=str))
        self.log(f"generate {label} [{scope}] — {how}")

        def task(log_fn):
            return ptype.generate(self.ws_path, job, settings, log_fn)
        self._start(f"generating {label}", task)

    def _default_run_all(self) -> None:
        if self._busy:
            self.log_error("a task is already running")
            return
        if pc is None:
            self.log_error("product_catalog unavailable — cannot run")
            return
        job = self.current_job()
        whole = getattr(job, "job_id", WHOLE) == WHOLE
        self.log("default run on all — " + ("whole trackline" if whole
                                            else f"job {getattr(job, 'name', '')}"))

        def task(log_fn):
            return pc.default_run_all(self.ws_path, log_fn, job=None if whole else job)
        self._start("default run on all", task)

    def _build_catalog(self) -> None:
        """Post-run step: gather the selected job's products into one PDF."""
        if self._busy:
            self.log_error("a task is already running")
            return
        build = _catalog_builder()
        if build is None:
            self.log_error("catalog_builder.py is not available in this checkout "
                           "— product catalog cannot be built")
            return
        job = self.current_job()
        job_id = getattr(job, "job_id", WHOLE)
        scope = "Whole trackline" if job_id == WHOLE else str(getattr(job, "name", job_id))
        self.log(f"building product catalog for {scope}")

        def task(log_fn):
            return build(self.ws_path, job_id=job_id, log=log_fn)
        self._start(f"product catalog — {scope}", task)

    def _start(self, title: str, fn) -> None:
        self._task_title = title
        self._fails, self._last_fail = 0, ""
        self._set_busy(True, title)
        self._thread = QThread(self)
        self._worker = Worker(fn)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        # Bound methods of self (a main-thread QObject) so the auto-connection
        # is queued back to the GUI thread; lambdas would run in the worker.
        self._worker.line.connect(self.log)
        self._worker.done.connect(self._on_task_done)
        self._worker.failed.connect(self._on_task_failed)
        self._thread.start()

    def _on_task_done(self, result) -> None:
        self._finish(result, None)

    def _on_task_failed(self, cause: str, tb: str) -> None:
        self._finish(None, cause, tb)

    def _finish(self, result, cause: str | None, tb: str = "") -> None:
        title = getattr(self, "_task_title", "task")
        if cause:
            # One compact red line; the traceback goes behind the details button.
            self.log_error(f"{title} FAILED — {cause}")
            self._details = tb.rstrip().splitlines()
            self._flush_details()
            verdict = f"{title} FAILED — {cause}"
        elif self._fails:
            # The backend's own last red line carries the exact tally; repeating
            # a count here would risk contradicting it.
            self.log_error(f"{title} finished WITH FAILURES — see the red lines "
                           "above (press “details” for the last traceback)")
            verdict = (self._last_fail or f"{title} finished with failures")[:160]
        else:
            verdict = f"{title} finished"
            self.log(verdict
                     + (f" → {getattr(result, 'label', result)}" if result else ""))
        if self._worker is not None:
            for signal in (self._worker.line, self._worker.done, self._worker.failed):
                try:
                    signal.disconnect()
                except (RuntimeError, TypeError):
                    pass
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._thread, self._worker = None, None
        self._set_busy(False, verdict, bad=bool(cause) or bool(self._fails))
        self.refresh_jobs()
        if self.track is None:      # a run may have just built interp_full.csv
            self._load_track()

    def _set_busy(self, busy: bool, message: str, bad: bool = False) -> None:
        self._busy = busy
        self.act_default_all.setEnabled(not busy)
        if getattr(self, "act_catalog", None) is not None:
            self.act_catalog.setEnabled(not busy and _catalog_builder() is not None)
        self.trackline.set_busy(busy)
        # The status bar is the last thing a user looks at: it must not say
        # "ready" after half the suite failed.
        self.statusBar().setStyleSheet(
            f"color: {C_ERR}; font-weight: bold;" if (bad and not busy) else "")
        self.statusBar().showMessage(f"running: {message}" if busy else message)

    # -- jobs ----------------------------------------------------------------
    def _create_job(self, staged: list) -> None:
        if pc is None:
            self.log_error("product_catalog unavailable — cannot create job")
            return
        job = self.current_job()
        base = None if getattr(job, "job_id", WHOLE) == WHOLE else job
        ivs = [_mk_interval(t0, t1) for t0, t1 in staged]
        try:
            new_job = pc.create_job(self.ws_path, ivs, base_job=base)
        except Exception as exc:
            self.log_error(f"create_job failed: {exc}")
            return
        self.log(f"created {getattr(new_job, 'name', '?')} "
                 f"({len(ivs)} new interval(s)"
                 + (f", derived from {getattr(base, 'name', '')}" if base else "") + ")")
        self.trackline.job_created()
        self.refresh_jobs()
        idx = self.job_combo.findData(getattr(new_job, "job_id", None))
        if idx >= 0:
            self.job_combo.setCurrentIndex(idx)

    def keyPressEvent(self, event) -> None:
        """Escape cancels a half-made interval pick (the map has no focus of its own)."""
        if event.key() == Qt.Key.Key_Escape:
            if self.trackline.cancel_pick():
                event.accept()
                return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            # quit() only ends the thread's event loop; a generate already inside
            # the worker cannot be interrupted.  Ask, then let go of it safely:
            # the QThread must not be destroyed (nor keep signalling destroyed
            # widgets) while its callable is still on the CPU.
            answer = QMessageBox.question(
                self, "Task still running",
                f"'{self._task_title}' is still running and cannot be "
                "interrupted.\n\nClose anyway? It will finish in the background "
                "and its remaining log lines will be lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.log("closing while a task runs — detaching it")
            if self._worker is not None:
                for signal in (self._worker.line, self._worker.done,
                               self._worker.failed):
                    try:
                        signal.disconnect()
                    except (RuntimeError, TypeError):
                        pass
            self._thread.setParent(None)         # survive this window's deletion
            _reap_orphans()                      # release earlier, now-idle ones
            _ORPHANS.add((self._thread, self._worker))
            self._thread.quit()
            self._thread, self._worker = None, None
        elif self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread, self._worker = None, None
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _remember(path: str) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps({"last_workspace": str(path)}, indent=1))
    except Exception:
        pass


def _last_workspace() -> str:
    try:
        return str(json.loads(SETTINGS_FILE.read_text()).get("last_workspace") or "")
    except Exception:
        return ""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    ws = argv[1] if len(argv) > 1 else ""
    if not ws:
        last = _last_workspace()
        ws = QFileDialog.getExistingDirectory(
            None, "Open EPR workspace (.eprproj)",
            last or str(Path.home())) or ""
    if not ws:
        print("no workspace selected", file=sys.stderr)
        return 1
    _remember(ws)
    win = MainWindow(ws)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
