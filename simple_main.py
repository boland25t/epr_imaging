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
import re
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import (QAbstractTableModel, QDateTime, QModelIndex, QObject,
                            Qt, QThread, QTimeZone, QUrl, Signal)
from PySide6.QtGui import (QAction, QDesktopServices, QFont, QIcon, QImage, QKeySequence,
                           QPalette, QPixmap)
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
import matplotlib.ticker                                                 # noqa: E402
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


def _on_light(dark_hex: str, light_hex: str) -> str:
    """Pick a chrome colour for the current Qt palette (canvas colours are dark-only).

    The palette above is tuned for the dark map/log background; plain Qt chrome
    (labels, status bar) sits on the system window colour, which is light by
    default, where those colours drop to ~2:1 contrast.
    """
    app = QApplication.instance()
    if app is None:
        return dark_hex
    dark = app.palette().color(QPalette.ColorRole.Window).lightness() < 128
    return dark_hex if dark else light_hex

# Worker log-line conventions, mirrored from product_catalog (kept as literals
# so the UI still classifies correctly when the backend is unavailable).
FAIL_PREFIX = getattr(pc, "LOG_FAIL", "!! ") if pc else "!! "
DETAIL_PREFIX = getattr(pc, "LOG_DETAIL", "  · ") if pc else "  · "
WARN_PREFIX = getattr(pc, "LOG_WARN", "note: ") if pc else "note: "
CLICK_TOLERANCE_PX = 30.0               # a click further than this misses the track

ROLE = Qt.ItemDataRole.UserRole
SETTINGS_FILE = Path.home() / ".epr_simple_ui.json"
GEN_LABEL = "+ Generate new…"
WHOLE = getattr(pc, "WHOLE_TRACKLINE", "__whole__") if pc else "__whole__"
RUN_ALL_LABEL = "Run all products (defaults)"

def _asset(name: str) -> Path:
    """First existing copy of a bundled image; the repo root copy is preferred.

    The WHOI images have lived beside this file and (after the plan-09 archive
    move) under archive/legacy_ui/; look in each so the branding never silently
    disappears.  Returns the preferred path even when none exists.
    """
    here = Path(__file__).resolve().parent
    candidates = [here / name, here / "assets" / name, here / "archive" / "legacy_ui" / name]
    return next((p for p in candidates if p.is_file()), candidates[0])


LOGO_LONG = _asset("whoilogolong.png")    # WHOI wordmark, opaque white bg
LOGO_MARK = _asset("whoilogo.png")        # WHOI mark, transparent bg
LOGO_HEIGHT_PX = 40

#: Rehearsal/seeding tags that live in on-disk labels (workspace files are never
#: rewritten by the UI); stripped at display time only.
_LABEL_TAG_RE = re.compile(r"\s*\[SEEDED FROM REAL [^\]]*\]\s*", re.IGNORECASE)


def _display_label(text) -> str:
    return _LABEL_TAG_RE.sub(" ", str(text)).strip()


def _wordmark_pixmap(height: int, dark: bool, text_rgb=(230, 230, 230),
                     dpr: float = 1.0) -> QPixmap | None:
    """WHOI wordmark with its white background made transparent.

    On a dark palette the navy lettering is recoloured to the palette's text
    colour (the saturated teal/blue waves keep their colour).  None when the
    asset is missing, so the UI never depends on it.
    """
    if not LOGO_LONG.is_file():
        return None
    try:
        img = QImage(str(LOGO_LONG)).convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = img.width(), img.height()
        a = (np.frombuffer(img.constBits(), np.uint8)
             .reshape(h, img.bytesPerLine())[:, :w * 4].reshape(h, w, 4).astype(np.float32))
        rgb = a[..., :3]
        alpha = 255.0 - rgb.min(axis=2)
        rgb = np.clip((rgb - (255.0 - alpha[..., None]))
                      / np.maximum(alpha, 1.0)[..., None] * 255.0, 0, 255)
        if dark:
            sat = rgb.max(axis=2) - rgb.min(axis=2)
            rgb[(rgb.max(axis=2) < 130) & (sat < 70)] = text_rgb
        out = np.ascontiguousarray(np.dstack([rgb, alpha]).astype(np.uint8))
        q = QImage(out.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        pm = QPixmap.fromImage(q).scaledToHeight(
            int(round(height * dpr)), Qt.TransformationMode.SmoothTransformation)
        pm.setDevicePixelRatio(dpr)
        return pm
    except Exception:
        return None


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


def _open_external(path: str) -> str | None:
    """Hand a file to the desktop's default viewer.  None on success, else why not.

    Qt's openUrl delegates to xdg-open on Linux, which WSL often lacks; it then
    returns False and nothing happens.  Fall back to the Windows shell through
    wslpath, the same route _open_folder takes.
    """
    try:
        if QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            return None
    except Exception:
        pass
    try:
        win = subprocess.run(["wslpath", "-w", path], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        if win:
            subprocess.Popen(["explorer.exe", win], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return None
    except Exception:
        pass
    return ("no desktop viewer is registered (xdg-open / wslview / explorer.exe "
            "not found) — use “Open containing folder” and open the file from there")


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
    if n * dt.itemsize <= 64_000_000:
        mm = np.memmap(path, mode="r", dtype=dt, offset=offset, shape=(n,))
        try:
            sub = np.asarray(mm[::max(1, n // cap)][:cap])
        finally:
            del mm
    else:
        # Contiguous blocks spread end to end: bounded I/O, full extent.
        # Metashape writes vertices in depth-map order, so one block is one
        # small patch of seafloor — many small blocks read as a cloud,
        # a few big ones read as a handful of blobs.  The blocks are fetched
        # with parallel pread: on the 9p /mnt drives each read is a ~20 ms
        # round trip, and serial page faults made a cold 3 GB cloud take ~6 s.
        blocks = 256
        per = max(1, min(n, cap // blocks))
        starts = sorted({min(n - per, int(b * (n - per) / (blocks - 1)))
                         for b in range(blocks)})
        nbytes = per * dt.itemsize
        fd = os.open(path, os.O_RDONLY)
        try:
            def _read(s: int) -> np.ndarray:
                buf = os.pread(fd, nbytes, offset + s * dt.itemsize)
                usable = len(buf) - len(buf) % dt.itemsize
                return np.frombuffer(buf[:usable], dtype=dt)
            with ThreadPoolExecutor(max_workers=16) as pool:
                sub = np.concatenate(list(pool.map(_read, starts)))
        finally:
            os.close(fd)
    xyz = np.stack([sub["x"].astype("f8"), sub["y"].astype("f8"),
                    sub["z"].astype("f8")], axis=1)
    rgb = None
    if all(c in dt.names for c in ("red", "green", "blue")):
        rgb = np.stack([sub["red"], sub["green"], sub["blue"]],
                       axis=1).astype("f4") / 255.0
    return xyz, rgb, n, int(counts.get("face", 0))


_OBJ_BLOCKS = 384                  # byte-offset samples across an OBJ
_OBJ_BLOCK_BYTES = 48_000          # bytes read at each sample


def _obj_block_lines(buf: bytes, first: bool) -> list[bytes]:
    """Complete lines of one block (a partial first line is dropped unless at 0)."""
    lines = buf.split(b"\n")
    if not first:
        lines = lines[1:]
    return lines[:-1] if len(lines) > 1 else []


def _sample_obj(path: str, cap: int = MESH_SAMPLE):
    """(xyz, None, n_vertices_est, n_faces_est) from a bounded OBJ read.

    Wavefront OBJ is ASCII.  Walking it line by line cost 5–96 s on the GUI
    thread and — because Metashape's coloured ``v`` block is larger than any
    sane byte budget — still covered only part of the mesh.  Instead read
    ~384 small blocks spread across the whole file with parallel pread and keep
    the complete ``v`` lines: a few MB of I/O, full spatial extent.  Blocks
    that land in the vt/vn/f sections simply contribute nothing.  Vertex and
    face totals are ESTIMATES scaled from the sampled bytes.
    """
    size = Path(path).stat().st_size
    fd = os.open(path, os.O_RDONLY)
    try:
        if size <= _OBJ_BLOCKS * _OBJ_BLOCK_BYTES:
            offsets = [0]
            nbytes = size
        else:
            offsets = [int(b * (size - _OBJ_BLOCK_BYTES) / (_OBJ_BLOCKS - 1))
                       for b in range(_OBJ_BLOCKS)]
            nbytes = _OBJ_BLOCK_BYTES

        def _read(off: int) -> list[bytes]:
            return _obj_block_lines(os.pread(fd, nbytes, off), off == 0)
        with ThreadPoolExecutor(max_workers=16) as pool:
            blocks = list(pool.map(_read, offsets))
    finally:
        os.close(fd)
    v_lines: list[bytes] = []
    sampled = v_bytes = f_bytes = 0
    n_f_seen = 0
    for lines in blocks:
        for ln in lines:
            sampled += len(ln) + 1
            if ln[:2] == b"v ":
                v_lines.append(ln)
                v_bytes += len(ln) + 1
            elif ln[:2] == b"f ":
                n_f_seen += 1
                f_bytes += len(ln) + 1
    scale = size / max(1, sampled)
    n_v_est = int(round(len(v_lines) * scale))
    n_f_est = int(round(n_f_seen * scale))
    step = max(1, len(v_lines) // cap)
    pts = []
    for ln in v_lines[::step][:cap]:
        part = ln.split()
        if len(part) >= 4:
            try:
                pts.append((float(part[1]), float(part[2]), float(part[3])))
            except ValueError:
                pass
    return np.asarray(pts, dtype="f8"), None, n_v_est, n_f_est


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


def _metres_axes(ax) -> None:
    """Plain "1,095,300" tick labels instead of matplotlib's "1.095 / 1e6" offset."""
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:,.0f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


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


class _TrimSpin(QDoubleSpinBox):
    """A float spin box that does not pad with zeros (0.2500 -> 0.25, 8.0000 -> 8)."""

    def textFromValue(self, v: float) -> str:
        return (f"{v:.{self.decimals()}f}".rstrip("0").rstrip(".")) or "0"


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


#: Main windows opened by File ▸ New/Open workspace (they have no parent).
_WINDOWS: list = []


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
        # BaseException, not Exception: a SystemExit (argparse inside a helper)
        # or KeyboardInterrupt escaping run() tears the QThread down under Qt
        # and aborts the whole process with a core dump and no message.
        try:
            self.done.emit(self._fn(self.line.emit))
        except BaseException as exc:                     # noqa: BLE001
            try:
                summary = (pc.one_line(exc) if pc is not None
                           else f"{type(exc).__name__}: {exc}")
            except Exception:
                summary = f"{type(exc).__name__}: {exc}"
            self.failed.emit(summary, traceback.format_exc())


# ---------------------------------------------------------------------------
# import dialogs
# ---------------------------------------------------------------------------

class NavImportDialog(QDialog):
    """Renav CSV (headerless) + optional DPA altitude CSV → NavigationConfig."""

    COLS = [("date", 0), ("clock", 1), ("latitude", 2), ("longitude", 3),
            ("depth", 4), ("heading", 5), ("pitch", 6), ("roll", 7)]

    def __init__(self, parent=None, start_dir: str = "", current=None) -> None:
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
        self.preview_note.setStyleSheet(f"color: {_on_light(C_DIM, '#57606a')};")
        form.addRow("First rows (column numbers as headers):", self.preview)
        form.addRow("", self.preview_note)

        self.spins: dict[str, QSpinBox] = {}
        for name, default in self.COLS:
            sp = QSpinBox()
            sp.setRange(0, 200)
            sp.setValue(default)
            sp.valueChanged.connect(self._highlight_columns)
            self.spins[name] = sp
            form.addRow(f"    {name.capitalize()} column:", sp)

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
        if current is not None:
            self._prefill(current)

    def _prefill(self, nav) -> None:
        """Show what is imported now, so a re-import starts from it, not from blank."""
        try:
            lat = nav.latitude_source
            sources = {"latitude": lat, "longitude": nav.longitude_source,
                       "depth": nav.depth_source, "heading": nav.heading_source,
                       "pitch": nav.pitch_source, "roll": nav.roll_source}
            for key, col in (("date", getattr(lat, "date_column", None)),
                             ("clock", getattr(lat, "timestamp_column", None))):
                if col is not None and str(col).strip().isdigit():
                    self.spins[key].setValue(int(col))
            for key, src in sources.items():
                col = getattr(src, "value_column", None) if src is not None else None
                if col is not None and str(col).strip().isdigit():
                    self.spins[key].setValue(int(col))
            self.negate.setChecked(bool(getattr(nav, "negate_depth", False)))
            alt = getattr(nav, "altitude_source", None)
            if alt is not None:
                self.alt_edit.setText(str(alt.csv_path))
                self.alt_date.setText(str(alt.date_column or "DATE"))
                self.alt_time.setText(str(alt.timestamp_column or "TIME"))
                self.alt_value.setText(str(alt.value_column or "ALTITUDE(m)"))
            self.nav_edit.setText(str(lat.csv_path))       # loads the preview
        except Exception:
            pass

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
        self.delay_spin.setToolTip("Positive = the sensor reading lags the vehicle "
                                   "position by this many seconds.")
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
            head = QLabel(f"<b>{type_label}</b><br>Job: <b>{scope}</b>")
            head.setTextFormat(Qt.TextFormat.RichText)
            outer.addWidget(head)
        form = QFormLayout()
        outer.addLayout(form)

        for entry in list(schema or []):
            entry = list(entry) + [None] * (5 - len(list(entry)))
            key, label, kind, default, extra = entry[:5]
            kind = (kind or "str").lower()
            if kind == "float":
                w = _TrimSpin()
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
                # Long paths (detector weights) were left-clipped to the tail.
                w.setToolTip(w.text())
                w.setMinimumWidth(260)
                w.setCursorPosition(0)
            self._widgets[str(key)] = (kind, w)
            self._defaults[str(key)] = default
            form.addRow(str(label or key) + ":", w)

        if not self._widgets:
            form.addRow(QLabel("This product takes no settings."))

        btns = QHBoxLayout()
        self.btn_default = QPushButton("Run with defaults")
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
        # Enter must never discard what was typed: the explicit-settings button
        # is the default; "defaults" needs a deliberate click.
        self.btn_default.setAutoDefault(False)
        self.btn_cancel.setAutoDefault(False)
        self.btn_custom.setAutoDefault(True)
        self.btn_custom.setDefault(True)
        btns.addStretch(1)
        for b in (self.btn_default, self.btn_custom, self.btn_cancel):
            btns.addWidget(b)
        outer.addLayout(btns)
        if not self._widgets:
            # Nothing to edit: two identical run buttons would only confuse.
            self.btn_custom.hide()
            self.btn_default.setText("Run")
            self.btn_default.setAutoDefault(True)
            self.btn_default.setDefault(True)

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
        self.readout.setWordWrap(True)          # never clipped in a narrow map pane
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight
                                  | Qt.AlignmentFlag.AlignVCenter)
        self.readout.setStyleSheet(f"color: {_on_light(MUT, '#57606a')};")
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
        self.btn_typed = btn_typed = QPushButton("Add typed interval")
        btn_typed.clicked.connect(self._add_typed)
        typed.addWidget(QLabel("From (UTC):"))
        typed.addWidget(self.dt_from)
        typed.addWidget(QLabel("To (UTC):"))
        typed.addWidget(self.dt_to)
        typed.addWidget(btn_typed)
        typed.addStretch(1)
        lay.addLayout(typed)

        row = QHBoxLayout()
        # "Add interval to job" read as if it changed the selected job; it only
        # moves the pending interval onto the staging list for a NEW job.
        self.btn_stage = QPushButton("Stage interval")
        self.btn_stage.setToolTip("Move the pending interval (cyan) onto the staged "
                                  "list (green) for the next new job.")
        self.btn_create = QPushButton("Create job")
        self.btn_create.setToolTip("Create a NEW job from the staged intervals "
                                   "(you are asked for its name).")
        self.btn_clear = QPushButton("Clear")
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
        # Without a track the From/To editors show Qt's 2000-01-01 default.
        for w in (self.dt_from, self.dt_to, self.btn_typed, self.btn_stage,
                  self.btn_create):
            w.setEnabled(self.track is not None)
        self._refresh_readout()
        self._redraw()
        self._zoom_to_job()

    def set_job_intervals(self, intervals) -> None:
        self.job_intervals = [_iv_bounds(iv) for iv in (intervals or [])]
        self.cancel_pick(quiet=True)    # a half-made pick must not outlive a job switch
        # A short job interval is one dot on a 16 h track: zoom to the job.
        self._zoomed = False
        self._redraw()
        self._zoom_to_job()

    def _zoom_to_job(self) -> None:
        if self.track is None or not self.job_intervals:
            return
        m = np.zeros(len(self.track), dtype=bool)
        for t0, t1 in self.job_intervals:
            m |= (self.track[:, 0] >= t0) & (self.track[:, 0] <= t1)
        if m.sum() < 2:
            return
        xs, ys = self.track[m, 1], self.track[m, 2]
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min(), 60.0) * 0.6
        self.ax.set_xlim(cx - half, cx + half)
        self.ax.set_ylim(cy - half, cy + half)
        self._zoomed = True
        self.canvas.draw_idle()

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
        idle = ("wheel = zoom · right-drag = pan · left-click twice = interval"
                "  ·  orange = job, cyan = pending, green = staged"
                if self.track is not None else "")
        self.readout.setText("   ·   ".join(bits) or idle)
        self.readout.setStyleSheet(
            f"color: {_on_light(C_MARK, '#c2185b') if self._first is not None else _on_light(MUT, '#57606a')};")

    def set_busy(self, busy: bool) -> None:
        self.btn_create.setEnabled(not busy and self.track is not None)

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
        _metres_axes(ax)
        if self.track is None:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.text(0.5, 0.5, "No trackline yet\nFile ▸ Import navigation and orientation…\n"
                    f"then Run ▸ {RUN_ALL_LABEL}", color=MUT, ha="center",
                    va="center", transform=ax.transAxes, fontsize=10, linespacing=1.6)
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
            v = self._df.iat[index.row(), index.column()]
            if isinstance(v, (float, np.floating)):
                return f"{float(v):.6g}"            # not 16-digit float noise
            return str(v)
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
    FIT_PX = 900                   # small rasters are scaled UP to about this
    CSV_ROWS = 500
    MAX_LIVE_3D = 4                # 3-D preview tabs kept built at once (~66 MB each)

    #: Short tab names for the files a photogrammetry chunk carries.
    _SHORT = {"orthomosaic.tif": "ortho", "dem.tif": "DEM", "report.pdf": "report",
              "dense.ply": "dense cloud", "sparse.ply": "sparse cloud",
              "mesh.obj": "mesh", "project.psx": "project (.psx)"}
    _PREVIEW_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    #: Emitted from the mesh-sampling thread; queued back to the GUI thread.
    sampleReady = Signal(object, object)

    def __init__(self, instance, track=None, parent=None) -> None:
        super().__init__(parent)
        # Closed viewers free their decoded rasters and 3-D figures at once,
        # instead of waiting for the next viewer to be opened.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        label = _display_label(getattr(instance, "label", "product"))
        tkey = str(getattr(instance, "type_key", "") or "")
        try:
            tlabel = pc.product_type(tkey).label if (pc is not None and tkey) else tkey
        except Exception:
            tlabel = tkey
        self.setWindowTitle(f"{tlabel} — {label}" if tlabel else f"View — {label}")
        self.resize(1000, 760)
        self._track = track
        self._keep: list = []                       # keep QImage buffers alive
        self._children: list = []                   # spawned 3-D viewer windows
        self._pending: dict[int, str] = {}           # tab index -> path, unbuilt
        self._paths: dict[int, str] = {}             # tab index -> path, always
        self._live3d: list[int] = []                 # built 3-D tabs, oldest first
        self._samples: dict[int, tuple] = {}         # token -> (placeholder, head, path)
        self._token = 0
        self.sampleReady.connect(self._on_sample_ready)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.TextElideMode.ElideNone)
        self.setCentralWidget(self.tabs)

        paths = list(getattr(instance, "view_paths", None)
                     or ([getattr(instance, "path", "")] if getattr(instance, "path", "")
                         else []))
        if not paths:
            self.tabs.addTab(self._msg("This product has no viewable files."), "empty")
            return
        names = [Path(str(p)).name or "file" for p in paths]
        repeated = {n for n in names if names.count(n) > 1}
        for p, raw in zip(paths, names):
            short = self._SHORT.get(raw, raw)
            # Chunks repeat the same filenames; the chunk is what tells them apart.
            name = (f"{Path(str(p)).parent.name} · {short}" if raw in repeated
                    else raw)
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
            self._paths[index] = str(p)
        # Open on the first previewable picture, never on a dead ".psx" tab.
        exts = [Path(str(p)).suffix.lower() for p in paths]
        first = next((i for i, e in enumerate(exts) if e in self._PREVIEW_EXTS),
                     next((i for i, e in enumerate(exts) if e != ".psx"), 0))
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(first)
        self.tabs.blockSignals(False)
        self.tabs.currentChanged.connect(self._ensure_tab)
        self._ensure_tab(self.tabs.currentIndex())

    def _set_tab_content(self, index: int, widget: QWidget) -> None:
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

    def _ensure_tab(self, index: int) -> None:
        """Materialise one tab's real widget the first time it is shown."""
        path = self._pending.pop(int(index), None)
        if path is None:
            return
        try:
            widget = self._widget_for(path)
        except Exception as exc:
            widget = self._msg(f"Could not open {path}\n\n{type(exc).__name__}: {exc}")
        self._set_tab_content(index, widget)
        if Path(path).suffix.lower() in MESH_EXTS:
            # Each 3-D preview holds a matplotlib figure of up to 120k points;
            # a 28-chunk run clicked through would hold ~2 GB.  Keep the most
            # recent few and let the rest rebuild when revisited.
            self._live3d = [i for i in self._live3d if i != index] + [int(index)]
            while len(self._live3d) > self.MAX_LIVE_3D:
                old = self._live3d.pop(0)
                self._set_tab_content(old, self._msg(
                    f"released to save memory — reopening {Path(self._paths[old]).name} …"))
                self._pending[old] = self._paths[old]

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
            btn.clicked.connect(lambda: self._external(path))
            folder = QPushButton("Open containing folder")
            folder.clicked.connect(lambda: self._folder(path))
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.addWidget(self._msg(f"{ext.lstrip('.').upper()} document:\n{path}"))
            lay.addWidget(btn)
            lay.addWidget(folder)
            lay.addStretch(1)
            return w
        if ext == ".csv":
            return self._csv_widget(path)
        if ext in (".geojson", ".json"):
            return self._geojson_canvas(path)
        if ext in MESH_EXTS:
            return self._mesh_widget(path)
        # Anything else (e.g. a Metashape .psx): say so and offer the folder.
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(self._msg(f"No built-in viewer for '{ext}' files:\n{path}"))
        btn = QPushButton("Open containing folder")
        btn.clicked.connect(lambda: self._folder(path))
        lay.addWidget(btn)
        lay.addStretch(1)
        return w

    # -- desktop hand-off (failures reach the main log, never vanish) ---------
    def _report(self, text: str, error: bool = True) -> None:
        main = self.parent()
        fn = getattr(main, "log_error" if error else "log", None)
        if callable(fn):
            fn(text)

    def _external(self, path: str) -> None:
        why = _open_external(path)
        if why:
            self._report(f"could not open {Path(path).name} externally: {why}")
            QMessageBox.information(self, "No external viewer", f"{path}\n\n{why}")
        else:
            self._report(f"opened {Path(path).name} in the external viewer", error=False)

    def _folder(self, path: str) -> None:
        if not _open_folder(path):
            self._report(f"could not open the folder of {path} — no file manager found "
                         "(xdg-open / explorer.exe)")

    @staticmethod
    def _count_rows(path: str, max_bytes: int = 400_000_000) -> int | None:
        """Data rows in a CSV (newline count minus the header); None if too big."""
        try:
            if Path(path).stat().st_size > max_bytes:
                return None
            n = 0
            with open(path, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 22), b""):
                    n += blk.count(b"\n")
            return max(0, n - 1)
        except Exception:
            return None

    def _csv_widget(self, path: str) -> QWidget:
        import pandas as pd
        df = pd.read_csv(path, nrows=self.CSV_ROWS)
        total = self._count_rows(path)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        top = QHBoxLayout()
        if total is not None and total <= len(df):
            note = f"{len(df):,} row(s)"
        else:
            note = (f"showing first {len(df):,} of "
                    + (f"{total:,}" if total is not None else "more")
                    + " rows — open the containing folder for the full file")
        if Path(path).name == "fathomnet_detections.csv":
            note += ("  ·  species-level labels (cls) are not reliable; trust the "
                     "bucket level only")
        lab = QLabel(note)
        lab.setWordWrap(True)
        top.addWidget(lab, 1)
        btn = QPushButton("Open containing folder")
        btn.clicked.connect(lambda: self._folder(path))
        top.addWidget(btn)
        lay.addLayout(top)
        view = QTableView()
        model = _DFModel(df)
        self._keep.append(model)
        view.setModel(model)
        view.resizeColumnsToContents()
        lay.addWidget(view, 1)
        return w

    def _directory_grid(self, path: str, limit: int = 12) -> QWidget:
        """Frame-set style view_path: a contact sheet sampled across the set.

        Frame sets keep their JPEGs in ``segment_*/frames/`` below the listed
        directory, so a flat listing showed "0 images" for every set.  Fall
        back to a recursive walk and sample evenly, not just the first few.
        """
        img_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        files = sorted(p for p in Path(path).iterdir() if p.is_file())
        imgs = [p for p in files if p.suffix.lower() in img_ext]
        deep = False
        if not imgs:
            deep = True
            for root, dirs, names in os.walk(path):
                dirs.sort()
                imgs.extend(Path(root) / n for n in sorted(names)
                            if n.lower().endswith(img_ext))
        if len(imgs) > limit:
            pick = sorted({int(i * (len(imgs) - 1) / (limit - 1)) for i in range(limit)})
            shown = [imgs[i] for i in pick]
            how = f"showing {len(shown)} sampled evenly across the set"
        else:
            shown = list(imgs)
            how = f"showing all {len(shown)}"
        w = QWidget()
        lay = QVBoxLayout(w)
        head = QLabel(f"{path}\n{len(imgs):,} image(s)"
                      + (" in sub-folders" if deep else "") + f" — {how}")
        head.setWordWrap(True)
        lay.addWidget(head)
        btn = QPushButton("Open containing folder")
        btn.clicked.connect(lambda: self._folder(path))
        lay.addWidget(btn)
        grid = QGridLayout()
        # Decode each frame straight at thumbnail size (JPEG scaled decoding):
        # full-resolution decodes of 12 video frames took ~16 s on /mnt.
        from PySide6.QtGui import QImageReader

        def thumb(fp: Path):
            try:
                rd = QImageReader(str(fp))
                sz = rd.size()
                if sz.isValid() and max(sz.width(), sz.height()) > 240:
                    rd.setScaledSize(sz.scaled(240, 240, Qt.AspectRatioMode.KeepAspectRatio))
                return rd.read()
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            thumbs = list(pool.map(thumb, shown))
        for i, (p, img) in enumerate(zip(shown, thumbs)):
            if img is None or img.isNull():
                continue
            cell = QLabel()
            cell.setPixmap(QPixmap.fromImage(img))
            try:
                cell.setToolTip(str(p.relative_to(path)))
            except ValueError:
                cell.setToolTip(p.name)
            grid.addWidget(cell, i // 4, i % 4)
        lay.addLayout(grid)
        lay.addStretch(1)
        return w

    def _mesh_widget(self, path: str) -> QWidget:
        """A 3-D product: full PyVista viewer when available, preview always.

        The preview is a rotatable matplotlib 3-D scatter of a bounded sample
        (see ``_sample_ply`` / ``_sample_obj``), so a 2 GB dense cloud never
        loads whole.  The sample is read on a background thread — a cold read
        through the /mnt 9p drives took 5–96 s and froze the whole app when it
        ran here — and the figure replaces the "sampling …" placeholder when it
        arrives.  The exact sample size is stated on the figure: this is a
        preview, not the product.
        """
        ext = Path(path).suffix.lower()
        w = QWidget()
        lay = QVBoxLayout(w)
        size_mb = Path(path).stat().st_size / 1e6
        head = QLabel(f"<b>{Path(path).name}</b> — {size_mb:,.0f} MB")
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
        ext_btn.clicked.connect(lambda: self._external(path))
        folder = QPushButton("Open containing folder")
        folder.clicked.connect(lambda: self._folder(path))
        for b in (btn3d, ext_btn, folder):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        if ext not in (".ply", ".obj"):
            lay.addWidget(self._msg(
                f"No built-in preview for '{ext}' — use the buttons above."))
            lay.addStretch(1)
            return w

        placeholder = self._msg(f"sampling {Path(path).name} for the preview …")
        lay.addWidget(placeholder, 1)
        self._token += 1
        token = self._token
        self._samples[token] = (placeholder, head, path)
        emit = self.sampleReady.emit

        def work() -> None:
            try:
                res = _sample_ply(path) if ext == ".ply" else _sample_obj(path)
            except Exception as exc:                        # noqa: BLE001
                res = exc
            try:
                emit(token, res)
            except RuntimeError:                  # the viewer was closed meanwhile
                pass
        threading.Thread(target=work, name="mesh-sample", daemon=True).start()
        return w

    def _on_sample_ready(self, token, result) -> None:
        """GUI thread: swap the finished sample's figure in for its placeholder."""
        entry = self._samples.pop(token, None)
        if entry is None:
            return
        placeholder, head, path = entry
        try:
            import shiboken6
            if not (shiboken6.isValid(placeholder) and shiboken6.isValid(head)):
                return                                  # tab released meanwhile
        except Exception:
            pass
        host = placeholder.parentWidget()
        lay = host.layout() if host is not None else None
        if lay is None:
            return
        if isinstance(result, BaseException):
            widget = self._msg(f"Could not sample {path}\n\n"
                               f"{type(result).__name__}: {result}")
        else:
            xyz, rgb, n_vertices, n_faces = result
            approx = Path(path).suffix.lower() == ".obj"
            kind = "mesh" if (n_faces or approx) else "point cloud"
            size_mb = Path(path).stat().st_size / 1e6
            head.setText(f"<b>{Path(path).name}</b> — {kind}, {size_mb:,.0f} MB"
                         + (f", {'≈' if approx else ''}{n_vertices:,} vertices"
                            if n_vertices else "")
                         + (f", {'≈' if approx else ''}{n_faces:,} faces"
                            if n_faces else ""))
            widget = (self._mesh_canvas(xyz, rgb, n_vertices, approx) if len(xyz)
                      else self._msg("The sample held no vertices — use the "
                                     "buttons above."))
        idx = lay.indexOf(placeholder)
        lay.removeWidget(placeholder)
        placeholder.setParent(None)
        placeholder.deleteLater()
        lay.insertWidget(max(0, idx), widget, 1)

    def _mesh_canvas(self, xyz, rgb, n_vertices: int, approx: bool) -> QWidget:
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
        ax.set_title(f"preview — {len(xyz):,} of {'≈' if approx else ''}{n_vertices:,} "
                     f"vertices (drag to rotate)", color=INK, fontsize=9)
        fig.tight_layout()
        return FigureCanvasQTAgg(fig)

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
        """GeoTIFF preview from a decimated (overview) read.

        Single-band rasters (depth, sensor channels) are drawn through a colour
        map with a labelled colour bar and map-metre extent, so a value, a
        direction and a scale can be read off them; RGB orthomosaics are shown
        as pictures, scaled up when small so they fill the tab.
        """
        import rasterio
        with rasterio.open(path) as ds:
            # ceil, not trunc: a 2.8x oversized raster must still be decimated.
            step = max(1, -(-max(ds.width, ds.height) // self.MAX_PX))
            h, w = max(1, ds.height // step), max(1, ds.width // step)
            n = 3 if ds.count >= 3 else 1
            arr = ds.read(list(range(1, n + 1)), out_shape=(n, h, w),
                          masked=True).astype("float32").filled(np.nan)
            b = ds.bounds
            extent = (b.left, b.right, b.bottom, b.top)
            units = (ds.units[0] if ds.units else "") or ""
            desc = (ds.descriptions[0] if ds.descriptions else "") or ""
            projected = bool(ds.crs and ds.crs.is_projected)
        if n == 1:
            return self._raster_figure(path, arr[0], extent, units, desc, projected)
        bands = []
        for band in arr:
            fin = band[np.isfinite(band)]
            lo, hi = (np.percentile(fin, 2), np.percentile(fin, 98)) if fin.size else (0, 1)
            if hi <= lo:
                hi = lo + 1.0
            scaled = np.clip((np.nan_to_num(band, nan=lo) - lo) / (hi - lo), 0, 1) * 255
            bands.append(scaled.astype(np.uint8))
        rgb = np.dstack(bands)
        buf = np.ascontiguousarray(rgb)
        img = QImage(buf.data, buf.shape[1], buf.shape[0], buf.strides[0],
                     QImage.Format.Format_RGB888).copy()
        self._keep.append(buf)
        pm = QPixmap.fromImage(img)
        if max(pm.width(), pm.height()) < self.FIT_PX:
            pm = pm.scaled(self.FIT_PX, self.FIT_PX, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        lab = QLabel()
        lab.setPixmap(pm)
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return lab

    def _raster_figure(self, path: str, band, extent, units: str, desc: str,
                       projected: bool) -> QWidget:
        name = Path(path).stem
        quantity = desc or name.replace("_2d", "").replace("_", " ")
        if not units and "depth" in name.lower():
            units = "m"
        fin = band[np.isfinite(band)]
        lo, hi = (float(np.percentile(fin, 2)), float(np.percentile(fin, 98))) \
            if fin.size else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1.0
        fig = Figure(facecolor=BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)
        cmap = matplotlib.colormaps["viridis"].copy()
        cmap.set_bad(BG)                         # nodata is background, not black
        # Rasters along a trackline can be 100x longer than wide; at equal
        # aspect they shrink to a sliver, so stretch those (and say so).
        w_m, h_m = abs(extent[1] - extent[0]), abs(extent[3] - extent[2])
        stretched = max(w_m, h_m) > 4 * max(min(w_m, h_m), 1e-9)
        im = ax.imshow(np.ma.masked_invalid(band), cmap=cmap, vmin=lo, vmax=hi,
                       extent=extent, origin="upper", interpolation="nearest",
                       aspect="auto" if stretched else "equal")
        if self._track is not None and len(self._track) and projected:
            tx, ty = self._track[:, 1], self._track[:, 2]
            if (extent[0] - 5000 <= tx.mean() <= extent[1] + 5000
                    and extent[2] - 5000 <= ty.mean() <= extent[3] + 5000):
                ax.plot(tx, ty, "-", color=INK, lw=0.4, alpha=0.35)
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(quantity + (f" ({units})" if units else "")
                     + " — colour range 2–98 %", color=MUT, fontsize=8)
        cb.ax.tick_params(colors=MUT, labelsize=7)
        cb.outline.set_edgecolor(GRIDC)
        for sp in ax.spines.values():
            sp.set_color(GRIDC)
        ax.tick_params(colors=MUT, labelsize=7)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(7))
        if projected:
            _metres_axes(ax)
            ax.set_xlabel("easting (m)", color=MUT, fontsize=8)
            ax.set_ylabel("northing (m)", color=MUT, fontsize=8)
        ax.set_title(Path(path).name + (" (aspect stretched to fit)" if stretched else ""),
                     color=INK, fontsize=9)
        fig.tight_layout()
        return FigureCanvasQTAgg(fig)

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
            if abs(float(xs.mean())) > 360 or abs(float(ys.mean())) > 90:   # metres
                _metres_axes(ax)
                ax.set_xlabel("easting (m)", color=MUT, fontsize=8)
                ax.set_ylabel("northing (m)", color=MUT, fontsize=8)
            else:
                ax.set_xlabel("longitude (°)", color=MUT, fontsize=8)
                ax.set_ylabel("latitude (°)", color=MUT, fontsize=8)
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
        self._dirty = False                 # imports not yet on disk (save failed)
        self._track_stale = False           # interp invalidated by a re-import

        self.setWindowTitle(f"EPR Imaging — {self.ws_dir.stem}")
        self.resize(1620, 940)
        self._build_ui()
        self._build_menus()

        self.log(f"workspace {self.ws_dir}")
        if PC_IMPORT_ERROR:
            self.log_error(f"product_catalog unavailable — {PC_IMPORT_ERROR}")
        self._load_workspace()
        if self.ws.get("navigation_file"):
            self.log(f"ready — expand a product type and double-click {GEN_LABEL}, "
                     f"or Run ▸ {RUN_ALL_LABEL}")
        self._refresh_inputs()
        self.refresh_jobs()
        self._load_track()
        self._refresh_stop()
        self.statusBar().showMessage(
            "ready" if self.ws.get("navigation_file")
            else "no navigation imported — File ▸ Import navigation and orientation…")

    # -- construction --------------------------------------------------------
    DEFAULT_SIZES = [360, 900, 360]

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
        pal = self.palette()
        dark = pal.color(QPalette.ColorRole.Window).lightness() < 128
        tc = pal.color(QPalette.ColorRole.WindowText)
        logo_pm = _wordmark_pixmap(LOGO_HEIGHT_PX, dark, (tc.red(), tc.green(), tc.blue()),
                                   self.devicePixelRatioF())
        if logo_pm is not None:
            self.logo = QLabel()
            self.logo.setObjectName("whoiLogo")
            self.logo.setPixmap(logo_pm)
            self.logo.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.logo.setContentsMargins(2, 2, 0, 6)
            self.logo.setToolTip("Woods Hole Oceanographic Institution")
            lv.addWidget(self.logo)

        # What is imported right now — the only place the user can see it.
        in_head = QLabel("Inputs")
        in_head.setStyleSheet("font-weight: 600;")
        lv.addWidget(in_head)
        self.inputs_label = QLabel("")
        self.inputs_label.setWordWrap(True)
        self.inputs_label.setTextFormat(Qt.TextFormat.RichText)
        self.inputs_label.setContentsMargins(6, 0, 0, 4)
        lv.addWidget(self.inputs_label)

        self.job_combo = QComboBox()
        self.job_combo.currentIndexChanged.connect(self._on_job_changed)
        self.job_combo.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.job_combo.customContextMenuRequested.connect(self._job_menu)
        self.job_combo.setToolTip("Right-click to rename or delete a job.")
        job_head = QHBoxLayout()
        job_head.setContentsMargins(0, 0, 0, 0)
        job_lbl = QLabel("Job")
        job_lbl.setStyleSheet("font-weight: 600;")
        job_lbl.setToolTip("A job is a named set of time intervals; every product "
                           "generated with it selected covers only those intervals. "
                           "“Whole trackline” is the whole dive.")
        job_head.addWidget(job_lbl)
        job_head.addStretch(1)
        self.btn_job_manage = QPushButton("⋯")
        self.btn_job_manage.setFixedWidth(26)
        self.btn_job_manage.setToolTip("Rename or delete the selected job")
        self.btn_job_manage.clicked.connect(
            lambda: self._job_menu(self.btn_job_manage.geometry().bottomLeft()))
        job_head.addWidget(self.btn_job_manage)
        lv.addLayout(job_head)
        lv.addWidget(self.job_combo)
        # The selected job's intervals, spelled out (the map shows them in orange).
        self.job_intervals_label = QLabel("")
        self.job_intervals_label.setWordWrap(True)
        self.job_intervals_label.setContentsMargins(6, 0, 0, 2)
        self.job_intervals_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        lv.addWidget(self.job_intervals_label)
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
        log_lbl = QLabel("Log")
        log_lbl.setStyleSheet("font-weight: 600;")
        head.addWidget(log_lbl)
        head.addStretch(1)
        self.chk_wrap = QCheckBox("Wrap")
        self.chk_wrap.setChecked(True)
        self.chk_wrap.setToolTip("Wrap long lines (off = horizontal scrollbar)")
        self.chk_wrap.toggled.connect(self._set_wrap)
        head.addWidget(self.chk_wrap)
        self.btn_details = QPushButton("Details…")
        self.btn_details.setToolTip("Show the collapsed traceback of the last failure")
        self.btn_details.setEnabled(False)
        self.btn_details.clicked.connect(self._show_details)
        head.addWidget(self.btn_details)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop)
        head.addWidget(self.btn_stop)
        rv.addLayout(head)
        self.logger = QPlainTextEdit()
        self.logger.setReadOnly(True)
        self.logger.setMaximumBlockCount(20000)
        self.logger.setFont(QFont("monospace", 8))
        # The log palette (C_ERR/C_WARN/C_DIM) is tuned for the dark canvas
        # background; on the system's white base "note:" lines were ~1.5:1.
        self.logger.setStyleSheet(
            f"QPlainTextEdit {{ background: {BG}; color: {INK}; "
            f"selection-background-color: {GRIDC}; selection-color: {INK}; }}")
        # Wrapped by default: this panel is the app's only feedback channel and
        # a clipped path is worse than a wrapped one.
        self.logger.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.logger.setPlaceholderText(
            "Start with File ▸ Import navigation and orientation… (and video, sensors).\n\n"
            "Then pick a job (or stay on Whole trackline), expand a product type and "
            f"double-click {GEN_LABEL} — or Run ▸ {RUN_ALL_LABEL}.\n\n"
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
        act_new = QAction("New workspace…", self, triggered=self._new_workspace)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_open = QAction("Open workspace…", self, triggered=self._open_workspace)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        fm.addAction(act_new)
        fm.addAction(act_open)
        fm.addSeparator()
        self.act_save = QAction("Save workspace", self, triggered=self._save)
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
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
        self.act_default_all = QAction(RUN_ALL_LABEL + "…", self,
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
        hm.addAction(QAction("Quick start", self, triggered=self._quick_start))
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
                       "“Details…” above to read them", C_DIM)

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
        # HTML collapses leading spaces, which flattened the backend's
        # "  note:" / "    note:" nesting; keep them as non-breaking spaces.
        text = str(text)
        lead = len(text) - len(text.lstrip(" "))
        text = text.lstrip(" ")
        safe = (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace("\n", "<br>"))
        self.logger.appendHtml(
            f'<span style="color:{colour}">[{_stamp()}] {"&nbsp;" * lead}{safe}</span>')

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
            if self.ws_json.exists():
                self.log_error(f"could not load workspace: {exc}")
            else:
                self.log(f"{WARN_PREFIX}new workspace — no {self.ws_json.name} yet; "
                         "start with File ▸ Import navigation and orientation…")

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

    def _save(self, quiet: bool = False) -> bool:
        if not getattr(self, "ws_loaded", True):
            # Saving now would replace an unreadable workspace.json with this
            # session's empty skeleton, destroying whatever is still in it.
            self.log_error(f"refusing to save over {self.ws_json.name}: it could not "
                           "be read at startup. Fix or move the file, or use "
                           "File ▸ Save workspace as… to write elsewhere.")
            self._dirty = True
            self._refresh_inputs()
            return False
        try:
            self._write_workspace(self.ws_json)
            self._dirty = False
            self.log(f"saved {self.ws_json.name}" if quiet else f"saved {self.ws_json}")
            ok = True
        except Exception as exc:
            self._dirty = True
            self.log_error(f"save failed: {exc}")
            ok = False
        self._refresh_inputs()
        return ok

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
            self._dirty = False
            self._refresh_inputs()
            self.setWindowTitle(f"EPR Imaging — {root.stem}")
            self.log(f"saved workspace as {self.ws_json}")
            _remember(str(root))
        except Exception as exc:
            self.log_error(f"save-as failed: {exc}")

    # -- switching workspace -------------------------------------------------
    def _switch_to(self, root: Path) -> None:
        if self._busy:
            self.log_error("a task is running — wait for it to finish before switching "
                           "workspace")
            return
        win = MainWindow(root)
        win.resize(self.size())
        win.show()
        _WINDOWS.append(win)             # unparented: keep it alive past this call
        if win.ws_json.is_file() and win.ws_loaded:
            _remember(str(root))
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        if self.close() and self in _WINDOWS:
            _WINDOWS.remove(self)

    def _open_workspace(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open EPR workspace (.eprproj)",
                                             str(self.ws_dir.parent))
        if not d:
            return
        root = Path(d)
        if root.suffix != ".eprproj" and not (root / "workspace.json").is_file():
            self.log_error(f"{root} is not a workspace — pick a *.eprproj folder "
                           "(or one containing workspace.json)")
            return
        self._switch_to(root)

    def _new_workspace(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "New workspace", str(self.ws_dir.parent),
                                           "EPR project bundle (*.eprproj)")
        if not p:
            return
        root = Path(p)
        if root.suffix != ".eprproj":
            root = root.with_suffix(".eprproj")
        if root.exists() and any(root.iterdir()):
            self.log_error(f"{root} already exists and is not empty — use File ▸ Open "
                           "workspace… instead")
            return
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.log_error(f"could not create {root}: {exc}")
            return
        self._switch_to(root)

    # -- imports -------------------------------------------------------------
    # Every run reads workspace.json from DISK (product_catalog._ws_data), so an
    # import held only in memory was invisible to the next run: it silently used
    # the previous configuration, or none.  Each successful import is therefore
    # saved at once — the same model the job store already follows — and a
    # change to the navigation or sensor inputs invalidates interp_full.csv so
    # the next run rebuilds it instead of reusing the stale table.

    def _after_import(self, what: str, interp_inputs_changed: bool) -> None:
        if interp_inputs_changed:
            self._invalidate_interp(what)
        if not self._save(quiet=True):
            self.log_error(f"{what} is NOT on disk yet — runs will not see it until "
                           "File ▸ Save workspace (or Save as…) succeeds")
        self._refresh_inputs()

    def _interp_sidecars(self, interp: Path) -> list[Path]:
        """interp_full.csv's fingerprint / meta sidecar(s), whatever the backend calls them."""
        out: list[Path] = []
        for name in ("interp_meta_path", "interp_fingerprint_path"):
            fn = getattr(pc, name, None) if pc is not None else None
            if callable(fn):
                try:
                    out.append(Path(fn(self.ws_path)))
                except Exception:
                    pass
        try:
            for p in interp.parent.iterdir():
                n = p.name
                if (p.is_file() and n.startswith(interp.stem) and n != interp.name
                        and not n.endswith(".stale.csv")
                        and (n.endswith(".json") or "fingerprint" in n
                             or n.endswith((".sha", ".sha256", ".meta")))):
                    out.append(p)
        except OSError:
            pass
        return list(dict.fromkeys(out))

    def _invalidate_interp(self, what: str) -> None:
        """Retire interp_full.csv (and its fingerprint) so the next run rebuilds it.

        The table is renamed to interp_full.stale.csv rather than deleted: it is
        derived data, but a rename costs nothing and can be undone by hand.
        """
        if pc is None:
            return
        try:
            interp = Path(pc.interp_path(self.ws_path))
        except Exception as exc:
            self.log_error(f"could not locate interp_full.csv to invalidate it: {exc}")
            return
        for side in self._interp_sidecars(interp):
            try:
                side.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                self.log_error(f"could not remove {side.name}: {exc}")
        if not interp.is_file():
            return
        stale = interp.with_name(interp.stem + ".stale.csv")
        try:
            os.replace(interp, stale)
        except OSError as exc:
            self.log_error(f"{what} changed but {interp.name} could not be retired "
                           f"({exc}) — delete it by hand, or the next run reuses "
                           "the OLD table")
            return
        self._track_stale = True
        self.log(f"{WARN_PREFIX}{what} changed — {interp.name} retired to {stale.name}; "
                 f"it rebuilds on the next run (Run ▸ {RUN_ALL_LABEL}). "
                 "The map keeps showing the old track until then.")

    def _import_nav(self) -> None:
        dlg = NavImportDialog(self, str(self.ws_dir), current=self.ws.get("navigation_file"))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        nav = dlg.navigation_config()
        if nav is None:
            self.log_error("navigation import not applied — no readable Renav CSV chosen")
            return
        old = self.ws.get("navigation_file")
        self.ws["navigation_file"] = nav
        span = (f"{nav.start_time} → {nav.end_time}" if nav.start_time else "unknown span")
        self.log(f"navigation imported: {Path(nav.latitude_source.csv_path).name} ({span})")
        if nav.altitude_source:
            self.log(f"altitude source: {Path(nav.altitude_source.csv_path).name}")
        self._after_import("navigation", interp_inputs_changed=(old != nav))

    def _import_video(self) -> None:
        dlg = VideoImportDialog(self, str(self.ws.get("video_directory") or ""),
                                str(self.ws.get("filename_datetime_format") or ""))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        directory, fmt = dlg.values()
        if not directory:
            self.log_error("video import not applied — no directory chosen")
            return
        self.ws["video_directory"] = directory
        self.ws["filename_datetime_format"] = fmt
        try:
            n = len([p for p in Path(directory).iterdir()
                     if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".m4v")])
        except Exception:
            n = 0
        self.log(f"video directory set: {directory} ({n} video file(s)), format '{fmt}'")
        # Video does not feed interp_full.csv (frame sets read it directly).
        self._after_import("video", interp_inputs_changed=False)

    def _import_sensor(self) -> None:
        dlg = SensorChannelDialog(self, str(self.ws_dir))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        vals = dlg.values()
        if not vals:
            self.log_error("sensor import not applied — need CSV, timestamp and channel column")
            return
        csv_path, ts_col, channel = vals
        # Refuse files the import already knows are unusable, instead of adding
        # them and failing (or silently producing all-NaN channels) at run time.
        try:
            with open(csv_path, "r", errors="replace") as fh:
                header = [c.strip() for c in fh.readline().split(",")]
        except Exception as exc:
            self.log_error(f"sensor import not applied — cannot read {csv_path}: {exc}")
            return
        missing = [c for c in (ts_col, channel.source_column) if c not in header]
        if missing:
            self.log_error(f"sensor import not applied — {Path(csv_path).name} has no "
                           f"column(s) {missing}")
            return
        files = list(self.ws.get("sensor_files") or [])
        target = next((sf for sf in files
                       if str(getattr(sf, "csv_path", "")) == csv_path), None)
        if target is None:
            t0, t1 = self._ts_bounds(csv_path, ts_col)
            if t0 is None:
                self.log_error(f"sensor import not applied — no readable timestamps in "
                               f"column '{ts_col}' of {Path(csv_path).name}")
                return
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
        self._after_import("sensor configuration",
                           interp_inputs_changed=(dup is None or dup != channel))

    # -- inputs summary --------------------------------------------------------
    def _refresh_inputs(self) -> None:
        """The "Inputs" block above the job row: what the next run will read."""
        if not hasattr(self, "inputs_label"):
            return

        def esc(t) -> str:
            return (str(t).replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;"))
        muted = _on_light(MUT, "#57606a")
        rows: list[str] = []
        tips: list[str] = []
        nav = self.ws.get("navigation_file")
        if nav is not None:
            try:
                src = nav.latitude_source
                span = ""
                if getattr(nav, "start_time", None) and getattr(nav, "end_time", None):
                    span = (f" · {nav.start_time:%m-%d %H:%M} → "
                            f"{nav.end_time:%m-%d %H:%M}")
                alt = " + altitude" if getattr(nav, "altitude_source", None) else ""
                rows.append(f"Nav: {esc(Path(src.csv_path).name)}{alt}{esc(span)}")
                tips.append(f"Navigation: {src.csv_path}")
                if getattr(nav, "altitude_source", None):
                    tips.append(f"Altitude: {nav.altitude_source.csv_path}")
            except Exception:
                rows.append("Nav: configured")
        else:
            rows.append(f'<span style="color:{muted}">Nav: not imported</span>')
        sensors = list(self.ws.get("sensor_files") or [])
        chans = [str(getattr(c, "display_name", "") or getattr(c, "source_column", ""))
                 for sf in sensors for c in (getattr(sf, "channels", None) or [])]
        if chans:
            shown = ", ".join(chans[:4]) + (f" +{len(chans) - 4}" if len(chans) > 4 else "")
            rows.append(f"Sensors: {len(chans)} channel(s) — {esc(shown)}")
            tips.extend(f"Sensor file: {getattr(sf, 'csv_path', '')}" for sf in sensors)
        else:
            rows.append(f'<span style="color:{muted}">Sensors: none</span>')
        vdir = str(self.ws.get("video_directory") or "")
        if vdir:
            short = "/".join(Path(vdir).parts[-2:]) or vdir
            rows.append(f"Video: {esc(short)}")
            tips.append(f"Video directory: {vdir} "
                        f"(format '{self.ws.get('filename_datetime_format') or ''}')")
        else:
            rows.append(f'<span style="color:{muted}">Video: not imported</span>')
        if self._dirty:
            rows.append(f'<span style="color:{_on_light(C_ERR, "#c62828")}">'
                        "● unsaved changes — runs read the file on disk</span>")
        self.inputs_label.setText("<br>".join(rows))
        self.inputs_label.setToolTip("\n".join(tips) or "Nothing imported yet — use the "
                                     "File menu imports.")

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
            "<b>EPR Imaging</b><br>Woods Hole Oceanographic Institution<br><br>"
            "Three panels: products, trackline, log.<br>"
            f"Pick a job, expand a product type, generate; or Run ▸ {RUN_ALL_LABEL}."
            f"<br><br>Workspace: {self.ws_dir}")

    def _quick_start(self) -> None:
        QMessageBox.information(
            self, "Quick start",
            "<b>1. Import</b> — File ▸ Import navigation and orientation…, then "
            "Import video… and Import new sensor channel…. Each import is saved "
            "to the workspace at once; the Inputs block (top left) shows what is "
            "loaded.<br><br>"
            "<b>2. Job (optional)</b> — a job is a named set of time intervals. "
            "Click two points on the track to make an interval, Stage interval, "
            "then Create job. Products generated with a job selected cover only "
            "its intervals. <i>Whole trackline</i> is the whole dive.<br><br>"
            f"<b>3. Generate</b> — expand a product type and double-click "
            f"{GEN_LABEL}, or Run ▸ {RUN_ALL_LABEL} for every product with the "
            "default settings (hours for photogrammetry and fauna).<br><br>"
            "<b>4. Browse</b> — double-click any product to view it.<br><br>"
            "<b>5. Catalog</b> — Run ▸ Build product catalog (PDF) gathers the "
            "selected job's products into survey/catalog/.<br><br>"
            "<b>Sampling technique</b> — photogrammetry and fauna work on frames "
            "pulled from the video; each sampling run (technique + spacing) is a "
            "separate frame set and gives separate downstream products.")

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
        if not getattr(self, "ws_loaded", True):
            self.log_error(f"not changing jobs: {self.ws_json.name} could not be read "
                           "at startup. Fix or move the file first.")
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
        if not getattr(self, "ws_loaded", True):
            self.log_error(f"not changing jobs: {self.ws_json.name} could not be read "
                           "at startup. Fix or move the file first.")
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
        intervals = getattr(job, "intervals", []) or []
        self.trackline.set_job_intervals(intervals)
        self._refresh_tree()
        whole = self._job_is_whole()
        scope = "Whole trackline" if whole else _display_label(getattr(job, "name", ""))
        self.act_default_all.setToolTip(
            f"Every product type with default settings, for: {scope}")
        self.act_default_all.setText(f"{RUN_ALL_LABEL} — {scope}…")
        # The job's intervals, listed (not only drawn in orange on the map).
        if whole:
            text = "All of the dive — no interval restriction."
        else:
            lines = []
            for iv in intervals[:6]:
                try:
                    t0, t1 = _iv_bounds(iv)
                    lines.append(f"{_utc(t0)} → {_qdt(t1).toString('HH:mm:ss')} "
                                 f"({_dur(t1 - t0)})")
                except Exception:
                    continue
            if len(intervals) > 6:
                lines.append(f"… and {len(intervals) - 6} more")
            text = (f"{len(intervals)} interval(s), UTC:\n" + "\n".join(lines)
                    if lines else "This job has no intervals.")
        self.job_intervals_label.setText(text)
        self.job_intervals_label.setStyleSheet(f"color: {_on_light(MUT, '#57606a')};")
        self.job_combo.setToolTip(text + "\n\nRight-click (or ⋯) to rename or delete a job.")

    def _interp_ready(self) -> bool:
        """True when interp_full.csv already exists for this workspace.

        track_polyline() builds it through the whole pipeline when it is
        missing — minutes of work whose log goes to stdout.  Never let that
        happen on the GUI thread; the user builds it from Run ▸ Run all
        products (defaults), which reports into the logger on a worker thread.
        """
        try:
            path = Path(pc.interp_path(self.ws_path))               # type: ignore[union-attr]
        except Exception:
            return True                     # unknown: let track_polyline decide
        if path.is_file() and path.stat().st_size > 0:
            return True
        # Guidance, not a failure (and a four-line absolute path wrapped badly).
        self.log(f"{WARN_PREFIX}no trackline yet — import navigation and sensors, then "
                 f"Run ▸ {RUN_ALL_LABEL} to build {path.name}")
        return False

    def _load_track(self) -> None:
        arr = None
        stale = False
        if pc is not None and self._interp_ready():
            # track_polyline() rebuilds a STALE table through the whole pipeline
            # (minutes) — never on the GUI thread.  A stale table is drawn from
            # disk for orientation only, and rebuilt by the next run.
            why = None
            check = getattr(pc, "interp_staleness", None)
            if callable(check):
                try:
                    is_stale, why = check(self.ws_path)
                    stale = bool(is_stale)
                except Exception:
                    stale = False
            try:
                arr = self._read_track_direct() if stale else pc.track_polyline(self.ws_path)
            except Exception as exc:
                self.log_error(f"trackline could not be read: {exc}")
            if stale:
                self.log(f"{WARN_PREFIX}interp_full.csv is out of date ({why}) — the map "
                         f"shows the old table; it rebuilds on the next run "
                         f"(Run ▸ {RUN_ALL_LABEL})")
        self.track = np.asarray(arr, dtype=float) if arr is not None and len(arr) else None
        self.trackline.set_track(self.track)
        self._track_stale = stale
        if self.track is not None:
            self.log(f"trackline: {len(self.track):,} vertices")
        elif arr is not None:
            self.log(f"{WARN_PREFIX}the interp table has no valid positions — is "
                     "navigation imported?")

    def _read_track_direct(self):
        """[t, easting, northing] straight from interp_full.csv — no rebuild, no writes."""
        import pandas as pd
        path = Path(pc.interp_path(self.ws_path))                # type: ignore[union-attr]
        df = pd.read_csv(path, usecols=lambda c: c in ("unix_time", "easting", "northing"))
        df = df.dropna().sort_values("unix_time", kind="stable")
        return df[["unix_time", "easting", "northing"]].to_numpy(dtype=float)

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
                ["Tip: whole-trackline products are listed under “Whole trackline”"])
            hint.setDisabled(True)
            hint.setToolTip(0, "This job only lists products generated for its own "
                               "intervals. Switch the Job selector to “Whole "
                               "trackline” to see the whole-trackline ones.")
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
            child = QTreeWidgetItem([_display_label(
                getattr(inst, "label", getattr(inst, "path", "?")))])
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
            menu.addAction("Generate new…").triggered.connect(
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
            # Viewers delete themselves on close (WA_DeleteOnClose); forget them.
            viewer.destroyed.connect(
                lambda _o=None, v=viewer: self._viewers.remove(v)
                if v in self._viewers else None)
            self._viewers.append(viewer)
            self.log(f"viewing {_display_label(getattr(inst, 'label', ''))}")
        except Exception as exc:
            self.log_error(f"viewer failed: {exc}")

    def _reveal(self, inst) -> None:
        path = str(getattr(inst, "path", "") or "")
        if path and _open_folder(path):
            self.log(f"opened folder for {Path(path).name}")
        else:
            self.log_error(f"could not open containing folder for '{path}'")

    # -- runs ----------------------------------------------------------------
    #: Rough duration class per product type, for the confirmation dialog.
    DURATION = {
        "nav_trackline": "seconds", "depth_raster": "about a minute",
        "sensor_raster": "about a minute per channel",
        "frame_set": "tens of minutes to hours (video decoding)",
        "photogrammetry": "hours (Metashape)",
        "anomaly_detection": "10–30 minutes (MATLAB)",
        "fauna_detection": "hours on a GPU, far longer on CPU",
        "spectrum_trackline": "minutes", "anomaly_trackline": "minutes",
        "survey_report": "minutes",
    }
    #: default_run_all's step order (product_catalog.default_run_all).
    RUN_ALL_ORDER = ("nav_trackline", "depth_raster", "sensor_raster", "frame_set",
                     "photogrammetry", "anomaly_detection", "fauna_detection",
                     "spectrum_trackline", "anomaly_trackline", "survey_report")

    def _type_label(self, key: str) -> str:
        try:
            return str(pc.product_type(key).label)             # type: ignore[union-attr]
        except Exception:
            return key.replace("_", " ")

    def _canonical_targets(self, keys, settings: dict | None = None) -> list[str]:
        """Existing canonical whole-trackline files a run of ``keys`` rewrites in place.

        Used only when product_catalog offers no ``replacement_preview``.  Mirrors
        the in-place writes listed in the integrity review (fauna census,
        anomaly catalog, merged mosaics, survey report).
        """
        out: list[str] = []
        try:
            res = pc._resolver(self.ws_path)                  # type: ignore[union-attr]
            survey = Path(res.survey_products())
            anomaly = Path(res.anomaly_dir())
        except Exception:
            survey = self.ws_dir / "survey"
            anomaly = survey / "anomaly"
        keys = set(keys)
        if "fauna_detection" in keys:
            same_sampling = True
            if settings:
                try:
                    same_sampling = bool(pc.sampling_matches(settings, {}))  # type: ignore
                except Exception:
                    same_sampling = True
            if same_sampling:
                names = getattr(pc, "FAUNA_PRODUCT_FILES", ()) or (
                    "fathomnet_detections.csv", "fauna_points_utm.geojson",
                    "fauna_density.csv", "occurrences.csv")
                out += [str(survey / "fauna" / n) for n in names
                        if (survey / "fauna" / n).is_file()]
        if "anomaly_detection" in keys and anomaly.is_dir():
            out += [str(p) for p in sorted(anomaly.iterdir()) if p.is_file()]
        if "photogrammetry" in keys:
            merged = survey / "photogrammetry" / "merged"
            out += [str(merged / n) for n in ("ortho_merged.tif", "dem_merged.tif")
                    if (merged / n).is_file()]
        if "survey_report" in keys:
            rep = self.ws_dir / "SURVEY_REPORT.html"
            if rep.is_file():
                out.append(str(rep))
        return out

    def _replacement_preview(self, job, keys, settings: dict | None = None) -> list[str]:
        fn = getattr(pc, "replacement_preview", None) if pc is not None else None
        if callable(fn):
            try:
                if set(keys) >= set(self.RUN_ALL_ORDER):
                    return [str(x) for x in (fn(self.ws_path, job) or [])]
                try:
                    got = fn(self.ws_path, job, type_keys=list(keys))
                    return [str(x) for x in (got or [])]
                except TypeError:
                    pass                      # whole-run-only signature: use ours
            except Exception as exc:
                self.log(f"{WARN_PREFIX}replacement_preview failed ({exc}); "
                         "listing the known canonical files instead")
        return self._canonical_targets(keys, settings)

    def _confirm_run(self, title: str, scope: str, steps: list[str], duration: str,
                     replaced: list[str]) -> bool:
        """Yes/No gate (default No) before a long or overwriting run."""
        shown = replaced[:12]
        body = (f"<b>{title}</b><br>Job: <b>{scope}</b><br><br>"
                f"<b>Steps:</b> {', '.join(steps)}<br>"
                f"<b>Expected duration:</b> {duration}<br><br>")
        if replaced:
            try:
                root = str(self.ws_dir) + os.sep
                rel = [p[len(root):] if p.startswith(root) else p for p in shown]
            except Exception:
                rel = shown
            body += (f"<b>This REPLACES {len(replaced)} existing canonical file(s)</b> "
                     "(previous versions are moved to _superseded/&lt;timestamp&gt;/), "
                     "including:<br>"
                     + "<br>".join(f"&nbsp;&nbsp;• {r}" for r in rel)
                     + (f"<br>&nbsp;&nbsp;… and {len(replaced) - len(shown)} more"
                        if len(replaced) > len(shown) else "")
                     + "<br><br>Anything derived from them (catalog, BIIGLE labels, "
                       "figures) will then refer to the new data.<br><br>")
        else:
            body += "No existing canonical files are replaced.<br><br>"
        body += ("It can be stopped between steps with the Stop button. Start it?"
                 if self._cancel_hook() is not None else
                 "Once started it cannot be stopped from the app. Start it?")
        box = QMessageBox(QMessageBox.Icon.Warning if replaced else QMessageBox.Icon.Question,
                          "Confirm run", "", QMessageBox.StandardButton.Yes
                          | QMessageBox.StandardButton.No, self)
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(body)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        box.button(QMessageBox.StandardButton.Yes).setText("Start run")
        box.button(QMessageBox.StandardButton.No).setText("Cancel")
        return box.exec() == QMessageBox.StandardButton.Yes

    def _generate(self, ptype) -> None:
        if self._busy:
            self.log_error("a task is already running")
            return
        label = str(getattr(ptype, "label", getattr(ptype, "key", "product")))
        key = str(getattr(ptype, "key", ""))
        schema: list = []
        try:                    # schema(ws) fills workspace-dependent choices
            schema = list(ptype.schema(self.ws_path))
        except Exception:
            try:
                schema = list(getattr(ptype, "settings_schema", []) or [])
            except Exception as exc:
                self.log_error(f"settings_schema unavailable for {label}: {exc}")
        job = self.current_job()
        whole = getattr(job, "job_id", WHOLE) == WHOLE
        scope = "Whole trackline" if whole else _display_label(getattr(job, "name", ""))
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
        if whole:
            replaced = self._replacement_preview(job, [key], settings)
            if replaced and not self._confirm_run(
                    f"Generate {label}", scope, [label],
                    self.DURATION.get(key, "minutes"), replaced):
                self.log(f"generate {label} not started")
                return
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
        scope = "Whole trackline" if whole else _display_label(getattr(job, "name", ""))
        if not whole:
            try:
                ivs = getattr(job, "intervals", []) or []
                scope += f" ({len(ivs)} interval(s))"
            except Exception:
                pass
        try:
            n_ch = len(list(pc.sensor_channels(self.ws_path)))
        except Exception:
            n_ch = None
        steps = ["interp table"]
        for key in self.RUN_ALL_ORDER:
            lab = self._type_label(key)
            if key == "sensor_raster" and n_ch is not None:
                lab += f" ×{n_ch}"
            steps.append(lab)
        # Job scopes write into their own run folders; only the whole-trackline
        # run rewrites canonical files (unless the backend says otherwise).
        replaced = (self._replacement_preview(job, list(self.RUN_ALL_ORDER))
                    if whole or callable(getattr(pc, "replacement_preview", None))
                    else [])
        if not self._confirm_run(RUN_ALL_LABEL, scope, steps,
                                 "hours — photogrammetry and fauna detection dominate",
                                 replaced):
            self.log(f"{RUN_ALL_LABEL} not started")
            return
        self.log(f"{RUN_ALL_LABEL.lower()} — {scope}")

        def task(log_fn):
            return pc.default_run_all(self.ws_path, log_fn, job=None if whole else job)
        self._start(RUN_ALL_LABEL.lower(), task)

    # -- stop ----------------------------------------------------------------
    @staticmethod
    def _cancel_hook():
        """product_catalog's cooperative-cancel entry point, if it has one."""
        if pc is None:
            return None
        for name in ("request_cancel", "cancel_run", "cancel"):
            fn = getattr(pc, name, None)
            if callable(fn):
                return fn
        return None

    def _stop(self) -> None:
        fn = self._cancel_hook()
        if fn is None or not self._busy:
            return
        try:
            fn()
            self.log(f"{WARN_PREFIX}stop requested — the run stops after the current step")
            self.btn_stop.setEnabled(False)
        except Exception as exc:
            self.log_error(f"stop failed: {exc}")

    def _refresh_stop(self) -> None:
        hook = self._cancel_hook()
        self.btn_stop.setEnabled(bool(self._busy and hook is not None))
        self.btn_stop.setToolTip(
            "Stop the running task after its current step" if hook is not None else
            "Not available yet — cancel arrives with the process-isolation change "
            "(post-delivery). A running task finishes on its own.")

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
            self.log_error(f"{title} failed — {cause}")
            self._details = tb.rstrip().splitlines()
            self._flush_details()
            verdict = f"{title} failed — {cause}"
        elif self._fails:
            # The backend's own last red line carries the exact tally; repeating
            # a count here would risk contradicting it.
            self.log_error(f"{title} finished with failures — see the red lines "
                           "above (press “Details…” for the last traceback)")
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
        # A run may have just built (or rebuilt) interp_full.csv.
        if self.track is None or self._track_stale:
            self._load_track()

    def _set_busy(self, busy: bool, message: str, bad: bool = False) -> None:
        self._busy = busy
        self.act_default_all.setEnabled(not busy)
        if getattr(self, "act_catalog", None) is not None:
            self.act_catalog.setEnabled(not busy and _catalog_builder() is not None)
        self.trackline.set_busy(busy)
        self._refresh_stop()
        # The status bar is the last thing a user looks at: it must not say
        # "ready" after half the suite failed.
        self.statusBar().setStyleSheet(
            f"color: {_on_light(C_ERR, '#c62828')}; font-weight: bold;"
            if (bad and not busy) else "")
        self.statusBar().showMessage(f"running: {message}" if busy else message)

    # -- jobs ----------------------------------------------------------------
    def _create_job(self, staged: list) -> None:
        if pc is None:
            self.log_error("product_catalog unavailable — cannot create job")
            return
        if not getattr(self, "ws_loaded", True):
            # The job store rewrites workspace.json; over an unreadable file
            # that would replace its navigation/sensor/video config with {}.
            self.log_error(f"not creating a job: {self.ws_json.name} could not be read "
                           "at startup. Fix or move the file first.")
            return
        job = self.current_job()
        base = None
        if getattr(job, "job_id", WHOLE) != WHOLE:
            # Inheriting the selected job's intervals used to happen silently.
            n_base = len(getattr(job, "intervals", []) or [])
            answer = QMessageBox.question(
                self, "Create job",
                f"Also include the {n_base} interval(s) of the selected job "
                f"'{_display_label(getattr(job, 'name', ''))}' in the new job?\n\n"
                "No = the new job holds only the staged interval(s).",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.No)
            if answer == QMessageBox.StandardButton.Cancel:
                return
            if answer == QMessageBox.StandardButton.Yes:
                base = job
        name, ok = QInputDialog.getText(
            self, "Create job", "Name for the new job (blank = automatic):")
        if not ok:
            return
        ivs = [_mk_interval(t0, t1) for t0, t1 in staged]
        try:
            new_job = pc.create_job(self.ws_path, ivs, base_job=base)
        except Exception as exc:
            self.log_error(f"create_job failed: {exc}")
            return
        name = name.strip()
        if name:
            try:
                new_job = pc.rename_job(self.ws_path, getattr(new_job, "job_id"), name) \
                    or new_job
            except Exception as exc:
                self.log_error(f"job created, but naming it failed: {exc}")
        self.log(f"created {getattr(new_job, 'name', '?')} "
                 f"({len(ivs)} new interval(s)"
                 + (f", plus the intervals of {getattr(base, 'name', '')}" if base else "")
                 + ")")
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
                "interrupted.\n\nClose anyway? The window closes, but the program "
                "keeps running in the background until the task finishes (its "
                "remaining log lines are lost).",
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
    app.setApplicationName("EPR Imaging")
    app.setOrganizationName("Woods Hole Oceanographic Institution")
    if LOGO_MARK.is_file():
        app.setWindowIcon(QIcon(str(LOGO_MARK)))
    ws = argv[1] if len(argv) > 1 else ""
    if not ws:
        last = _last_workspace()
        ws = QFileDialog.getExistingDirectory(
            None, "Open EPR workspace (.eprproj)",
            last or str(Path.home())) or ""
    if not ws:
        print("no workspace selected", file=sys.stderr)
        return 1
    win = MainWindow(ws)
    if win.ws_json.is_file() and win.ws_loaded:
        _remember(ws)          # only a workspace that loaded becomes the default
    win.show()
    code = app.exec()
    # A task detached by closing mid-run is still on its QThread.  Returning now
    # would let interpreter teardown destroy that running QThread, which aborts
    # the process (core dump) and kills the task mid-write.  Keep the promise
    # the close dialog made: wait for it to finish.
    pending = [th for th, _w in list(_ORPHANS) if th.isRunning()]
    if pending:
        print(f"waiting for {len(pending)} background task(s) to finish …",
              file=sys.stderr, flush=True)
    for th, _w in list(_ORPHANS):
        th.wait()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
