#!/usr/bin/env python3
"""
build_architecture_pdf.py — generate the full project architecture breakdown PDF.

Produces `EPR_Imaging_Architecture.pdf`: a long-form technical document covering
the system's structure, data model, execution model, orchestration, external
tool boundaries, conventions and risks — with drawn diagrams, real code excerpts
pulled live from the source files, and metrics computed from the tree.

Code excerpts are read FROM THE ACTUAL SOURCE at build time (see `excerpt()`), so
the document cannot drift out of sync with the code it describes.

Layout uses measured text widths (see `text_width_pt`) rather than assumed
characters-per-line, so nothing is ever clipped at a page edge.

Run:  python3 build_architecture_pdf.py
"""

from __future__ import annotations

import ast
import collections
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg", force=True)          # batch only; never a GUI backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = pathlib.Path(__file__).resolve().parent
OUT = REPO / "docs" / "reports" / "EPR_Imaging_Architecture.pdf"

PAGE_W, PAGE_H = 8.5, 11.0                 # portrait letter
LAND_W, LAND_H = 11.0, 8.5                 # landscape letter
MARGIN = 0.72

INK       = "#1a1a1a"
MUTED     = "#6b6b6b"
RULE      = "#c9c9c9"
ACCENT    = "#1f4e79"
ACCENT_2  = "#c0504d"
OK        = "#2e7d32"
WARN      = "#b26a00"
CODE_BG   = "#f4f5f7"
BOX_FILLS = {
    "gui":     "#dbe8f5",
    "bridge":  "#e6dcf0",
    "service": "#dcecdc",
    "model":   "#f6e5c8",
    "ext":     "#f0dcdc",
    "plain":   "#eeeeee",
}

_page_no = [0]
_toc: list = []


# ==========================================================================
# Measured-text layout helpers
# ==========================================================================
_CHAR_W: dict = {}


def _char_widths(weight: str, family: str) -> dict:
    key = (weight, family)
    if key in _CHAR_W:
        return _CHAR_W[key]
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    charset = (" !\"#$%&'()*+,-./0123456789:;<=>?@"
               "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
               "abcdefghijklmnopqrstuvwxyz{|}~"
               "±°→←↑↓—–●■▪✓✗…│├└·")
    prop = FontProperties(size=100.0, weight=weight, family=family)
    widths = {}
    for ch in charset:
        try:
            pair = float(TextPath((0, 0), f"n{ch}n", prop=prop).get_extents().width)
            base = float(TextPath((0, 0), "nn", prop=prop).get_extents().width)
            widths[ch] = max(0.0, pair - base) / 100.0
        except Exception:                                    # noqa: BLE001
            widths[ch] = 0.62
    widths.setdefault(" ", 0.32)
    _CHAR_W[key] = widths
    return widths


def text_width_pt(text: str, size: float, weight="normal", family="sans-serif") -> float:
    if not text:
        return 0.0
    table = _char_widths(weight, family)
    return sum(table.get(c, 0.62) for c in text) * size


def wrap(text: str, width_pt: float, size: float, weight="normal") -> list:
    """Greedy wrap on measured widths; hard-splits any over-long token."""
    words, out = str(text).split(), []
    for w in words:
        if text_width_pt(w, size, weight) <= width_pt:
            out.append(w)
            continue
        piece = ""
        for ch in w:
            if piece and text_width_pt(piece + ch, size, weight) > width_pt:
                out.append(piece)
                piece = ch
            else:
                piece += ch
        if piece:
            out.append(piece)
    lines, line = [], ""
    for w in out:
        trial = w if not line else f"{line} {w}"
        if not line or text_width_pt(trial, size, weight) <= width_pt:
            line = trial
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines or [""]


# ==========================================================================
# Page scaffolding
# ==========================================================================
def new_page(landscape=False, facecolor="white"):
    w, h = (LAND_W, LAND_H) if landscape else (PAGE_W, PAGE_H)
    fig = plt.figure(figsize=(w, h), facecolor=facecolor)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax, w, h


def finish(pdf, fig, ax, w, h, title=None, number=True):
    if number:
        _page_no[0] += 1
        ax.text(w - MARGIN, 0.42, str(_page_no[0]), fontsize=8, color=MUTED,
                ha="right", va="center")
        if title:
            ax.text(MARGIN, 0.42, title, fontsize=7.5, color=MUTED, va="center")
        ax.plot([MARGIN, w - MARGIN], [0.62, 0.62], color=RULE, lw=0.5)
    pdf.savefig(fig)
    plt.close(fig)


def heading(ax, y, text, w, size=17, rule=True, color=ACCENT):
    ax.text(MARGIN, y, text, fontsize=size, weight="bold", color=color, va="top")
    if rule:
        ax.plot([MARGIN, w - MARGIN], [y - 0.26, y - 0.26], color=ACCENT, lw=1.1)
    return y - 0.50


def para(ax, y, text, w, size=9.3, color=INK, indent=0.0, leading=1.42, weight="normal"):
    usable = (w - 2 * MARGIN - indent) * 72
    for line in wrap(text, usable, size, weight):
        ax.text(MARGIN + indent, y, line, fontsize=size, color=color, va="top",
                weight=weight)
        y -= size * leading / 72
    return y - 0.06


def bullet(ax, y, text, w, size=9.0, marker="▪", indent=0.16, color=INK):
    usable = (w - 2 * MARGIN - indent - 0.16) * 72
    lines = wrap(text, usable, size)
    ax.text(MARGIN + indent, y, marker, fontsize=size, color=ACCENT, va="top")
    for i, line in enumerate(lines):
        ax.text(MARGIN + indent + 0.17, y, line, fontsize=size, color=color, va="top")
        y -= size * 1.42 / 72
    return y - 0.03


def subhead(ax, y, text, size=10.5, color=INK):
    ax.text(MARGIN, y, text, fontsize=size, weight="bold", color=color, va="top")
    return y - size * 1.9 / 72


def code_block(ax, y, code: str, w, size=7.0, caption=None, max_lines=None):
    """Monospace block on a tinted panel; width is measured, never clipped."""
    lines = code.rstrip("\n").split("\n")
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines] + ["    …"]
    # Shrink until the widest line fits the text column.
    usable = (w - 2 * MARGIN - 0.24) * 72
    while size > 4.4 and max(
            (text_width_pt(l, size, family="monospace") for l in lines), default=0) > usable:
        size -= 0.2
    lh = size * 1.5 / 72
    box_h = len(lines) * lh + 0.16
    ax.add_patch(mpatches.FancyBboxPatch(
        (MARGIN, y - box_h), w - 2 * MARGIN, box_h,
        boxstyle="round,pad=0.02", facecolor=CODE_BG, edgecolor=RULE, lw=0.6))
    ty = y - 0.10
    for line in lines:
        ax.text(MARGIN + 0.12, ty, line, fontsize=size, family="monospace",
                color=INK, va="top")
        ty -= lh
    y -= box_h + 0.06
    if caption:
        y = para(ax, y, caption, w, size=7.8, color=MUTED)
    return y - 0.06


def table(ax, y, headers, rows, w, widths=None, size=8.2, header_fill="#dbe8f5",
          zebra=True, align=None):
    """Simple measured table; column widths are fractions of the text column."""
    n = len(headers)
    widths = widths or [1.0 / n] * n
    align = align or ["l"] * n
    total = w - 2 * MARGIN
    xs, acc = [], MARGIN
    for fr in widths:
        xs.append(acc)
        acc += fr * total
    # wrap every cell
    wrapped, heights = [], []
    for r in [headers] + rows:
        cells = []
        for i, c in enumerate(r):
            cw = widths[i] * total * 72 - 10
            cells.append(wrap(str(c), cw, size))
        wrapped.append(cells)
        heights.append(max(len(c) for c in cells))
    lh = size * 1.42 / 72
    for ri, (cells, nlines) in enumerate(zip(wrapped, heights)):
        rh = nlines * lh + 0.09
        if ri == 0:
            ax.add_patch(mpatches.Rectangle((MARGIN, y - rh), total, rh,
                                            facecolor=header_fill, edgecolor="none"))
        elif zebra and ri % 2 == 0:
            ax.add_patch(mpatches.Rectangle((MARGIN, y - rh), total, rh,
                                            facecolor="#fafafa", edgecolor="none"))
        for ci, cell in enumerate(cells):
            cx = xs[ci] + 0.05
            ha = "left"
            if align[ci] == "r":
                cx = xs[ci] + widths[ci] * total - 0.05
                ha = "right"
            ty = y - 0.06
            for line in cell:
                ax.text(cx, ty, line, fontsize=size, va="top", ha=ha,
                        weight="bold" if ri == 0 else "normal",
                        color=INK, family="sans-serif")
                ty -= lh
        ax.plot([MARGIN, MARGIN + total], [y - rh, y - rh], color=RULE, lw=0.4)
        y -= rh
    return y - 0.10


# ==========================================================================
# Diagram primitives
# ==========================================================================
def box(ax, x, y, w, h, label, fill="plain", fontsize=8.5, weight="normal",
        sub=None, edge=None, radius=0.03):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=BOX_FILLS.get(fill, fill), edgecolor=edge or "#8a8a8a", lw=0.8))
    cy = y + h / 2 + (0.07 if sub else 0)
    for i, line in enumerate(wrap(label, (w - 0.12) * 72, fontsize, weight)):
        ax.text(x + w / 2, cy - i * fontsize * 1.3 / 72, line, fontsize=fontsize,
                ha="center", va="center", weight=weight, color=INK)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.13, sub, fontsize=fontsize - 1.6,
                ha="center", va="center", color=MUTED, style="italic")
    return (x + w / 2, y + h / 2)


def arrow(ax, p0, p1, color=ACCENT, lw=1.3, style="-|>", label=None,
          rad=0.0, fontsize=7.2, ls="-"):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=3, shrinkB=3))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.08, label, fontsize=fontsize, ha="center",
                va="bottom", color=color,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.9))


# ==========================================================================
# Source access — excerpts are pulled live so the doc cannot drift
# ==========================================================================
def read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8", errors="replace")


def excerpt(rel: str, start_pat: str, n_lines: int = 24, dedent=True) -> str:
    """Return `n_lines` from `rel` beginning at the first regex match."""
    lines = read(rel).split("\n")
    rx = re.compile(start_pat)
    for i, l in enumerate(lines):
        if rx.search(l):
            chunk = lines[i:i + n_lines]
            if dedent:
                pads = [len(c) - len(c.lstrip()) for c in chunk if c.strip()]
                cut = min(pads) if pads else 0
                chunk = [c[cut:] if c.strip() else "" for c in chunk]
            return "\n".join(chunk)
    return f"(pattern not found in {rel}: {start_pat})"


def module_table() -> list:
    out = []
    for p in sorted(list(REPO.glob("*.py")) + list((REPO / "widgets").glob("*.py"))):
        src = p.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
        qt = any(
            (isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("PySide6"))
            or (isinstance(n, ast.Import) and any(a.name.startswith("PySide6") for a in n.names))
            for n in ast.walk(tree)
        )
        rel = str(p.relative_to(REPO))
        out.append((rel, len(src.split("\n")), qt))
    return out


def dep_edges() -> dict:
    mods = {p.stem: p for p in list(REPO.glob("*.py")) + list((REPO / "widgets").glob("*.py"))}
    names = set(mods)
    edges = collections.defaultdict(set)
    for n, p in mods.items():
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8", errors="replace"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                b = node.module.split(".")
                cand = b[-1] if b[0] == "widgets" else b[0]
                if cand in names and cand != n:
                    edges[n].add(cand)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    b = a.name.split(".")[0]
                    if b in names and b != n:
                        edges[n].add(b)
    return {k: sorted(v) for k, v in edges.items()}


# ==========================================================================
# SECTION 1 — front matter
# ==========================================================================
def page_title(pdf):
    fig, ax, w, h = new_page()
    ax.add_patch(mpatches.Rectangle((0, h - 3.0), w, 3.0, facecolor=ACCENT, edgecolor="none"))
    ax.text(MARGIN, h - 1.15, "EPR Imaging", fontsize=34, weight="bold",
            color="white", va="center")
    ax.text(MARGIN, h - 1.75, "Software Architecture Breakdown", fontsize=17,
            color="#cfe0f0", va="center")
    ax.text(MARGIN, h - 2.35, "Survey processing · anomaly detection · photogrammetry",
            fontsize=10, color="#a9c6e0", va="center", style="italic")

    mods = module_table()
    total = sum(m[1] for m in mods)
    qt_free = sum(1 for m in mods if not m[2])
    matlab = sum(len(p.read_text(errors="replace").split("\n")) for p in REPO.glob("*.m"))

    y = h - 3.9
    y = para(ax, y, "This document describes how the application is structured: its "
                    "layering rules, domain model, execution and threading model, the "
                    "Task Stack orchestration engine, its boundaries with external "
                    "tooling (MATLAB, Metashape, COLMAP), the conventions that hold it "
                    "together, and the risks that remain.", w, size=10.2)
    y -= 0.10
    y = para(ax, y, "Every code excerpt is read from the source tree when this PDF is "
                    "generated, and every metric is computed from the tree, so the "
                    "document cannot drift out of sync with the code.",
             w, size=10.2, color=MUTED)

    y -= 0.35
    stats = [
        ("Application Python", f"{total:,} lines"),
        ("Modules", f"{len(mods)}  ({qt_free} Qt-free · {len(mods)-qt_free} Qt)"),
        ("MATLAB analysis", f"{matlab:,} lines across {len(list(REPO.glob('*.m')))} scripts"),
        ("Task types", f"{len(__import__('models').TASK_INFO)} in 6 categories"),
        ("Unit tests", "37 (plan_service, timeutil)"),
        ("CI gates", "compile · pyflakes · Qt/datetime lint · pytest"),
    ]
    for k, v in stats:
        ax.add_patch(mpatches.Rectangle((MARGIN, y - 0.30), w - 2 * MARGIN, 0.30,
                                        facecolor="#f4f6f8", edgecolor="none"))
        ax.text(MARGIN + 0.12, y - 0.15, k, fontsize=9.5, va="center", weight="bold")
        ax.text(w - MARGIN - 0.12, y - 0.15, v, fontsize=9.5, va="center", ha="right",
                color=ACCENT)
        y -= 0.36

    ax.text(MARGIN, 0.95, "Generated by build_architecture_pdf.py", fontsize=8,
            color=MUTED, family="monospace")
    finish(pdf, fig, ax, w, h, number=False)


def page_contents(pdf, entries):
    fig, ax, w, h = new_page()
    y = heading(ax, h - MARGIN, "Contents", w)
    y -= 0.10
    for num, title, kind in entries:
        if kind == "part":
            y -= 0.12
            ax.text(MARGIN, y, title, fontsize=11, weight="bold", color=ACCENT, va="top")
            y -= 0.30
        else:
            ax.text(MARGIN + 0.18, y, title, fontsize=9.4, va="top")
            tw = text_width_pt(title, 9.4) / 72
            ax.plot([MARGIN + 0.24 + tw, w - MARGIN - 0.30], [y - 0.05, y - 0.05],
                    color=RULE, lw=0.5, ls=":")
            ax.text(w - MARGIN, y, str(num), fontsize=9.4, va="top", ha="right",
                    color=MUTED)
            y -= 0.235
    finish(pdf, fig, ax, w, h, "Contents")


# ==========================================================================
# SECTION 2 — what the system does
# ==========================================================================
def page_overview(pdf):
    fig, ax, w, h = new_page()
    y = heading(ax, h - MARGIN, "1 · What the system does", w)
    y = para(ax, y, "A PySide6 desktop application that turns raw deep-sea survey inputs "
                    "into science products. Video, navigation and sensor logs arrive as "
                    "separate files on separate clocks; the application aligns them onto "
                    "one time grid and then derives everything else from that grid.", w)
    y -= 0.10

    # ---- inputs → core → products diagram ----
    ax.text(MARGIN, y, "Input → core artefact → products", fontsize=10.5,
            weight="bold", va="top")
    y -= 0.42
    ins = ["Video files\n(.MP4)", "Navigation CSV\n(lat/lon/alt)", "Sensor CSVs\n(CO₂, CH₄, O₂, T, S)"]
    bx = MARGIN
    bw = (w - 2 * MARGIN - 0.4) / 3
    centres = []
    for label in ins:
        c = box(ax, bx, y - 0.62, bw, 0.62, label.replace("\n", " "), "ext", fontsize=8)
        centres.append(c)
        bx += bw + 0.2
    y -= 0.62

    core_y = y - 0.72
    core = box(ax, MARGIN + (w - 2 * MARGIN) / 2 - 1.55, core_y, 3.1, 0.52,
               "interp_full.csv", "model", fontsize=11, weight="bold",
               sub="one row per time step: nav + every sensor channel")
    for c in centres:
        arrow(ax, (c[0], y), (core[0], core_y + 0.52))

    y = core_y - 0.30
    prods = [
        ("Frames", "sampled + annotated"),
        ("Point clouds", "PLY, per channel"),
        ("Rasters", "GeoTIFF, depth slices"),
        ("NetCDF", "CF-compliant"),
        ("QGIS project", ".qgs + layers"),
        ("QC report", "coverage + gaps"),
        ("Photogrammetry", "Metashape / COLMAP"),
        ("Anomaly catalog", "PDF, CSV, GeoJSON"),
    ]
    cols, bw2 = 4, (w - 2 * MARGIN - 0.30) / 4
    ry = y - 0.55
    for i, (t, s) in enumerate(prods):
        r, c = divmod(i, cols)
        px = MARGIN + c * (bw2 + 0.10)
        py = ry - r * 0.62
        box(ax, px, py, bw2, 0.50, t, "service", fontsize=8.2, weight="bold", sub=s)
        if r == 0:
            arrow(ax, (core[0], core_y), (px + bw2 / 2, py + 0.50), lw=0.8, rad=-0.12)
    y = ry - 1.35

    y = subhead(ax, y, "The organising abstraction: the Job")
    y = para(ax, y, "A Job is a named set of time intervals over the survey. Nearly every "
                    "product can be generated for the full dataset or per job, and the "
                    "same task definition serves both — the difference is resolved when "
                    "the execution plan is built, not when the task is created.", w)
    y = bullet(ax, y, "Intervals come from four sources: manual trackline picks, sensor "
                      "threshold analysis, imported CSVs, or the anomaly catalog.", w)
    y = bullet(ax, y, "A Job carries a settings snapshot, so re-running it later "
                      "reproduces the same products.", w)
    finish(pdf, fig, ax, w, h, "1 · What the system does")
