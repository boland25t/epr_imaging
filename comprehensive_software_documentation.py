#!/usr/bin/env python3
"""Build the beginner-oriented, module-by-module EPR Imaging handbook.

The generated PDF deliberately depends only on matplotlib, already required by
the application.  Module inventories and code excerpts are read from the live
source tree so that rebuilding the handbook follows the checked-out code.
"""
from __future__ import annotations

import ast
import datetime as dt
import pathlib
import re
import textwrap

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "docs" / "reports" / "EPR_Imaging_Complete_Software_Documentation.pdf"
W, H = 8.5, 11
ML, MR, TOP, BOTTOM = .72, .64, .68, .72
BLUE, NAVY, INK, GRAY = "#2672a3", "#173f5f", "#222222", "#666666"
PALE, CODE, GREEN, ORANGE = "#eaf3f8", "#f2f4f6", "#dcefdc", "#f7e8cf"


MODULE_PURPOSE = {
    "app.py": ("Application bootstrap", "Creates the Qt application, applies global appearance and icon settings, constructs MainWindow, and enters Qt's event loop."),
    "models.py": ("Domain model", "Defines the small dataclasses that give names and structure to videos, sensor sources, navigation, time intervals, jobs, tasks, and processing history."),
    "timeutil.py": ("UTC compatibility helpers", "Centralizes creation of naive UTC datetimes so old and new Python versions preserve the application's time convention."),
    "video_service.py": ("Video discovery", "Scans a directory, recognizes supported clips, infers each clip's start time, and reads duration and frame-rate metadata with OpenCV."),
    "sensor_service.py": ("Tabular input normalization", "Reads CSV/PPI sources, repairs common coordinate formats, normalizes timestamps, and interpolates sensor and navigation values."),
    "interval_io.py": ("Interval interchange", "Imports time windows from CSV and checks whether those windows fall inside available video and data coverage."),
    "dynamicsampling.py": ("Adaptive frame scheduling", "Chooses extraction times from vehicle motion and altitude so overlapping seafloor imagery is neither too sparse nor unnecessarily dense."),
    "pipeline_service.py": ("Core survey pipeline", "Builds the master interpolated table, finds threshold intervals, extracts frames, annotates images, applies CLAHE, and creates legacy raster products."),
    "output_service.py": ("Science-product façade", "Presents one consistent API for point clouds, rasters, slices, NetCDF, QGIS, QC reports, and per-job filtered tables."),
    "point_cloud_pipeline.py": ("Grid and point-cloud engine", "Projects observations into a regular 3-D grid, fills missing values with IDW, Kriging, or RBF, and writes CSV, PLY, and raster slices."),
    "netcdf_export.py": ("NetCDF writer", "Converts a gridded point table into coordinate arrays and writes a CF-oriented NetCDF file with projection metadata."),
    "qgis_export.py": ("QGIS project writer", "Discovers produced GeoTIFFs and writes QGIS XML containing grouped raster layers and pseudocolor styling."),
    "qc_report.py": ("Data-quality reporting", "Calculates per-channel statistics, gaps, temporal coverage, histograms, and a machine-readable quality summary."),
    "frame_stats.py": ("Image-set quality", "Measures sharpness, brightness, contrast, and clipping for extracted JPEGs and emits CSV/JSON summaries plus plots."),
    "photogrammetry_service.py": ("3-D reconstruction adapters", "Detects Metashape and COLMAP, prepares versioned runs, processes images, seeds camera positions, and exports trajectories and models."),
    "anomaly_service.py": ("Anomaly workflow adapter", "Checks MATLAB availability, invokes the detector, runs the Python catalog builder, loads catalog windows, and converts selected windows into Jobs."),
    "build_anomaly_site_catalog.py": ("Anomaly catalog builder", "Fuses detector event CSVs, adds navigation/video context, clusters spatial sites, writes tables and GeoJSON, and lays out the anomaly review PDF."),
    "plan_service.py": ("Task planner", "Resolves abstract task targets into concrete full-survey or per-job scopes, validates requirements, and expands tasks into executable steps."),
    "stack_runner.py": ("Sequential executor", "Runs a prebuilt plan in a worker thread, dispatches each product type to the correct service, isolates failures, logs outputs, and supports cancellation."),
    "preset_service.py": ("Reusable task templates", "Stores and restores user-named task and stack templates in JSON without coupling them to a particular task identifier."),
    "config_service.py": ("Workspace persistence", "Serializes the complete workspace, restores dataclasses, makes internal paths portable, and detects outputs from older workspaces."),
    "main_window.py": ("Desktop application coordinator", "Builds the principal user interface and coordinates every service, widget, worker thread, workspace state transition, and user workflow."),
    "stack_panel.py": ("Task-stack editor", "Lets users add, reorder, duplicate, edit, remove, save, and run tasks while showing requirement and failure state."),
    "task_config_dialog.py": ("Task settings editor", "Builds type-specific forms for task targets, channels, gridding, sampling, slices, anomaly detection, and photogrammetry."),
    "one_click_dialog.py": ("Pipeline recipe generator", "Collects a high-level set of desired products and converts it into ordinary editable Task objects for the normal execution engine."),
    "workspace_panel.py": ("Workspace browser", "Displays source files and versioned outputs, summarizes metadata, preserves tree expansion, and opens artifacts in native tools."),
    "chat_panel.py": ("Embedded assistant UI", "Renders a compact Markdown-like conversation, assembles context, streams responses, and manages API-key status."),
    "claude_service.py": ("Assistant API worker", "Loads and stores the Anthropic key and streams Claude messages from a QObject worker."),
    "viewer_widget.py": ("Interactive 3-D viewer", "Loads point clouds and meshes into PyVista, controls layer visibility/color/opacity and level of detail, captures images, and exports Potree scenes."),
    "widgets/map_widget.py": ("Interactive survey map", "Projects geographic positions for display, draws tracklines and overlays, provides rectangle selection and point/interval picking, and reports cursor context."),
    "widgets/timeline_widget.py": ("Temporal overview", "Draws video coverage, sensor availability, and selected time ranges on a linked pyqtgraph timeline."),
    "widgets/navigation_import_dialog.py": ("Navigation mapping dialog", "Lets a user map arbitrary file columns to latitude, longitude, altitude, depth, heading, pitch, and roll sources."),
    "widgets/sensor_import_dialog.py": ("Sensor mapping dialog", "Previews a sensor file and maps its timestamp and value columns into one or more named SensorChannel objects."),
    "widgets/annotation_settings_dialog.py": ("Frame-overlay editor", "Configures which navigation/sensor values appear on extracted images and controls font, placement, colors, and background."),
    "3dvistool/point_cloud_pipeline.py": ("Legacy standalone gridding", "An earlier self-contained point-cloud implementation retained for the separate 3dvistool example."),
    "3dvistool/example_usage.py": ("Legacy API example", "Demonstrates the standalone point-cloud pipeline in a short script."),
}

MATLAB_PURPOSE = {
    "GrapherMatrix.m": "Runs the strategy matrix across baselines, smoothers, detectors, channels, and masking regimes.",
    "ExportDetectorFamilyEvents.m": "Exports detector-family event intervals into the CSV contract consumed by Python.",
    "ExportFineConsensusEvents.m": "Exports fine-grained consensus event intervals for catalog fusion.",
    "navSpeed.m": "Derives navigation speed used to diagnose survey motion.",
}

CONCEPTS = {
    "datetime": "A Python object representing a date and clock time. In this program a datetime is deliberately timezone-naive but is interpreted as UTC.",
    "DataFrame": "A pandas table: named columns and numbered rows, similar to a spreadsheet but designed for code and large numerical datasets.",
    "interpolation": "Estimating a value between measured samples. It is what allows navigation, sensor readings, and video frames recorded at different instants to share one timeline.",
    "UTM": "A map projection that converts longitude/latitude angles to local easting and northing distances in metres.",
    "raster": "A rectangular grid of cells. Each cell stores a value such as depth, methane concentration, or temperature.",
    "point cloud": "A collection of points in 3-D space. Every point can also carry a scalar value or RGB color.",
    "signal": "Qt's event notification mechanism. A worker emits a signal; a slot on another object receives the data safely.",
    "dataclass": "A Python class optimized for records. Python generates its constructor and other routine methods from declared fields.",
    "service": "A module containing reusable application logic without the buttons and windows of the GUI.",
    "GeoTIFF": "A TIFF image whose metadata says where its pixels lie on Earth.",
    "NetCDF": "A portable scientific-data file containing named multidimensional arrays and metadata.",
    "IDW": "Inverse-distance weighting: nearby observations influence an estimated grid cell more than distant observations.",
}


def source_files():
    return sorted([*ROOT.glob("*.py"), *(ROOT / "widgets").glob("*.py"), *(ROOT / "3dvistool").glob("*.py")], key=lambda p: str(p.relative_to(ROOT)))


def module_info(path):
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    classes, funcs, imports = [], [], []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            classes.append((node.name, methods))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return src, tree, classes, funcs, imports


def excerpt(path, preferred=None, lines=15):
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    patterns = preferred or (r"^class ", r"^def ", r"^@dataclass")
    start = 0
    for pat in patterns:
        found = next((i for i, line in enumerate(raw) if re.search(pat, line)), None)
        if found is not None:
            start = found
            break
    chunk = raw[start:start + lines]
    nonempty = [len(x) - len(x.lstrip()) for x in chunk if x.strip()]
    cut = min(nonempty) if nonempty else 0
    return "\n".join(x[cut:] for x in chunk)


class Document:
    def __init__(self, pdf):
        self.pdf, self.fig, self.ax = pdf, None, None
        self.page_no = 0
        self.section = ""
        self.y = 0

    def page(self, section=None):
        if self.fig is not None:
            self.finish()
        self.fig = plt.figure(figsize=(W, H), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W); self.ax.set_ylim(0, H); self.ax.axis("off")
        self.page_no += 1
        if section is not None:
            self.section = section
        self.y = H - TOP

    def finish(self):
        if self.fig is None:
            return
        self.ax.plot([ML, W - MR], [.57, .57], color="#c9d1d6", lw=.5)
        self.ax.text(ML, .34, self.section, size=7.2, color=GRAY, va="center")
        self.ax.text(W - MR, .34, str(self.page_no), size=7.5, color=GRAY, ha="right", va="center")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None

    def need(self, height):
        if self.fig is None or self.y - height < BOTTOM:
            self.page()

    def title(self, text, subtitle=None):
        self.need(.8)
        self.ax.text(ML, self.y, text, size=20, weight="bold", color=NAVY, va="top")
        self.y -= .32
        self.ax.plot([ML, W - MR], [self.y, self.y], color=BLUE, lw=1.4)
        self.y -= .19
        if subtitle:
            self.paragraph(subtitle, size=9.2, color=GRAY)

    def heading(self, text, level=2):
        size = 14 if level == 2 else 11
        self.need(.55)
        self.y -= .08
        self.ax.text(ML, self.y, text, size=size, weight="bold", color=BLUE if level == 2 else INK, va="top")
        self.y -= .31 if level == 2 else .25

    @staticmethod
    def wrap(text, size=9.2, indent=0, mono=False):
        chars = int((W - ML - MR - indent) * (10.8 if mono else 12.1) * (9.2 / size))
        out = []
        for para in str(text).split("\n"):
            out.extend(textwrap.wrap(para, width=max(18, chars), break_long_words=True, replace_whitespace=False) or [""])
        return out

    def paragraph(self, text, size=9.2, color=INK, indent=0, leading=1.42):
        lines = self.wrap(text, size, indent)
        lh = size * leading / 72
        for line in lines:
            self.need(lh + .04)
            self.ax.text(ML + indent, self.y, line, size=size, color=color, va="top")
            self.y -= lh
        self.y -= .07

    def bullets(self, items):
        for item in items:
            lines = self.wrap(item, 8.9, .30)
            self.need(len(lines) * .18 + .05)
            self.ax.text(ML + .06, self.y, "•", size=9.2, color=BLUE, va="top")
            for line in lines:
                self.ax.text(ML + .27, self.y, line, size=8.9, color=INK, va="top")
                self.y -= .176
            self.y -= .035

    def code(self, text, caption=None):
        lines = []
        for raw in text.rstrip().splitlines():
            lines.extend(textwrap.wrap(raw, width=102, subsequent_indent="    ", replace_whitespace=False, drop_whitespace=False) or [""])
        lh = .132
        max_lines = max(8, int((self.y - BOTTOM - .4) / lh))
        while lines:
            part, lines = lines[:max_lines], lines[max_lines:]
            height = len(part) * lh + .20
            self.need(height + (.27 if caption else 0))
            self.ax.add_patch(patches.FancyBboxPatch((ML, self.y - height), W - ML - MR, height,
                              boxstyle="round,pad=.02", fc=CODE, ec="#c4ccd1", lw=.6))
            yy = self.y - .10
            for line in part:
                self.ax.text(ML + .11, yy, line, size=6.35, family="monospace", color=INK, va="top")
                yy -= lh
            self.y -= height + .07
            if lines:
                self.page()
        if caption:
            self.paragraph(caption, size=7.8, color=GRAY)

    def callout(self, title, text, fill=PALE):
        lines = self.wrap(text, 8.7, .25)
        height = .34 + len(lines) * .17
        self.need(height + .1)
        self.ax.add_patch(patches.FancyBboxPatch((ML, self.y-height), W-ML-MR, height,
                          boxstyle="round,pad=.02", fc=fill, ec="none"))
        self.ax.text(ML+.12, self.y-.10, title, size=8.8, weight="bold", color=NAVY, va="top")
        yy = self.y-.31
        for line in lines:
            self.ax.text(ML+.12, yy, line, size=8.7, color=INK, va="top"); yy -= .17
        self.y -= height + .12

    def close(self):
        self.finish()


def cover(pdf):
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.add_patch(patches.Rectangle((0, 7.65), W, 3.35, fc=NAVY, ec="none"))
    ax.text(.75, 9.73, "EPR Imaging", size=34, weight="bold", color="white", va="top")
    ax.text(.77, 9.03, "Complete Software Documentation", size=18, color="#cfe7f3", va="top")
    ax.text(.77, 8.52, "A bottom-up guide for Python users", size=11, color="#add2e4", va="top")
    py = sum(len(p.read_text(errors="replace").splitlines()) for p in source_files())
    ml = sum(len(p.read_text(errors="replace").splitlines()) for p in ROOT.glob("*.m"))
    body = ("This handbook explains the program from fundamental vocabulary and stored data, through "
            "services and scientific algorithms, to worker threads and the desktop interface. It covers "
            "every Python module and every MATLAB analysis script, shows representative source excerpts, "
            "and traces complete workflows from imported files to final products.")
    y = 6.85
    for line in textwrap.wrap(body, 76):
        ax.text(.82, y, line, size=11, color=INK, va="top"); y -= .25
    y -= .35
    for label, value in [
        ("Python source", f"{py:,} lines · {len(source_files())} modules"),
        ("MATLAB source", f"{ml:,} lines · {len(list(ROOT.glob('*.m')))} scripts"),
        ("Primary architecture", "Qt GUI → plan → worker → services → artifacts"),
        ("Documentation build", dt.datetime.now().strftime("%Y-%m-%d")),
    ]:
        ax.add_patch(patches.Rectangle((.82, y-.34), 6.85, .34, fc=PALE, ec="none"))
        ax.text(.97, y-.17, label, size=9.5, weight="bold", va="center")
        ax.text(7.52, y-.17, value, size=9.5, color=NAVY, ha="right", va="center")
        y -= .44
    ax.text(.82, .75, "Generated directly from the repository by comprehensive_software_documentation.py",
            size=7.6, color=GRAY, family="monospace")
    pdf.savefig(fig); plt.close(fig)


def architecture_diagram(doc):
    doc.page("System architecture")
    doc.title("System architecture at one glance", "Arrows show the usual direction of control or data.")
    ax = doc.ax
    layers = [
        (8.65, 1.0, PALE, "Desktop interface", "main_window · panels · dialogs · widgets · viewer"),
        (7.38, 1.0, "#e8def0", "Planning and execution", "plan_service → StackWorker → service dispatch"),
        (5.58, 1.45, GREEN, "Qt-free processing services", "pipeline · sensor · video · output · gridding · exports · QC"),
        (3.87, 1.25, ORANGE, "Persistent domain and files", "models · workspace JSON · interp_full.csv · job folders"),
        (2.18, 1.2, "#f2dddd", "External engines and formats", "OpenCV · rasterio · NetCDF · QGIS · MATLAB · Metashape · COLMAP"),
    ]
    for y, h, color, title, sub in layers:
        ax.add_patch(patches.FancyBboxPatch((1.08, y), 6.35, h, boxstyle="round,pad=.03", fc=color, ec="#82919a"))
        ax.text(4.255, y+h*.62, title, size=12, weight="bold", color=NAVY, ha="center", va="center")
        ax.text(4.255, y+h*.30, sub, size=8.1, color=GRAY, ha="center", va="center")
    for y0, y1 in [(8.65, 8.38), (7.38, 7.03), (5.58, 5.12), (3.87, 3.38)]:
        ax.annotate("", xy=(4.25, y1), xytext=(4.25, y0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5))
    doc.y = 1.75
    doc.callout("The central idea", "The interface does not implement scientific algorithms itself. It collects user choices, turns them into model objects, asks the planner for concrete steps, and lets a worker invoke reusable services.", fill=PALE)


def dataflow_diagram(doc):
    doc.page("Data flow")
    doc.title("The main data pipeline")
    ax = doc.ax
    nodes = [
        (.7, 8.9, 1.8, .65, "Video clips", "#f2dddd"), (.7, 7.8, 1.8, .65, "Navigation files", "#f2dddd"),
        (.7, 6.7, 1.8, .65, "Sensor files", "#f2dddd"), (3.25, 7.75, 2.0, .9, "Normalize time\nand columns", PALE),
        (5.95, 7.75, 1.85, .9, "interp_full.csv", ORANGE), (3.25, 5.75, 2.0, .9, "Jobs / intervals", PALE),
        (5.95, 5.75, 1.85, .9, "filtered_interp.csv", ORANGE), (.7, 3.7, 1.8, .75, "Task Stack", "#e8def0"),
        (3.25, 3.7, 2.0, .75, "Concrete plan", "#e8def0"), (5.95, 3.7, 1.85, .75, "StackWorker", "#e8def0"),
    ]
    for x,y,w,h,t,c in nodes:
        ax.add_patch(patches.FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.02",fc=c,ec="#82919a"))
        ax.text(x+w/2,y+h/2,t,size=9,weight="bold",ha="center",va="center")
    arrows=[((2.5,9.2),(3.25,8.2)),((2.5,8.12),(3.25,8.2)),((2.5,7.02),(3.25,8.2)),
            ((5.25,8.2),(5.95,8.2)),((6.88,7.75),(6.88,6.65)),((5.25,6.2),(5.95,6.2)),
            ((1.6,4.45),(1.6,5.75)),((2.5,4.08),(3.25,4.08)),((5.25,4.08),(5.95,4.08))]
    for a,b in arrows: ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="-|>",color=BLUE,lw=1.25))
    products=["annotated frames","PLY point clouds","GeoTIFF rasters","NetCDF grids","QGIS project","QC report","3-D reconstruction","anomaly catalog"]
    x0,y0=.75,1.45
    for i,p in enumerate(products):
        x=x0+(i%4)*1.93; y=y0-(i//4)*.65
        ax.add_patch(patches.FancyBboxPatch((x,y),1.67,.42,boxstyle="round,pad=.015",fc=GREEN,ec="#8ba38b"))
        ax.text(x+.835,y+.21,p,size=7.4,ha="center",va="center")
        ax.annotate("",xy=(x+.835,y+.42),xytext=(6.88,3.7),arrowprops=dict(arrowstyle="-",color="#8ba3b0",lw=.7))


def write_foundations(doc):
    doc.page("Part I · Foundations")
    doc.title("1. How to read this handbook")
    doc.paragraph("Begin with the vocabulary and data model even if your goal is to change a button. The interface is the outermost layer; the meanings of Job, Task, Scope, interval, and interpolated table are established lower down. Once those objects are clear, the larger modules become much less intimidating.")
    doc.callout("Reading source code", "A function definition beginning with def names an operation. A class groups state and related operations. self means the current instance. A return value is the result sent back to the caller. A type annotation such as list[Path] documents the intended kind of value but usually does not perform runtime validation.")
    doc.heading("A minimal Python mental model")
    doc.bullets([
        "Modules are .py files. Importing a module executes its top-level definitions and makes its names available.",
        "Objects carry state. Calling object.method(...) asks that object to perform an operation using its state.",
        "Dataclasses are record-like objects. In EPR Imaging they are the vocabulary shared by the GUI, persistence, planner, and processing services.",
        "Callbacks are functions passed into another function. Services accept log and progress callbacks so they remain independent of Qt.",
        "Exceptions represent failures. Worker objects catch them at the GUI boundary and emit an error signal instead of freezing the interface.",
    ])
    doc.heading("Scientific-library orientation")
    for name in ["DataFrame", "interpolation", "UTM", "raster", "point cloud", "GeoTIFF", "NetCDF", "IDW"]:
        doc.callout(name, CONCEPTS[name], fill="#fafafa")

    doc.page("Part I · Foundations")
    doc.title("2. The problem the software solves")
    doc.paragraph("An underwater survey rarely produces one tidy file. Cameras record frames; navigation instruments record position and attitude; chemical and physical sensors record values; and every instrument samples at its own rate. The first job of the application is therefore temporal alignment. It converts timestamps to one numerical scale, orders and cleans every source, and estimates all requested values at a common series of times.")
    doc.paragraph("The central table is interp_full.csv. Think of it as the application's common language. A row identifies an instant; columns hold file/frame identity, navigation, projected UTM coordinates, and sensor channels. Jobs filter this table to selected intervals. Most scientific products are projections, grids, reports, or images derived from one of these tables.")
    doc.heading("Important invariants")
    doc.bullets([
        "All naive datetime values mean UTC. They must not be interpreted in the computer's local timezone.",
        "Longitude/latitude are geographic coordinates; UTM easting/northing are projected metres. Algorithms that use Euclidean distance require the latter.",
        "A Job is a named collection of closed time ranges plus a settings snapshot. It is not itself an execution thread or output file.",
        "A Task states intent. A plan step is the concrete instruction produced after target and channel expansion.",
        "Services avoid Qt. The GUI/service seam is formed by workers and callback functions.",
        "Outputs are versioned into run directories when overwriting would destroy useful provenance.",
    ])
    doc.code(excerpt(ROOT/"timeutil.py", (r"^def utc_now",), 20), "The two time helpers preserve naive-UTC behavior explicitly.")


def write_workflows(doc):
    doc.page("Part II · End-to-end workflows")
    doc.title("3. Workspace creation and import")
    doc.paragraph("At startup app.py constructs MainWindow. The user selects or creates a workspace directory. MainWindow delegates file inspection to VideoService and SensorService, while the two import dialogs let arbitrary source columns be mapped into the program's domain model. The workspace is saved by ConfigService as JSON, with internal paths stored relative to the workspace whenever possible.")
    doc.bullets([
        "VideoService first attempts configured and common filename patterns, then falls back to file modification time. It reads duration and FPS from OpenCV.",
        "Navigation is modeled as independent time/value sources because latitude, depth, and attitude may reside in different files or use different timestamp columns.",
        "SensorService accepts ordinary delimited tables and PPI files, supports combined or separate date/time columns, and rejects unusable timestamps.",
        "Import dialogs preview data before committing a configuration. The configuration describes how to reload the source; it does not copy the whole table into JSON.",
    ])
    doc.heading("4. Building the master table")
    doc.paragraph("PipelineService._build_full_interp_csv is the convergence point. It creates a time axis, asks SensorService to load each configured source, interpolates values onto that axis, and adds UTM coordinates. The resulting CSV can be regenerated from source configuration and becomes the input to downstream task scopes.")
    doc.code(excerpt(ROOT/"pipeline_service.py", (r"^    def _build_full_interp_csv",), 22),
             "Representative excerpt from the live implementation. The full method also validates sources, updates progress, and writes metadata.")
    doc.heading("5. Selecting work with Jobs")
    doc.paragraph("Users can stage intervals manually on the map, derive them from threshold constraints, import them from CSV, or convert anomaly-catalog windows. A Job stores these intervals. OutputService.generate_job_interval_interps filters the master table and writes a job-local filtered_interp.csv, which makes each job independently inspectable and reproducible.")

    doc.page("Part II · End-to-end workflows")
    doc.title("6. From Task to finished artifact")
    doc.paragraph("The Task Stack is the application's orchestration language. Task objects remain compact and editable. plan_service resolves each target into Scope objects, checks prerequisites, and fans per-channel tasks into multiple step dictionaries. StackWorker then executes those dictionaries sequentially.")
    doc.code(excerpt(ROOT/"plan_service.py", (r"^def build_plan",), 28),
             "The planner is intentionally Qt-free, which makes target expansion testable without starting the desktop application.")
    doc.bullets([
        "Full target: uses interp_full.csv and the workspace output directory.",
        "Job target: uses that job's filtered table and output directory.",
        "All-jobs target: expands once for every valid saved Job.",
        "Per-channel product: expands again for every selected sensor channel.",
        "Skip-existing mode: StackWorker can recognize completed versioned products and avoid recomputation.",
        "Failure isolation: one failed step is recorded; subsequent independent steps may continue.",
    ])
    doc.callout("Critical planning rule", "Targets are resolved before execution begins. A task cannot refer to a Job that a later task will create unless that Job is materialized before the plan is built.", fill=ORANGE)
    doc.heading("Threading")
    doc.paragraph("Long work must not run on Qt's main thread because that thread paints windows and handles input. MainWindow creates a QObject worker, moves it to QThread, connects typed signals to bound QObject methods, and starts the thread. Services receive plain callbacks from the worker and know nothing about widgets.")
    doc.code(excerpt(ROOT/"stack_runner.py", (r"^class StackWorker",), 25), "StackWorker is the deliberate bridge between Qt and the service layer.")

    doc.page("Part II · End-to-end workflows")
    doc.title("7. Product-generation workflows")
    doc.heading("Frames")
    doc.paragraph("PipelineService maps desired timestamps to a source video and frame index, seeks with OpenCV, writes JPEG files, and creates a matching record. Fixed-rate sampling uses a regular clock. Dynamic sampling estimates how far the vehicle has moved and chooses spacing based on altitude and overlap targets. Optional CLAHE improves local contrast; annotations render selected metadata over the image.")
    doc.heading("Point clouds, grids, and rasters")
    doc.paragraph("OutputService builds a PointCloudPipeline from the filtered observations. The pipeline establishes bounds and cell size, creates a regular grid, and fills the grid from measurements. IDW is simple and local; ordinary Kriging models spatial covariance; RBF fits a smooth function. PLY retains 3-D geometry, while GeoTIFF and PNG slices expose 2-D depth bands.")
    doc.heading("NetCDF and QGIS")
    doc.paragraph("netcdf_export reshapes grid rows into coordinate arrays and writes dimensions, variables, fill values, and projection metadata. qgis_export discovers GeoTIFF results and writes QGIS project XML with logical groups and pseudocolor renderers. QGIS therefore consumes normal portable files; it is not embedded into the application.")
    doc.heading("Photogrammetry")
    doc.paragraph("photogrammetry_service is an adapter around two external reconstruction engines. Both workflows collect frames, prepare versioned output directories, use navigation to seed or align camera positions, run engine stages, and export trajectories. Metashape is driven through its Python API; COLMAP is driven through command-line programs and its SQLite database.")
    doc.heading("Anomalies")
    doc.paragraph("MATLAB evaluates detector strategies and exports event CSVs. Python then fuses overlapping evidence, assigns context and severity, clusters nearby windows into sites, links windows to video clips, and produces review tables, GeoJSON/QGIS assets, and a multi-page PDF catalog. Keeping CSV at the boundary means an existing detector run can be recataloged without MATLAB.")


def module_reference(doc):
    doc.page("Part III · Python module reference")
    doc.title("8. How the Python modules connect")
    doc.paragraph("The following pages cover every shipped Python module. 'Imports' names major direct dependencies visible at module scope. The inventories are extracted with Python's ast parser, so nested helper functions are not confused with public module functions.")
    grouped = [
        ("Foundation", ["models.py","timeutil.py","interval_io.py","config_service.py","preset_service.py"]),
        ("Input and processing", ["video_service.py","sensor_service.py","dynamicsampling.py","pipeline_service.py"]),
        ("Products", ["point_cloud_pipeline.py","output_service.py","netcdf_export.py","qgis_export.py","qc_report.py","frame_stats.py"]),
        ("Orchestration", ["plan_service.py","stack_runner.py","anomaly_service.py","photogrammetry_service.py","build_anomaly_site_catalog.py"]),
        ("GUI", ["app.py","main_window.py","stack_panel.py","task_config_dialog.py","one_click_dialog.py","workspace_panel.py","chat_panel.py","claude_service.py","viewer_widget.py"]),
        ("Widgets", ["widgets/map_widget.py","widgets/timeline_widget.py","widgets/navigation_import_dialog.py","widgets/sensor_import_dialog.py","widgets/annotation_settings_dialog.py"]),
        ("Legacy example", ["3dvistool/point_cloud_pipeline.py","3dvistool/example_usage.py"]),
        ("Documentation generator", ["build_architecture_pdf.py","comprehensive_software_documentation.py"]),
    ]
    for group, names in grouped:
        doc.heading(group, 3)
        doc.paragraph(", ".join(names), size=8.6)

    for path in source_files():
        rel = str(path.relative_to(ROOT))
        src, tree, classes, funcs, imports = module_info(path)
        title, purpose = MODULE_PURPOSE.get(rel, ("Documentation utility" if "documentation" in rel or "architecture" in rel else "Supporting module",
                                                  "A repository utility or supporting source module."))
        doc.page(f"Module reference · {rel}")
        doc.title(rel, f"{title} · {len(src.splitlines()):,} source lines")
        doc.heading("Responsibility", 3)
        doc.paragraph(purpose)
        internal = [x for x in imports if (ROOT/(x.split(".")[0]+".py")).exists() or x.startswith("widgets")]
        external = [x for x in imports if x not in internal and x != "__future__"]
        doc.heading("Public shape", 3)
        if classes:
            for cname, methods in classes:
                shown = ", ".join(methods[:20]) + (" …" if len(methods) > 20 else "")
                doc.bullets([f"Class {cname}: {shown or 'record fields only'}"])
        if funcs:
            doc.bullets(["Module functions: " + ", ".join(funcs[:30]) + (" …" if len(funcs)>30 else "")])
        if not classes and not funcs:
            doc.paragraph("This is primarily a script with top-level configuration or execution.")
        doc.heading("Connections", 3)
        doc.paragraph("Internal imports: " + (", ".join(sorted(set(internal))) or "none. This is a leaf or external-facing module."))
        doc.paragraph("Important library imports: " + (", ".join(sorted(set(external))[:18]) or "Python standard library only."))
        if "PySide6" in src:
            doc.callout("Qt boundary", "This module belongs to the GUI/worker side. Widget state must be touched only on the main Qt thread.", fill="#e8def0")
        else:
            doc.callout("Headless behavior", "This module does not import PySide6 and can be called without constructing a QApplication.", fill=GREEN)
        doc.heading("Representative source", 3)
        preferred = None
        if rel == "models.py": preferred=(r"^@dataclass\s*$",)
        elif rel == "main_window.py": preferred=(r"^class MainWindow",)
        elif rel == "pipeline_service.py": preferred=(r"^class PipelineService",)
        elif rel == "output_service.py": preferred=(r"^class OutputService",)
        code = excerpt(path, preferred, 16)
        doc.code(code, "Excerpt copied from the checked-out source. Read surrounding code before changing behavior; the excerpt illustrates structure, not the entire contract.")
        doc.heading("Maintenance notes", 3)
        if rel == "main_window.py":
            doc.paragraph("This is the largest and most coupled module. New scientific logic should normally enter a service first; MainWindow should coordinate it. Worker signal connections deserve special review because a lambda can execute in the emitting thread.")
        elif rel in ("pipeline_service.py","output_service.py"):
            doc.paragraph("Treat paths, columns, units, nodata values, and progress callbacks as part of the public contract. These services are used both from direct GUI actions and the Task Stack.")
        elif rel == "models.py":
            doc.paragraph("Changing a field affects JSON compatibility, dialogs, planner behavior, and workspace reload. Supply defaults and update both serialization directions.")
        elif rel == "plan_service.py":
            doc.paragraph("Planning should remain deterministic and side-effect free. A plan can be inspected, counted, and tested before any expensive task runs.")
        elif rel == "stack_runner.py":
            doc.paragraph("Dispatch names must stay aligned with TASK_INFO and plan step product_type values. Ensure every external resource and log file is finalized on success, error, and cancellation.")
        elif rel.endswith("point_cloud_pipeline.py"):
            doc.paragraph("Gridding parameters encode scientific assumptions. Record method, cell size, neighborhood behavior, coordinate system, and scalar range beside every output.")
        elif rel.startswith("widgets/") or rel.endswith("_dialog.py") or rel.endswith("_panel.py"):
            doc.paragraph("Keep this layer focused on presentation and translation between controls and dataclasses. Expensive reads and computations should move to services or workers.")
        else:
            doc.paragraph("Preserve the module's current boundary. When adding an option, trace it from model/configuration through planning and execution to output metadata.")


def matlab_reference(doc):
    doc.page("Part IV · MATLAB analysis")
    doc.title("9. MATLAB/Python boundary")
    doc.paragraph("MATLAB is optional at application startup but required to calculate a fresh anomaly detector matrix. anomaly_service locates the executable and invokes matlab -batch in a controlled working directory. The exporter scripts write event CSVs; build_anomaly_site_catalog.py consumes those files. This disk contract keeps catalog generation testable and repeatable independently of MATLAB.")
    doc.callout("Why the boundary matters", "Do not replace the event CSV seam with in-memory MATLAB objects unless there is a compelling reproducibility plan. Files provide provenance, debuggability, and graceful degradation.")
    for path in sorted(ROOT.glob("*.m")):
        rel = path.name
        src = path.read_text(encoding="utf-8", errors="replace")
        doc.page(f"MATLAB reference · {rel}")
        doc.title(rel, f"{len(src.splitlines()):,} source lines")
        doc.paragraph(MATLAB_PURPOSE.get(rel, "Supporting MATLAB analysis script."))
        doc.heading("How it participates", 3)
        if rel.startswith("Export"):
            doc.paragraph("This script defines the interchange format. Column names, timestamp formatting, channel names, and interval semantics must remain compatible with build_anomaly_site_catalog.load_events.")
        elif rel == "GrapherMatrix.m":
            doc.paragraph("This is the broad detector experiment driver. It combines baseline, smoothing, detector, and masking choices, calculates agreement/consensus, saves figures and MAT results, and calls the event exporters.")
        elif rel.startswith("CTD") or rel.startswith("ctd"):
            doc.paragraph("This belongs to the CTD background and comparison family. Its outputs inform how chemical and physical anomalies are interpreted relative to water-column structure.")
        else:
            doc.paragraph("This is an exploratory or supporting analysis used by the detector workflow. It operates on files in the analysis working directory rather than importing Python modules.")
        doc.heading("Representative source", 3)
        lines = src.splitlines()
        start = next((i for i,x in enumerate(lines) if x.strip() and not x.strip().startswith("%")), 0)
        doc.code("\n".join(lines[start:start+18]), "MATLAB excerpt from the checked-out script.")
        doc.heading("Beginner translation", 3)
        doc.paragraph("MATLAB arrays are one-indexed, and table variables are accessed by names. A semicolon suppresses console output. Logical masks select matching rows. Functions such as readtable and writetable correspond roughly to pandas.read_csv and DataFrame.to_csv in Python.")


def operations_and_glossary(doc):
    doc.page("Part V · Operations and maintenance")
    doc.title("10. Installation, optional dependencies, and outputs")
    doc.paragraph("The normal developer entry point is python app.py after installing requirements.txt. The Windows installer automates Python and virtual-environment setup. Core imports include PySide6, pandas, NumPy, SciPy, OpenCV, rasterio, pykrige, netCDF4, Pillow, matplotlib, pyqtgraph, PyVista, and PyVistaQt. Metashape, COLMAP, MATLAB, and PotreeConverter are discovered at runtime and remain optional for unrelated workflows.")
    doc.heading("Workspace layout", 3)
    doc.code("""workspace/
  workspace.json                 saved model and settings
  interp_full.csv                aligned whole-survey table
  inputs/ nav/ sensor/           copied or referenced source material
  outputs/                       full-survey products
  job_001_name/
    filtered_interp.csv          only rows in the Job's intervals
    filtered_interp.meta.json    filtering provenance
    outputs/                     job-scoped products
  logs/task_log_TIMESTAMP.txt    execution record""")
    doc.heading("Safe extension checklist", 3)
    doc.bullets([
        "Add or change a dataclass field with a backward-compatible default and update JSON serialization.",
        "Keep scientific work in a Qt-free service and accept log/progress callables.",
        "Register a new product type in TASK_INFO, expose settings in TaskConfigDialog, expand it in plan_service, and dispatch it in StackWorker.",
        "Write output metadata containing source paths, settings, units, coordinate reference system, and software version.",
        "Connect worker signals to QObject methods, arrange quit/deleteLater cleanup, and test cancellation and error paths.",
        "Run byte compilation, static undefined-name checks, the Qt/threading linter, and pytest.",
    ])
    doc.heading("Known concentration of risk", 3)
    doc.paragraph("main_window.py is a very large coordinator, so seemingly local UI edits can cross workspace, worker, and output concerns. pipeline_service.py and photogrammetry_service.py are long procedural workflows whose intermediate-file contracts matter. Scientific gridding and anomaly thresholds require domain validation in addition to ordinary code tests.")

    doc.page("Part V · Operations and maintenance")
    doc.title("11. Testing and debugging")
    doc.paragraph("The current automated tests emphasize plan expansion and UTC semantics. Those are valuable because they protect two application-wide invariants without needing Qt or scientific libraries. A broader test strategy should add fixture-driven sensor parsing, small deterministic interpolation grids, workspace round trips, output metadata checks, and smoke tests for worker dispatch.")
    doc.bullets([
        "Start a failure investigation with the task log and the relevant run's metadata JSON.",
        "Confirm the selected interp path and inspect its timestamp, latitude/longitude, UTM, depth/altitude, and requested sensor columns.",
        "Distinguish missing optional software from bad input data; detection helpers provide explicit availability reasons.",
        "For a map offset, verify coordinate system and UTM zone before tuning visualization.",
        "For an empty Job, compare interval UTC times against master-table coverage and video coverage.",
        "For a frozen or crashing GUI, audit worker signal connections and ensure no widget is touched from a worker callback.",
    ])
    doc.heading("A practical change trace", 3)
    doc.paragraph("Suppose a new gridding option is added. Begin at the output algorithm and its metadata, then expose it in the service signature. Add a Task.settings key and dialog control. Confirm plan_service copies the key into the step. Confirm StackWorker forwards it. Finally, confirm workspace and preset serialization preserve it. This bottom-up trace follows the program's architecture and reveals missing links early.")

    doc.page("Appendix")
    doc.title("Appendix A · Glossary")
    for name, meaning in CONCEPTS.items():
        doc.heading(name, 3)
        doc.paragraph(meaning, size=8.8)
    extra = {
        "Job": "A named collection of survey time intervals and a snapshot of processing settings.",
        "Task": "A user-editable declaration of a desired product, target, channels, and settings.",
        "Scope": "The planner's concrete interpretation of a target: input table, output directory, label, and optional Job.",
        "plan step": "A plain dictionary containing everything StackWorker needs for one unit of execution.",
        "CLAHE": "Contrast Limited Adaptive Histogram Equalization, an OpenCV method that improves local image contrast while limiting noise amplification.",
        "photogrammetry": "Recovering camera poses and 3-D structure from overlapping photographs.",
        "nodata": "A sentinel or mask indicating that a raster/grid cell has no valid observation or estimate.",
    }
    for name, meaning in extra.items():
        doc.heading(name, 3); doc.paragraph(meaning, size=8.8)

    doc.page("Appendix")
    doc.title("Appendix B · Source inventory")
    for path in source_files():
        rel = str(path.relative_to(ROOT))
        src, _, classes, funcs, imports = module_info(path)
        doc.bullets([f"{rel} — {len(src.splitlines()):,} lines; {len(classes)} classes; {len(funcs)} module functions"])
    for path in sorted(ROOT.glob("*.m")):
        doc.bullets([f"{path.name} — {len(path.read_text(errors='replace').splitlines()):,} lines; MATLAB analysis"])


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "EPR Imaging — Complete Software Documentation"
        metadata["Author"] = "Generated from the EPR Imaging source repository"
        metadata["Subject"] = "Bottom-up module reference and architecture handbook"
        metadata["Keywords"] = "EPR Imaging, Python, PySide6, scientific imaging, documentation"
        cover(pdf)
        doc = Document(pdf)
        write_foundations(doc)
        architecture_diagram(doc)
        dataflow_diagram(doc)
        write_workflows(doc)
        module_reference(doc)
        matlab_reference(doc)
        operations_and_glossary(doc)
        doc.close()
    print(OUT)


if __name__ == "__main__":
    build()
