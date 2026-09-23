# Simple UI — simulated first-session usability report

**Subject:** `simple_main.py` + `product_catalog.py` (the three-panel simplified UI)
**Method:** simulated scientist driving the real widgets through Qt's event system
(`QT_QPA_PLATFORM=offscreen`, PySide6 `QtTest`), 2026-09-18, 13:52–14:04 local.
**Screenshots:** `/tmp/claude-1000/-home-troyboland-epr-imaging-epr-imaging/b42d433a-1c4c-483f-85d3-887aa7ad2ce4/scratchpad/shots/`
(referenced below by filename).

## Test setup

A throwaway copy of J1756 was used — the real `/mnt/f` workspaces were never
written to, and the user's live app instance was never touched.

| | |
|---|---|
| Scratch workspace | `…/scratchpad/sim_ws.eprproj` (17 MB) |
| Copied from | `/mnt/f/EPR_2026_PROCESSED/J1756_down.eprproj` |
| Copied content | `workspace.json`, `inputs/interp_full.csv` (14 MB, 59,760 nav vertices), `survey/spectrum_trackline/`, `survey/nav_trackline/`, `survey/sensor_2d/Temperature/` (run_001+run_002) |
| Path fixes | `workspace_path` repointed at the scratch dir; `video_directory` repointed at a deliberately absent `NO_VIDEO_DIR` |
| Absent on purpose | video directory, any frame set, any photogrammetry run, any anomaly output |
| `simple_jobs` at start | none |

### File versions under test

Both modules were imported fresh at the start of each session part and their
mtimes recorded. **`simple_main.py` changed under the test mid-session**
(13:39:18 → 13:59:08, between part 2 and part 4); the visible difference was a
new on-canvas hint line *"wheel = zoom · right-drag = pan · left click ×2 =
interval"* (see `28_default_run_all_finished.png`), which is an improvement and
is credited below. `product_catalog.py` stayed at 13:38:00 throughout.

### Where QTest could not reach (full disclosure)

Everything below was driven by real Qt events unless listed here.

| Interaction | Why direct driving failed | What was done instead |
|---|---|---|
| Tree double-click | `QTest.mouseDClick` alone does not make `QAbstractItemView` emit `itemDoubleClicked`. **Verified identical on a bare `QTreeWidget` control** — not an app defect. | Prefixed a real single click, then `mouseDClick`. Works. |
| Tree right-click | The offscreen platform plugin does not synthesise Qt's `ContextMenu` event from a raw button release. | Posted a real `QContextMenuEvent` through the same event queue. |
| Menu-bar items | `Alt+F` / `Alt+R` never posted the popup offscreen ("This plugin does not support grabbing the keyboard"). | Triggered the very `QAction` the menu item is wired to. The popups themselves were grabbed and read (`25_run_menu.png`). |
| Mouse wheel zoom | `QtTest` has no wheel helper. | Posted a real `QWheelEvent`. |
| Splitter extremes | — | The hard-left collapse **was** a real `QTest` press/move/release drag on the handle. The `[1400,20,20]` and `[0,1600,0]` cases were `setSizes()` calls. |
| MATLAB detector | MATLAB **is** installed on this machine, and `anomaly_service.matlab_unavailable_reason()` really returns `None`, so `Default run on all` would have launched a real MATLAB run against the scratch workspace. | **Stubbed `anomaly_service.matlab_unavailable_reason` to return a blocked reason.** This exercises the documented "engine unavailable" path. The Metashape gate was **not** stubbed — `metashape_driver()` really returned `'subprocess'` and the run was allowed to reach it. |

No `QFileDialog` stub was needed in the end: *Save workspace as…* and the
*Browse…* buttons were never used, so `~/.epr_simple_ui.json` was never written
by this test.

**External-process guard.** A background thread polled
`pgrep -af 'matlab|MATLAB|MetashapePro|metashape|Metashape'` every 250 ms
throughout the `Default run on all`, with a pre-run baseline, and would have
SIGKILLed any new match **descended from the test process only**. One
pre-existing user-owned `metashape.exe` (pid 1004062, running
`params_longstrips300.json`) was in the baseline and was correctly left alone.
**Result: zero new MATLAB or Metashape processes. Nothing had to be killed.**

---

## The session, as it happened

### 13:52:59 — Step 1: launch. "Where do I start?"

`MainWindow` came up in under a second on the scratch workspace.
Title *"EPR Imaging — sim_ws.eprproj"*, 1620×940, three panes at
`[300, 982, 330]`, status bar `ready`, three log lines: workspace path,
`loaded workspace.json (1 sensor file(s))`, `trackline: 59,760 vertices`.
The trackline drew immediately — a long north–south transit with a dense
station cluster at the south end.

**Screenshot `38_launch_clean_no_harness_artifact.png`** (and
`01_launch_main_window.png`, which also shows a stray `x` menu — that is *my*
harness artifact, ignore it).

Two things a newcomer meets straight away:

- **Nothing on screen says what to do.** There is a `Job` combo reading
  "Whole trackline", a `Products` tree, a map, and an empty-ish log. The
  intended entry point — the bolded `Default run on all` — is invisible until
  you open the Run menu. The one paragraph that actually explains the workflow
  exists only under **Help ▸ About** (`30_about_dialog.png`): *"Pick a job,
  expand a product type, generate; or Run ▸ Default run on all."* That sentence
  belongs on the screen, not in an About box.
- **All nine product-type names are cut off.** Measured: column 0 is **100 px**
  wide at launch while the labels need **101–194 px**. With
  `ElideNone` there is not even an "…" to hint at it, and no horizontal
  scrollbar appears. The user reads *"Photogramm"*, *"Sensor Raste"*,
  *"Spectrum Tra"* inside a 300 px panel that has 200 px of empty space to the
  right of them.

### 13:53:00 — Step 2: browse every product type

Expanding each of the nine types with the keyboard (select + Right) worked
first time and populated instantly. The column resized to 259 px **only once
something was expanded**, which is why the launch state is clipped.

```
Frame Set             -> + Generate New… | (none yet for this job)
Photogrammetry        -> + Generate New… | (none yet for this job)
Nav Trackline         -> + Generate New… | 2026-09-18 12:40 · 59,760 vertices
Depth Raster          -> + Generate New… | (none yet for this job)
Sensor Raster         -> + Generate New… | 2026-09-18 13:49 · Temperature · run_002
                                         | 2026-09-18 13:49 · Temperature · run_001
Anomaly Detection     -> + Generate New… | (none yet for this job)
Spectrum Trackline    -> + Generate New… | 2026-09-18 12:46 · 5 channels · log10 turbo
Anomaly Trackline     -> + Generate New… | (none yet for this job)
Survey Report         -> + Generate New… | (none yet for this job)
```

`02_all_types_expanded.png`. Note what the *user* actually sees there: both
sensor rasters read `… · Temperature · run_00` — **truncated at exactly the
character that distinguishes them.** You cannot tell run_001 from run_002
without dragging a horizontal scrollbar.

Double-clicking the spectrum trackline opened the viewer and logged
`viewing 2026-09-18 12:46 · 5 channels · log10 turbo`
(`10_viewer_spectrum_trackline.png`). The five-channel figure is legible and
genuinely useful — but it is shown at native size in a scroll area with no
fit-to-window, so the **Temperature panel is cut off the right edge** while
large grey margins sit above and below it. The tab title is truncated to 28
characters (`spectrum_whole_20260918T1246`), losing the seconds and the
extension.

Right-click offered exactly **View** and **Open containing folder**
(`03_context_menu.png`). Invoking the latter logged
`opened folder for spectrum_whole_20260918T140051.png` — the action is wired
correctly (the `xdg-open` itself is a no-op offscreen, as expected).

### 13:53:04 — Step 3: pick an interval on the map

Two left clicks on the canvas, computed from data coordinates through
`ax.transData`, landed on vertices 14940 and 20916:

```
[13:53:04] interval start 2026-01-18 05:52:46 UTC
[13:53:05] pending interval 6360 s (1 pending)
```

The click markers **have landed** — pink stars appear at the picked vertices on
the first click, before the second (`04_trackline_after_first_click.png`), and
the pending span draws in cyan (`05_trackline_after_second_click.png`).

But this is where the first real friction shows. Those two clicks were **19
pixels apart** and produced a **1 h 46 m** interval. The whole 16.6-hour dive
is squeezed into ~900 × 700 px, the station work is a knot of self-crossing
loops a few dozen pixels across, and the picker takes the nearest vertex in
*space*. Where the track crosses itself, the nearest vertex can be from a
completely different hour. There is no hover readout of the time under the
cursor, no "you picked 05:52:46 → 07:38:46, 1 h 46 m, 6,412 samples" preview,
and no way to nudge an edge once picked.

Typing a timestamp into the two `QDateTimeEdit`s is the precise alternative,
and both fields are **genuinely UTC** — I verified a 0-second round trip from
`track[0,0]`/`track[-1,0]` through `setDateTime` and back out of
`toSecsSinceEpoch()`, and they prefill to the dive bounds. But they are
six-section spinboxes: typing `20260118114736` as a run of digits gives you a
wrong datetime with no complaint. (My own mangled entry produced a 16-hour
interval that I only caught by reading the epoch back — a scientist would not.)

`Add interval to job` staged both pending intervals (`staged: 2`), and
`Create job` produced **Job 1 (2 intervals)**: the combo switched to it by
itself, the intervals highlighted in orange on the map, the staging counter
reset to 0, and `workspace.json` gained a correct `simple_jobs` entry
(`08_job_created.png`). That whole chain is right.

What is jarring is the state you land in (`09_tree_under_new_job.png`): under
the new job **every one of the nine types now reads "(none yet for this
job)"**. The four products you were looking at thirty seconds ago have
apparently vanished. Nothing says "these exist under Whole trackline".

### 13:57:09 — Step 4: generate for real

Fresh window (it opens on "Whole trackline" regardless of what you last
selected), one `Down` on the job combo to reach Job 1, double-click
`＋ Generate New…` under Spectrum Trackline. The dialog
(`12_generate_dialog_spectrum.png`) is a single `Channel: all` combo plus
`Default Run` / `Run with these settings` / `Cancel`. It does **not say which
job it will run against** — the only scope cue is the combo in a panel that can
be collapsed to nothing. And with the form pre-filled at defaults the two run
buttons are identical, with no hint of the difference.

Pressing `Default Run`, the threading behaved exactly as designed:

```
status bar        -> "running: generating Spectrum Trackline (PNG)"
Run menu action   -> disabled
"Create job"      -> disabled
```

The logger streamed live and the run completed in ~1.9 s:

```
[13:57:10] generate Spectrum Trackline (PNG) — defaults
[13:57:10] === Spectrum Trackline (PNG) — Job 1 (2 intervals) ===
[13:57:11]   5 channel panel(s) over 59,759 samples
[13:57:12]   spectrum trackline -> …/survey/spectrum_trackline/spectrum_job_0001_20260918T135711.png
[13:57:12] generating Spectrum Trackline (PNG) finished → 2026-09-18 13:57 · 5 channels · log10 turbo
```

The new instance appeared under Job 1 on the next refresh, with the file and a
sibling `.meta.json` on disk. **Step 4 passes.**

And here is the second big friction, visible in `crop2_14_generate_finished.png`:
at the default 322 px the Log panel is `NoWrap`, so **every one of those lines
is cut off**. The path of the product you just made reads
`spectrum trackline -> /tmp/clau`. The log is the app's only feedback channel
and it is unreadable without dragging a horizontal scrollbar.

### 13:57:12 — Step 5: job filtering

Switching the combo with the keyboard re-filtered every type correctly:

| Type | Whole trackline | Job 1 |
|---|---|---|
| Spectrum Trackline | `2026-09-18 12:46 …` | `2026-09-18 13:57 …` |
| Nav Trackline | `2026-09-18 12:40 · 59,760 vertices` | (none yet) |
| Sensor Raster | Temperature run_002, run_001 | (none yet) |
| Frame Set | (none yet) | (none yet) |

`16_whole_trackline_lists.png`, `17_job1_lists.png`. Highlighted intervals also
switch (empty for Whole, two spans for Job 1). **Step 5 passes.**

### 13:57:14 — Step 6: save and reopen

`File ▸ Save workspace` logged `saved …/workspace.json`. The file kept all
**60 keys** and its 1 sensor-file config; `video_directory` came back as the
bare `'NO_VIDEO_DIR'` because `ConfigService` stores workspace-relative paths
and re-resolves them on load — I checked, that is by design, not a bug.

A brand-new `MainWindow` on the same directory showed
`['Whole trackline', 'Job 1 (2 intervals)']`, restored both intervals exactly,
and listed the generated product under the reopened job
(`18_reopened_job_persisted.png`). **Step 6 passes.**

### 13:59:12 — Step 7: trying to break it

| Attempt | What happened | Verdict |
|---|---|---|
| `Create job` with nothing staged | `nothing staged — add at least one interval`; no job written | Good |
| Click outside the axes (figure margin) | Silently ignored, no marker | Good |
| Backwards typed interval (From 15:06:48, To 05:09:12) | **Silently swapped and accepted** as a 9 h 57 m interval: `pending interval 35856 s`. No mention that the input was reversed (`19_break_backwards_interval.png`) | Bad |
| Typed interval entirely outside the dive (2030-01-01 → 2030-01-02) | **Accepted**, staged, and `Job 2 (1 interval)` created and persisted (`20_break_out_of_range_job.png`). Every product under it will be empty | Bad |
| …and then delete that bogus job | **There is no way to.** No rename, no delete, no right-click on the job combo. `Job 2` is permanent unless you hand-edit `workspace.json` | Bad |
| Two clicks in opposite **empty corners** of the plot | Both snapped to the nearest track vertex regardless of distance and produced a **13:45:24 → 16:43:13 (2 h 58 m)** interval the user never pointed at (`21_break_empty_corner_clicks.png`) | Bad |
| Drag the Products/Trackline handle hard left | Products panel — job selector *and* the whole product tree — **collapses to 0 px** (`[0, 1282, 330]`). No View menu, no reset-layout, and the handle is now flush with the window edge (`22_break_splitter_left.png`) | Bad |
| Force `[1400, 20, 20]` | Clamped to `[537, 815, 260]`; minimum widths respected, all buttons still visible (`23_break_splitter_squeeze.png`) | Good |
| `Generate New` while a run is active | `a task is already running`, no dialog, no queue, no crash | Good, but see below |
| Close with nothing running | Closes silently. Nothing is lost — `create_job` already wrote to disk | Good |

On "Generate while busy": the guard works, but `GenerateDialog` accepts a
`busy=` flag that disables its two run buttons — and `_generate` returns before
constructing the dialog when busy, so that state is unreachable dead code.
Meanwhile the `＋ Generate New…` rows stay fully enabled-looking, so the user
gets a log line (in the unreadable panel) instead of a greyed row or a
message box.

Also observed here: a **dangling first click is invisible and sticky.** When a
second click missed the axes, `_first` stayed set with one star on the map, and
it survives job switches, `Reset view`, and every redraw — the next click made
minutes later silently pairs with it. The only way to cancel is a button
labelled **"Clear staged"**, which does not mention pending picks or markers.

### 14:00:41 — Step 8: `Run ▸ Default run on all`, no video and no frames

Scope: Whole trackline. The Run menu contains exactly one bolded item
(`25_run_menu.png`). The suite ran **7 seconds** and the UI stayed responsive
throughout (I opened a Generate dialog mid-run and it was correctly refused).

| Step | Outcome |
|---|---|
| interp_full.csv | ok 0.0 s (already present) |
| nav trackline | **ok** 3.1 s → `trackline.geojson` |
| depth raster | **ok** 0.6 s → `nav_depth.tif` (61 × 139 @ 5 m) |
| sensor raster × 5 (CO2, CH4, O2, Salinity, Temperature) | **all ok**, 0.2–3.2 s each |
| frame set | failed — `frame set is empty — video directory unavailable (…/NO_VIDEO_DIR)`, preceded by `!! video directory unavailable … extraction of uncovered spans skipped` |
| photogrammetry | failed — tried to build a frame set first, same reason. **Never reached `run_metashape_batch`** |
| anomaly detection | failed — `MATLAB unavailable: …` (harness-stubbed gate) |
| spectrum trackline | **ok** 1.9 s |
| anomaly trackline | failed — `missing …/survey/anomaly/anomaly_segments_utm.geojson — run anomaly detection (and anomaly_utm.build_utm_layers) first` |
| survey report | failed — `survey report needs the merged orthomosaic (…/ortho_merged.tif); run photogrammetry (and the ortho merge) for this dive first` |

Afterwards: status `ready`, Run action and Create-job re-enabled, tree
re-discovered nine types with the six new instances, and the trackline still
took clicks. **Step 8 passes on the stated criterion** — every step failed or
skipped in isolation, the messages are intelligible and *actionable* (they name
the missing file and the step that produces it), the suite continued, and the
UI came back alive.

**Process guard: no `metashape.exe` and no MATLAB process was started.**
Photogrammetry died at `resolve_frame_set` before the engine call, even though
`metashape_driver()` was live and un-stubbed.

But look at what the scientist is left staring at
(`28_default_run_all_finished.png`): the Log panel is a solid wall of Python
traceback, every line cut off at 322 px, and the failure lines
(`!! frame set FAILED: …`) are in the **same colour as the successes** —
worker output arrives via `log`, not `log_error`, so nothing is red. The final
lines are `DEFAULT RUN COMPLETE (0.2 min)` and `default run on all finished`,
and the status bar says `ready`. **Five of ten steps failed and there is no
indication of that anywhere except 60 lines of clipped traceback.** A user
would reasonably conclude the run succeeded.

### 14:02:19 — Navigation and zoom

Four wheel-ups over the station cluster zoomed 2.9× centred on the cursor;
`Reset view` returned to the home extent
(`34_trackline_zoomed_in.png`, `36_trackline_after_reset.png`). I specifically
checked for view drift — repeating `_redraw()` six times over a zoomed view
moved the limits by **0.0 m**, and clicks while zoomed resolve to the exact
vertex under the cursor. Zoom, pan and pick are all geometrically correct.

---

## UX findings

Severity: **BLOCKER** = cannot complete the task · **MAJOR** = wrong results,
lost work, or a state the user cannot understand/undo · **MINOR** = real
friction with a workaround · **POLISH** = cosmetic.

**BLOCKER: 0**

| # | Sev | Finding | Evidence |
|---|---|---|---|
| 1 | MAJOR | **All nine product-type names are clipped on launch.** Column 0 is 100 px; labels need 101–194 px. `ElideNone` means no "…", and no h-scrollbar appears. The column is only sized in `_populate`, so it widens on first expand (259 px) and **shrinks again on every job switch / post-run refresh** (259 → 173 → 259). | `38_launch_clean_no_harness_artifact.png`, `01`, `28`, `39_tree_column_after_job_switch.png` |
| 2 | MAJOR | **The Log — the app's only feedback channel — is unreadable at default width.** `NoWrap` at 322 px cuts every path and message; the output path of a product you just generated is invisible. | `crop2_14_generate_finished.png`, `28` |
| 3 | MAJOR | **Failures are indistinguishable from successes.** Worker lines go through `log`, not `log_error`, so `!! X FAILED` is not red; each failure adds 10–20 lines of raw Python traceback; there is no end-of-run summary ("5 of 10 steps failed") and the status bar returns to plain `ready`. | `28` |
| 4 | MAJOR | **A backwards typed interval is silently swapped, not rejected.** `_add_typed` does `sorted((a, b))`. From 15:06:48 / To 05:09:12 became an accepted 35,856 s interval with no notice. | `19_break_backwards_interval.png` |
| 5 | MAJOR | **No validation against the dive's time bounds.** A 2030-01-01 → 2030-01-02 interval was accepted and persisted as `Job 2 (1 interval)`. Products under it can only come out empty. | `20_break_out_of_range_job.png` |
| 6 | MAJOR | **Jobs cannot be renamed or deleted.** The bogus Job 2 is permanent short of hand-editing `workspace.json`. | `20`, `26_default_run_all_started.png` |
| 7 | MAJOR | **Blank-canvas clicks snap to an arbitrary track vertex.** Two clicks in opposite empty corners yielded a 2 h 58 m interval (13:45→16:43). No proximity tolerance, no hover time readout, and on a self-crossing track the nearest *spatial* vertex can be hours from what the user meant. | `21_break_empty_corner_clicks.png` |
| 8 | MAJOR | **One drag collapses the Products panel (job selector + tree) to 0 px**; the Log collapses the same way. No View menu, no reset-layout, and the handle ends up flush with the window edge. | `22_break_splitter_left.png`, `24_break_splitter_collapse.png` |
| 9 | MAJOR | **The Generate dialog never states its scope.** Identical dialog for "Whole trackline" and for a job; the only cue is a combo in a panel that can be collapsed (finding 8). Easy to spend a long run on the wrong job. | `12_generate_dialog_spectrum.png` |
| 10 | MAJOR | **Selecting a job makes every existing product appear to vanish** — all nine types read "(none yet for this job)" with no hint that whole-track products exist elsewhere. | `09_tree_under_new_job.png` vs `16_whole_trackline_lists.png` |
| 11 | MAJOR | **Nav import demands 0-based column indices into a headerless CSV, with no file preview** — while the *sensor* dialog reads the header and offers dropdowns. The harder of the two is the mandatory one. | `32_import_nav_dialog.png` vs `33_import_sensor_dialog.png` |
| 12 | MAJOR | **A dangling first interval-click is invisible and sticky.** `_first` survives job switches, `Reset view` and redraws; the only way to cancel it is a button labelled "Clear **staged**". | probe transcript; `04_trackline_after_first_click.png` |
| 13 | MINOR | Nothing on screen tells a newcomer what to do; the workflow sentence lives only in Help ▸ About. | `30_about_dialog.png` |
| 14 | MINOR | Job names carry no time information — "Job 1 (2 intervals)", "Job 2 (1 interval)". With five jobs they are unidentifiable. | `26` |
| 15 | MINOR | Instance labels truncate at the distinguishing character: both sensor rasters read "… · Temperature · run_00". | `02_all_types_expanded.png` |
| 16 | MINOR | Viewer has no fit-to-window or zoom: the 5-panel spectrum PNG is clipped on the right (Temperature panel lost) while large grey margins sit above and below. | `10_viewer_spectrum_trackline.png` |
| 17 | MINOR | Viewer tab titles hard-truncated at 28 chars, dropping the extension and the seconds. | `10` |
| 18 | MINOR | `Default Run` and `Run with these settings` are identical until the form is edited, with no explanation of the difference. | `12` |
| 19 | MINOR | `GenerateDialog(busy=…)` disables its run buttons, but `_generate` returns early when busy — unreachable dead code. The user gets a log line in the unreadable panel instead of a greyed row or a message box. | step 7f transcript, `27_generate_while_busy.png` |
| 20 | MINOR | No progress feedback beyond log lines — no busy cursor, no progress bar, no "step 3 of 10". A long run looks the same as a hung one. | `13_generating_busy.png`, `26` |
| 21 | MINOR | `Clear staged` also silently discards *pending* intervals and the click markers; the label mentions only staged ones. | `_clear()` behaviour, `07_trackline_staged.png` |
| 22 | POLISH | Dark-navy matplotlib canvas inside an otherwise light-grey app — mixed theme. | `01`, `38` |
| 23 | POLISH | App messages and worker output share one timestamp prefix and one colour; you cannot tell the UI talking from the pipeline talking. | `28` |
| 24 | POLISH | `Reset view`'s tooltip duplicates the (better) new on-canvas hint line. | `28` |
| 25 | POLISH | No keyboard shortcuts anywhere — not even `Ctrl+S` for Save workspace. | `File` menu enumeration |

**MAJOR: 12 · MINOR: 9 · POLISH: 4**

---

## What a first-time scientist would struggle with

1. **The first ten seconds.** Three panels, a map, and no instruction. The
   thing you are supposed to press is bolded — inside a menu you have no reason
   to open. The explanation is under *Help ▸ About*. One line of placeholder
   text in the empty Log ("Pick a job, expand a product type, and Generate — or
   Run ▸ Default run on all") would fix this outright.
2. **Reading the product list.** On launch, all nine type names are chopped at
   100 px with no ellipsis, inside a panel with 200 px of unused width. Once
   you do expand, the two runs you need to choose between are both labelled
   `… run_00`.
3. **Believing a failed run succeeded.** This is the most dangerous one. After
   `Default run on all`, five of ten products did not exist, but the status bar
   said `ready`, nothing was red, and the evidence was 60 lines of
   horizontally-clipped traceback. The individual error messages are
   genuinely good — they name the missing file *and* the step that makes it —
   and they are buried.
4. **Picking an interval on a 16-hour dive squeezed into 900 px.** The station
   work — the part anyone actually wants to process — is a knot of
   self-crossing loops tens of pixels across. Two clicks 19 px apart gave a
   1 h 46 m window. Clicking near a crossing can silently pick a time hours
   away. There is no time-under-cursor readout and no preview of the interval
   before you commit it.
5. **Typing an interval instead.** The fields *are* correctly UTC and prefill
   to the dive bounds — good. But they are six-section spinboxes, a reversed
   From/To is silently swapped, and an interval in 2030 is accepted and turned
   into a permanent job. A scientist will make a bogus job in their first
   session, and then find there is no way to delete it.
6. **Losing a panel.** One careless drag and the job selector and product tree
   are gone, with no View menu and no reset-layout to get them back.
7. **Importing their own dive.** The nav import dialog asks for eight 0-based
   column indices into a headerless CSV with no preview of the file — while the
   sensor dialog, right next to it in the same menu, reads the header and offers
   dropdowns. The inconsistency teaches the wrong expectation first.

---

## What worked well

- **Jobs are correct and durable.** Click-picked and typed intervals staged
  together, `Create job` auto-selected the new job, highlighted its spans,
  reset the counter, and wrote a clean `simple_jobs` entry. A reopened window
  restored the job, its exact interval bounds, and its products. `create_job`
  never mutated the base job, as specified.
- **Save preserves everything else.** All 60 `workspace.json` keys and the
  sensor-file config survived a save from the new UI; the relative-path
  round-trip through `ConfigService` is correct.
- **UTC is really UTC.** Zero-second round trip on both `QDateTimeEdit`s, and
  they prefill to the dive bounds. I went looking for a timezone bug here and
  there isn't one.
- **Threading is solid.** Live log streaming from the worker, Run menu and
  Create-job disabled while busy, `a task is already running` on a second
  attempt, no double-run, no frozen UI during a 7-second suite, clean return to
  `ready`. The detach-on-close orphan handling is thoughtful.
- **Failure isolation in `default_run_all` is exactly right.** Five failures,
  five continues, and the messages name both the missing artifact and the step
  that produces it. Also: the no-video path fails *before* any external engine
  is touched — photogrammetry died at `resolve_frame_set` even with a live
  Metashape driver.
- **Map navigation is geometrically exact.** Wheel zoom centres on the cursor,
  `Reset view` returns home, and repeated redraws over a zoomed view drift by
  0.0 m. Clicks while zoomed hit the intended vertex.
- **The click markers landed.** A pink star appears at the picked vertex on the
  first click, before the second — precisely the missing feedback that feature
  was for.
- **Instance labels are well designed** where they fit: time + characteristic
  (`5 channels · log10 turbo`, `depth · 5 m cell · run_001`,
  `59,760 vertices`) is exactly what distinguishes runs.
- **Frame recycling logs its reasoning** (`frame pool: empty (nothing to
  recycle)`, `0 of 13,865 samples already on disk — 1 span(s) coalesced from 1
  at min_run=15`) — that transparency will matter a lot once video is present.
- **The new on-canvas hint** (*wheel = zoom · right-drag = pan · left click ×2 =
  interval*) that appeared mid-session is the right pattern; the app needs more
  of exactly that.
