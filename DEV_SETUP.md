# Developer Setup — EPR Sampling Tool

This documents how to run **two independent instances** on one Windows PC:

1. **Production install** — the packaged app end users get (`EPRSamplingToolSetup.exe`).
2. **Dev copy** — an editable git clone you keep working on.

Keep them physically separate. **Never edit the installed copy** in
`%LOCALAPPDATA%\EPRSamplingTool` — a reinstall overwrites it and it has no git
history.

---

## Versioning

The single source of truth is **`version.txt`** at the repo root. Bump the
number there and nowhere else:

- the installer reads it at compile time (`setup.iss`),
- the app shows it in the window title (`Sampling Tool  vX.Y.Z`).

---

## Instance A — Production install

Build the installer (on a machine with **Inno Setup 6.x**):

```bat
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\setup.iss
```

Output: `installer\EPRSamplingToolSetup.exe`. Run it → installs to
`%LOCALAPPDATA%\EPRSamplingTool`, creates a venv, installs dependencies, and
adds a desktop shortcut. No admin needed.

(Alternative with no .exe build: copy the source folder somewhere and run
`setup.ps1` — it does the same setup in place.)

---

## Instance B — Dev copy (recommended: native Windows)

**Develop natively on Windows.** On a Windows PC this is simpler and more
capable than WSL:

- PySide6/Qt renders natively — no X server / WSLg blank-window issues.
- The GPU works directly — PyVista/VTK transparency, and CUDA for COLMAP/Metashape.
- **Metashape's Python module imports natively** — no "can't import from WSL" wall.

```bat
git clone <repo-url> C:\dev\epr_imaging
cd C:\dev\epr_imaging\epr_imaging
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python app.py
```

Open `C:\dev\epr_imaging` in VS Code, select `.venv` as the interpreter, edit,
`git commit`. When you want to ship, rebuild the installer from this clone.

### Photogrammetry engines (optional, dev only)

- **COLMAP** — install a CUDA build and put `colmap.exe` on `PATH`. The app
  auto-detects it.
- **Metashape** — install Metashape Professional, then `pip install` Agisoft's
  *Metashape Python 3 Module* wheel (abi3, covers 3.11/3.12) into `.venv`.
  Activate the license. The app then detects and drives it headlessly.

---

## Instance B (alternative) — WSL dev

Only needed if you specifically want a Linux environment (e.g. easier COLMAP
builds, or to mirror a Linux server). It's a **separate, independent instance**:
a clone inside the WSL filesystem with its own Linux venv. It shares nothing
with the Windows install or the Windows dev clone (different filesystems,
different Pythons), so they cannot conflict.

```bash
git clone <repo-url> ~/epr_imaging
cd ~/epr_imaging/epr_imaging
python3.12 -m venv .venv        # or conda create -n epr python=3.12
.venv/bin/pip install -r requirements.txt
QT_QPA_PLATFORM=xcb .venv/bin/python app.py
```

Caveats (why native Windows is preferred): WSLg needs `QT_QPA_PLATFORM=xcb`
(Wayland renders blank); GPU features need the NVIDIA CUDA-on-WSL driver; and
Metashape's Python module is **not** importable from WSL unless a Linux
Metashape is installed there. COLMAP works in WSL with a CUDA build.

---

## Shipping checklist

1. Bump `version.txt`.
2. `python -m py_compile *.py widgets/*.py` (or run the app) — no errors.
3. Rebuild `EPRSamplingToolSetup.exe`.
4. On a clean Windows VM/account: install, launch, confirm the window title
   shows the new version and the app opens to the workspace dialog.
