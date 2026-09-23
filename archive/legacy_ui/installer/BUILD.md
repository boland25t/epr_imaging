# Building the EPR Sampling Tool Installer

These steps produce a single `EPRSamplingToolSetup.exe` that end-users download and double-click.

---

## Prerequisites (build machine only — not required by end users)

1. **Inno Setup 6.x** — free download: https://jrsoftware.org/isinfo.php  
   Install with default options.

2. **Python 3.11+ with Pillow** — needed to generate the `.ico` file:
   ```
   pip install Pillow
   ```

---

## Steps

### 1 — Generate the application icon

Run once from the `installer\` folder (or from the repo root):

```
python installer\create_icon.py
```

This writes `installer\app_icon.ico` from `whoilogo.png`.  
Only needs to be re-run if the logo changes.

### 2 — Build the installer

**Option A — GUI (easiest):**
1. Open Inno Setup Compiler
2. File → Open → select `installer\setup.iss`
3. Press **F9** (or Build → Compile)
4. Output: `installer\EPRSamplingToolSetup.exe`

**Option B — Command line:**
```bat
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\setup.iss
```

---

## What the installer does (end-user flow)

| Step | What happens |
|------|-------------|
| Double-click `EPRSamplingToolSetup.exe` | Wizard opens |
| Welcome page | Explains requirements (internet, ~500 MB space) |
| Directory page | Default: `%LOCALAPPDATA%\EPRSamplingTool` (no admin needed) |
| Installing page | Extracts source files, then runs `install_helper.ps1` |
| `install_helper.ps1` | Installs Python 3.11 if missing (winget or direct download), creates venv, runs `pip install -r requirements.txt`, writes `launch.bat`, converts logo to `.ico` |
| Finish page | Desktop shortcut + Start Menu entry created. "View Install Log" button available. |

Total install time: ~3–5 minutes (mostly pip download on first run).

---

## Updating the app

To push an update, just rebuild the installer (`F9`) with the new source files.  
End users run the new `.exe` — it overwrites the source files and re-runs
`pip install -r requirements.txt` to update packages.  
The user's workspace data lives outside the install directory and is untouched.

---

## Troubleshooting

- **Install log:** written to `%LOCALAPPDATA%\EPRSamplingTool\install_log.txt`  
  Also accessible from the "View Install Log" button on the Finish page.
- **rasterio / GDAL errors on pip install:** these resolve themselves with the binary
  wheels on PyPI. If they fail, the user can run `pip install rasterio` manually
  inside `venv\Scripts\` after install.
