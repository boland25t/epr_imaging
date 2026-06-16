#Requires -Version 5.1
<#
.SYNOPSIS
    One-click setup for the EPR Sampling Tool.

.DESCRIPTION
    Run this script once on a fresh Windows machine.  It will:
      1. Install Python 3.11 (via winget) if not already present
      2. Create a virtual environment inside this folder (.\venv\)
      3. Install all Python dependencies from requirements.txt
      4. Convert the WHOI logo to a Windows icon file
      5. Create a desktop shortcut that launches the tool cleanly

.EXAMPLE
    Right-click setup.ps1 -> "Run with PowerShell"
    -- or --
    powershell -ExecutionPolicy Bypass -File setup.ps1
#>

param(
    [switch]$Force  # Wipe and recreate the venv if it already exists
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Step  { param($msg) Write-Host "`n>>> $msg" -ForegroundColor Cyan  }
function Write-OK    { param($msg) Write-Host "    [OK]  $msg" -ForegroundColor Green  }
function Write-Warn  { param($msg) Write-Host "    [!!]  $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "    [ERR] $msg" -ForegroundColor Red    }

function Refresh-Path {
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") +
                ";" +
                [System.Environment]::GetEnvironmentVariable("PATH","User")
}

function Find-Python {
    foreach ($cmd in @("python", "python3", "py -3")) {
        try {
            $raw = (& cmd /c "$cmd --version" 2>&1)
            if ($raw -match "Python (\d+)\.(\d+)") {
                if ([int]$Matches[1] -ge 3 -and [int]$Matches[2] -ge 11) {
                    return $cmd.Trim()
                }
            }
        } catch {}
    }
    return $null
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
$AppName  = "EPR Sampling Tool"
$AppDir   = Split-Path -Parent $MyInvocation.MyCommand.Path   # same dir as setup.ps1
$VenvDir  = Join-Path $AppDir "venv"
$ReqFile  = Join-Path $AppDir "requirements.txt"
$AppPy    = Join-Path $AppDir "app.py"
$LogoPng  = Join-Path $AppDir "whoilogo.png"
$IconFile = Join-Path $AppDir "sampling_tool.ico"
$LaunchBat = Join-Path $AppDir "launch.bat"

# Sanity check — make sure we're in the right folder
if (-not (Test-Path $AppPy)) {
    Write-Fail "app.py not found in $AppDir"
    Write-Fail "Please run setup.ps1 from the folder that contains app.py."
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "  ================================================" -ForegroundColor Cyan
Write-Host "   $AppName — Setup Wizard" -ForegroundColor Cyan
Write-Host "  ================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Step 1 — Python 3.11+
# ---------------------------------------------------------------------------
Write-Step "Checking for Python 3.11+..."

$pythonCmd = Find-Python

if (-not $pythonCmd) {
    Write-Warn "Python 3.11+ not found.  Installing via winget..."

    # Check winget is available
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Fail "winget is not available on this machine."
        Write-Fail "Please install Python 3.11+ manually from https://www.python.org/downloads/"
        Write-Fail "then re-run this setup script."
        Read-Host "Press Enter to exit"
        exit 1
    }

    winget install --id Python.Python.3.11 `
        --accept-source-agreements `
        --accept-package-agreements `
        --silent

    Refresh-Path
    $pythonCmd = Find-Python

    if (-not $pythonCmd) {
        Write-Fail "Python installation did not complete successfully."
        Write-Fail "Please restart this script, or install Python 3.11+ manually"
        Write-Fail "from https://www.python.org/downloads/ and then re-run setup.ps1."
        Read-Host "Press Enter to exit"
        exit 1
    }
}

$verStr = (& cmd /c "$pythonCmd --version" 2>&1)
Write-OK "Using: $verStr  ($pythonCmd)"

# ---------------------------------------------------------------------------
# Step 2 — Virtual environment
# ---------------------------------------------------------------------------
Write-Step "Setting up virtual environment..."

if (Test-Path $VenvDir) {
    if ($Force) {
        Write-Warn "Removing existing venv (--Force specified)..."
        Remove-Item $VenvDir -Recurse -Force
    } else {
        Write-OK "Venv already exists at .\venv\  (run with -Force to recreate)"
    }
}

if (-not (Test-Path $VenvDir)) {
    & cmd /c "$pythonCmd -m venv `"$VenvDir`""
    Write-OK "Created: $VenvDir"
}

$PipExe    = Join-Path $VenvDir "Scripts\pip.exe"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$PythonwExe = Join-Path $VenvDir "Scripts\pythonw.exe"

if (-not (Test-Path $PipExe)) {
    Write-Fail "Virtual environment creation failed.  pip.exe not found at $PipExe"
    Read-Host "Press Enter to exit"
    exit 1
}

# ---------------------------------------------------------------------------
# Step 3 — Install dependencies
# ---------------------------------------------------------------------------
Write-Step "Installing dependencies (this may take 2–5 minutes)..."
Write-Host "    Upgrading pip..." -ForegroundColor DarkGray
& $PipExe install --upgrade pip --quiet

Write-Host "    Installing packages from requirements.txt..." -ForegroundColor DarkGray
& $PipExe install -r $ReqFile

if ($LASTEXITCODE -ne 0) {
    Write-Fail "pip install failed (exit code $LASTEXITCODE)."
    Write-Fail "Check your internet connection and try again."
    Read-Host "Press Enter to exit"
    exit 1
}

Write-OK "All packages installed."

# ---------------------------------------------------------------------------
# Step 4 — Convert PNG logo → .ico  (uses Pillow, which is now installed)
# ---------------------------------------------------------------------------
Write-Step "Creating application icon..."

if (Test-Path $LogoPng) {
    $iconScript = @"
from PIL import Image
img = Image.open(r'$LogoPng').convert('RGBA')
sizes = [(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)]
img.save(r'$IconFile', format='ICO', sizes=sizes)
print('icon written')
"@
    $result = & $PythonExe -c $iconScript 2>&1
    if ($result -match "icon written") {
        Write-OK "Icon: $IconFile"
    } else {
        Write-Warn "Icon conversion skipped ($result).  Shortcut will use default icon."
        $IconFile = $PythonwExe   # fallback to Python icon
    }
} else {
    Write-Warn "whoilogo.png not found — shortcut will use default icon."
    $IconFile = $PythonwExe
}

# ---------------------------------------------------------------------------
# Step 5 — Write launch.bat
# ---------------------------------------------------------------------------
Write-Step "Writing launcher..."

$launchContent = @"
@echo off
REM  EPR Sampling Tool launcher
REM  Double-click this file to start the application.
cd /d "%~dp0"
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0app.py"
"@
Set-Content -Path $LaunchBat -Value $launchContent -Encoding ASCII
Write-OK "Launcher: $LaunchBat"

# ---------------------------------------------------------------------------
# Step 6 — Desktop shortcut
# ---------------------------------------------------------------------------
Write-Step "Creating desktop shortcut..."

$Desktop = [System.Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $Desktop "$AppName.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath     = $LaunchBat
$Shortcut.WorkingDirectory = $AppDir
$Shortcut.Description    = $AppName
$Shortcut.IconLocation   = "$IconFile,0"
$Shortcut.Save()

Write-OK "Shortcut: $ShortcutPath"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "  ================================================" -ForegroundColor Green
Write-Host "   Setup complete!" -ForegroundColor Green
Write-Host "  ================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Double-click  '$AppName'  on your desktop to launch."  -ForegroundColor Green
Write-Host "  Or run: $LaunchBat" -ForegroundColor DarkGray
Write-Host ""

Read-Host "Press Enter to close this window"
