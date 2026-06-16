# install_helper.ps1 — runs inside Inno Setup's [Run] phase
# Receives the install directory as $AppDir.
# Bundled into the installer and deleted after it finishes.

param([string]$AppDir)

$ErrorActionPreference = "Stop"
$LogFile = Join-Path $AppDir "install_log.txt"

function Log {
    param($msg)
    $ts = (Get-Date).ToString("HH:mm:ss")
    $line = "$ts  $msg"
    Add-Content -Path $LogFile -Value $line
}

function Find-Python {
    foreach ($cmd in @("python", "py -3", "python3")) {
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

function Refresh-Path {
    $env:PATH = ([System.Environment]::GetEnvironmentVariable("PATH","Machine")) +
                ";" +
                ([System.Environment]::GetEnvironmentVariable("PATH","User"))
}

# -------------------------------------------------------------------------
Log "=== EPR Sampling Tool — install helper started ==="
Log "Install directory: $AppDir"

# -------------------------------------------------------------------------
# 1.  Ensure Python 3.11+ is present
# -------------------------------------------------------------------------
Log "--- Checking Python ---"
$pythonCmd = Find-Python

if (-not $pythonCmd) {
    Log "Python 3.11+ not found — attempting winget install..."

    $hasWinget = Get-Command winget -ErrorAction SilentlyContinue
    if ($hasWinget) {
        winget install --id Python.Python.3.11 `
            --accept-source-agreements `
            --accept-package-agreements `
            --silent 2>&1 | ForEach-Object { Log $_ }
        Refresh-Path
        $pythonCmd = Find-Python
    }

    # Fallback: direct download from python.org
    if (-not $pythonCmd) {
        Log "winget unavailable or failed — downloading Python installer directly..."
        $tmpInstaller = Join-Path $env:TEMP "python_3.11_setup.exe"
        $url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
        Log "Downloading $url ..."
        Invoke-WebRequest -Uri $url -OutFile $tmpInstaller -UseBasicParsing
        Log "Running Python installer silently..."
        Start-Process -FilePath $tmpInstaller `
            -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0" `
            -Wait
        Remove-Item $tmpInstaller -Force -ErrorAction SilentlyContinue
        Refresh-Path
        $pythonCmd = Find-Python
    }

    if (-not $pythonCmd) {
        $msg = "Could not install Python 3.11+. Please install it manually from https://www.python.org/downloads/ and then re-run this installer."
        Log "FATAL: $msg"
        [System.Windows.Forms.MessageBox]::Show($msg, "EPR Sampling Tool Setup", "OK", "Error") | Out-Null
        exit 1
    }
}

$pyVer = (& cmd /c "$pythonCmd --version" 2>&1)
Log "Using: $pyVer  ($pythonCmd)"

# -------------------------------------------------------------------------
# 2.  Create virtual environment
# -------------------------------------------------------------------------
Log "--- Creating virtual environment ---"
$VenvDir = Join-Path $AppDir "venv"

if (-not (Test-Path $VenvDir)) {
    & cmd /c "$pythonCmd -m venv `"$VenvDir`"" 2>&1 | ForEach-Object { Log $_ }
    Log "Venv created at: $VenvDir"
} else {
    Log "Venv already exists — skipping creation."
}

$PipExe    = Join-Path $VenvDir "Scripts\pip.exe"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $PipExe)) {
    $msg = "Virtual environment creation failed (pip.exe not found). Check install_log.txt in $AppDir for details."
    Log "FATAL: $msg"
    [System.Windows.Forms.MessageBox]::Show($msg, "EPR Sampling Tool Setup", "OK", "Error") | Out-Null
    exit 1
}

# -------------------------------------------------------------------------
# 3.  Install Python packages
# -------------------------------------------------------------------------
Log "--- Installing Python packages ---"
& $PipExe install --upgrade pip --quiet 2>&1 | ForEach-Object { Log $_ }

$ReqFile = Join-Path $AppDir "requirements.txt"
& $PipExe install -r $ReqFile 2>&1 | ForEach-Object { Log $_ }
Log "Package installation complete."

# -------------------------------------------------------------------------
# 4.  Convert PNG logo → .ico  (Pillow is now installed)
# -------------------------------------------------------------------------
Log "--- Creating application icon ---"
$LogoPng = Join-Path $AppDir "whoilogo.png"
$IconIco = Join-Path $AppDir "sampling_tool.ico"

if (Test-Path $LogoPng) {
    $icoScript = @"
from PIL import Image, ImageOps
import sys
img = Image.open(sys.argv[1]).convert("RGBA")
img.save(sys.argv[2], format="ICO",
         sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
"@
    & $PythonExe -c $icoScript $LogoPng $IconIco 2>&1 | ForEach-Object { Log $_ }
    Log "Icon written: $IconIco"
} else {
    Log "whoilogo.png not found — skipping icon."
}

# -------------------------------------------------------------------------
# 5.  Write launch.bat  (the desktop shortcut points at this)
# -------------------------------------------------------------------------
Log "--- Writing launcher ---"
$LaunchBat = Join-Path $AppDir "launch.bat"
$launchContent = @"
@echo off
cd /d "%~dp0"
if not exist "%~dp0venv\Scripts\pythonw.exe" (
    echo EPR Sampling Tool is not set up correctly.
    echo Please reinstall using EPRSamplingToolSetup.exe.
    pause
    exit /b 1
)
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0app.py"
"@
Set-Content -Path $LaunchBat -Value $launchContent -Encoding ASCII
Log "Launcher written: $LaunchBat"

Log "=== Install helper finished successfully ==="
exit 0
