#!/usr/bin/env bash
# Launch the EPR Imaging simple UI on one dive workspace.
#
#   ./launch_simple_ui.sh /path/to/<dive>.eprproj
#
# Environment overrides:
#   EPR_PYTHON        interpreter to use (default: <repo>/.venv/bin/python if present, else python3)
#   QT_QPA_PLATFORM   Qt platform plugin (default on WSL: xcb -- Wayland renders a blank window)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

if [[ $# -lt 1 || -z "${1:-}" || "$1" == "-h" || "$1" == "--help" ]]; then
    echo "usage: $(basename "$0") <workspace.eprproj>" >&2
    echo "  Opens the simple UI on that dive workspace." >&2
    echo "  (Or run '$(basename "$0") --pick' to choose one in a folder dialog.)" >&2
    exit 2
fi

WORKSPACE_ARGS=()
if [[ "$1" != "--pick" ]]; then
    if [[ ! -d "$1" ]]; then
        echo "error: workspace not found: $1" >&2
        exit 1
    fi
    WORKSPACE_ARGS=("$(readlink -f "$1")")
fi

if [[ -n "${EPR_PYTHON:-}" ]]; then
    PY="$EPR_PYTHON"
elif [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
    PY="$REPO_DIR/.venv/bin/python"
else
    PY="$(command -v python3)"
fi

# WSLg: force X11 unless the caller chose a platform. Drop any inherited plugin
# path (e.g. from ~/.bashrc pointing at a different env's PySide6) so Qt loads
# the plugins that belong to this interpreter's PySide6.
if [[ -z "${QT_QPA_PLATFORM:-}" ]] && { [[ -n "${WSL_DISTRO_NAME:-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; }; then
    export QT_QPA_PLATFORM=xcb
fi
unset QT_QPA_PLATFORM_PLUGIN_PATH

cd "$REPO_DIR"
exec "$PY" simple_main.py "${WORKSPACE_ARGS[@]}"
