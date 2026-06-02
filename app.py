# app.py — Application entry point
#
# This is the top-level script that boots the EPR imaging GUI.  It creates the
# Qt application object, instantiates the main window, and hands control to the
# Qt event loop.  Nothing domain-specific lives here; all real logic is in
# main_window.py and the service/widget modules.

from __future__ import annotations

import sys  # Needed to pass command-line arguments to QApplication and to exit cleanly.

from PySide6.QtWidgets import QApplication  # The root Qt object; exactly one must exist per process.

from main_window import MainWindow  # The single top-level window that contains all UI and logic.


def main() -> int:
    """Create the Qt application, show the main window maximised, and run the event loop.

    Returns the integer exit code produced by QApplication.exec(), which is
    forwarded to the OS via SystemExit so that the process signals success (0)
    or failure (non-zero) correctly.
    """

    # QApplication must be constructed before any Qt widgets.  sys.argv lets Qt
    # parse its own command-line flags (e.g. -platform, -style) before the rest
    # of the app sees them.
    app = QApplication(sys.argv)

    # Application name shown in window title bars, OS task-switchers, and
    # platform-specific "About" dialogs.
    app.setApplicationName("EPR Video + Sensor Processing Tool")

    # Build the main window.  The constructor wires all child widgets and
    # connects all internal signals before the window is visible.
    window = MainWindow()

    # Start maximised so the user immediately sees the full split-pane layout
    # without having to resize the window manually.
    window.showMaximized()

    # app.exec() enters the Qt event loop and blocks until the last window is
    # closed or QApplication.quit() is called.  The return value is the exit
    # code we pass back to the shell.
    return app.exec()


# Guard ensures that main() is only called when this file is run directly
# (e.g. `python app.py`), not when it is imported as a module.
if __name__ == "__main__":
    # Wrap in SystemExit so the integer return value becomes the process exit
    # code, following Python's recommended pattern for Qt apps.
    raise SystemExit(main())
