"""chat_panel.py — Claude AI assistant dock panel for the EPR Imaging Tool.

Provides ChatPanel, a QWidget suitable for embedding in a QDockWidget.
Maintains full conversation history, streams responses as they arrive,
and injects a fresh workspace-context summary with each message.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QThread, QTimer, Qt
from PySide6.QtGui import QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from claude_service import ClaudeWorker, load_api_key

_LOGO_PATH = Path(__file__).parent / "whoilogolong.png"


# ---------------------------------------------------------------------------
# System prompt — kept here so it is easy to iterate on without touching UI code
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are an AI assistant embedded in the EPR Imaging Tool, a scientific software \
application used by marine biologists and environmental scientists to process, \
visualize, and analyze underwater sensor data collected during transect surveys.

The tool processes GPS tracklines co-registered with sensor measurements \
(temperature, salinity, dissolved oxygen, turbidity, chlorophyll-a, etc.) that are \
interpolated onto 3D volumetric grids. Users can then generate:
- 2D GeoTIFF raster maps (flat top-down view)
- 3D PLY point clouds (openable in CloudCompare)
- Depth-band slice image stacks (PNG files at each depth interval)

Key workflow:
1. Load nav (GPS) and sensor CSV files on the Inputs tab.
2. Run the pipeline to produce interp_full.csv — the interpolated, UTM-projected dataset.
3. Generate output products from the Outputs tab.
4. Depth slices are secondary products derived from an existing 3D PLY run — \
select a run in the Workspace panel first, then click "Generate Depth Slices."

Your role is to help users — primarily biologists rather than software engineers — \
understand their data, navigate the workflow, interpret their outputs, and make good \
decisions about processing parameters. Use plain, accessible language. Explain \
technical terms the first time you use them. Focus on what the science means, \
not just what buttons to click.

Current workspace state:
{context}"""


# ---------------------------------------------------------------------------
# Minimal markdown → HTML converter (enough for Claude's typical output)
# ---------------------------------------------------------------------------

def _md_to_html(text: str) -> str:
    """Convert a small subset of markdown to HTML suitable for QTextBrowser."""
    # Escape first so subsequent replacements don't double-escape
    text = html.escape(text)

    # Code blocks (``` … ```) — must come before inline code
    text = re.sub(
        r"```(?:\w+)?\n?(.*?)```",
        r'<pre style="background:#1a1a1a;padding:6px;border-radius:4px;">\1</pre>',
        text, flags=re.DOTALL,
    )
    # Inline code
    text = re.sub(r"`([^`]+)`", r'<code style="background:#1a1a1a;padding:1px 4px;">\1</code>', text)

    # Bold and italic
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"\*\*(.+?)\*\*",     r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",         r"<i>\1</i>", text)

    # Headers
    text = re.sub(r"^### (.+)$", r'<b>\1</b>',                              text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$",  r'<span style="font-size:1.1em;"><b>\1</b></span>', text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$",   r'<span style="font-size:1.2em;"><b>\1</b></span>', text, flags=re.MULTILINE)

    # Bullet lists — convert leading "- " or "* " to an HTML bullet
    text = re.sub(r"^[\-\*] (.+)$", r"&bull;&nbsp;\1", text, flags=re.MULTILINE)
    # Numbered lists
    text = re.sub(r"^(\d+)\. (.+)$", r"\1.&nbsp;\2", text, flags=re.MULTILINE)

    # Line breaks
    text = text.replace("\n", "<br>")
    return text


# ---------------------------------------------------------------------------
# ChatPanel widget
# ---------------------------------------------------------------------------

_USER_STYLE = (
    "text-align:right;margin:4px 0 4px 48px;"
)
_USER_BUBBLE = (
    "display:inline-block;background:#1565c0;color:#e3f2fd;"
    "padding:7px 11px;border-radius:9px;"
)
_ASST_STYLE = (
    "text-align:left;margin:4px 48px 4px 0;"
)
_ASST_BUBBLE = (
    "display:inline-block;background:#263238;color:#eceff1;"
    "padding:7px 11px;border-radius:9px;max-width:100%;"
)
_ERR_BUBBLE = (
    "display:inline-block;background:#4e342e;color:#ffccbc;"
    "padding:7px 11px;border-radius:9px;"
)


class ChatPanel(QWidget):
    """Streaming AI chat panel.

    Parameters:
        context_fn: callable that returns a plain-text workspace summary
                    injected into the system prompt on every send.
    """

    def __init__(self, context_fn: Callable[[], str], parent=None) -> None:
        super().__init__(parent)
        self._context_fn = context_fn

        # Conversation history in Anthropic message format
        self._history: list[dict] = []

        # Accumulates streaming text for the in-progress assistant turn
        self._streaming_text: str = ""

        # True while waiting for the first chunk (replaces "Connecting…" placeholder)
        self._first_chunk: bool = False

        # Rendered HTML for all completed turns (rebuilt from _history)
        self._history_html: str = ""

        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[ClaudeWorker]   = None

        self._build_ui()
        self._add_welcome()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Logo at the top of the panel
        if _LOGO_PATH.exists():
            logo_label = QLabel()
            logo_label.setAlignment(Qt.AlignHCenter)
            px = QPixmap(str(_LOGO_PATH))
            logo_label.setPixmap(px.scaledToHeight(72, Qt.SmoothTransformation))
            logo_label.setStyleSheet("padding: 4px 0 0 0;")
            layout.addWidget(logo_label)

        # "Claude Assistant" title below the logo
        title_label = QLabel("Claude Assistant")
        title_label.setAlignment(Qt.AlignHCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 11px; padding: 2px 0 4px 0;")
        layout.addWidget(title_label)

        # API key status
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self._status_label)
        self._refresh_status()

        # Chat view — single QTextBrowser for the full conversation
        self._chat_view = QTextBrowser()
        self._chat_view.setOpenLinks(False)
        self._chat_view.setStyleSheet(
            "background:#121212; border:1px solid #333; border-radius:4px;"
        )
        layout.addWidget(self._chat_view, stretch=1)

        # Input row: text box on left, Send + Clear stacked on right
        input_row = QHBoxLayout()
        self._input = QTextEdit()
        self._input.setPlaceholderText("Ask a question…  (Enter to send, Shift+Enter for newline)")
        self._input.setFixedHeight(58)
        self._input.setAcceptRichText(False)
        self._input.installEventFilter(self)
        input_row.addWidget(self._input, stretch=1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(2)
        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(52)
        self._send_btn.setFixedHeight(27)
        self._send_btn.setStyleSheet("font-weight: bold;")
        self._send_btn.clicked.connect(self._send)
        btn_col.addWidget(self._send_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedWidth(52)
        self._clear_btn.setFixedHeight(27)
        self._clear_btn.setToolTip("Clear conversation history")
        self._clear_btn.clicked.connect(self._clear_conversation)
        btn_col.addWidget(self._clear_btn)

        input_row.addLayout(btn_col)
        layout.addLayout(input_row)

    # ------------------------------------------------------------------
    # Event filter: Enter sends, Shift+Enter adds newline
    # ------------------------------------------------------------------

    def eventFilter(self, source, event) -> bool:
        if source is self._input and event.type() == QEvent.KeyPress:
            key: QKeyEvent = event
            is_enter = key.key() in (Qt.Key_Return, Qt.Key_Enter)
            shift_held = bool(key.modifiers() & Qt.ShiftModifier)
            if is_enter and not shift_held:
                self._send()
                return True
        return super().eventFilter(source, event)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def refresh_status(self) -> None:
        """Re-read the API key and update the status label.  Call after saving a new key."""
        self._refresh_status()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_status(self) -> None:
        if load_api_key():
            self._status_label.setText("Model: claude-sonnet-4-6")
        else:
            self._status_label.setText(
                "⚠ No API key configured.  Use File → Claude API Key to set one."
            )

    def _add_welcome(self) -> None:
        welcome = (
            "Hi! I'm here to help you work with your sensor data.\n\n"
            "You can ask things like:\n"
            "- *What data do I have, and what products can I make?*\n"
            "- *What does the cell size parameter control?*\n"
            "- *How do I generate depth slices from a 3D PLY run?*\n"
            "- *What might cause an anomalous cluster at 15 m depth?*"
        )
        self._history_html += self._render_assistant_bubble(welcome)
        self._render_view()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_user_bubble(self, text: str) -> str:
        escaped = html.escape(text).replace("\n", "<br>")
        return (
            f'<div style="{_USER_STYLE}">'
            f'<span style="{_USER_BUBBLE}">{escaped}</span>'
            f'</div>'
        )

    def _render_assistant_bubble(self, text: str, cursor: bool = False) -> str:
        body = _md_to_html(text)
        if cursor:
            body += '<span style="opacity:0.6;">&#x2588;</span>'
        return (
            f'<div style="{_ASST_STYLE}">'
            f'<span style="{_ASST_BUBBLE}">{body}</span>'
            f'</div>'
        )

    def _render_error_bubble(self, text: str) -> str:
        escaped = html.escape(text).replace("\n", "<br>")
        return (
            f'<div style="{_ASST_STYLE}">'
            f'<span style="{_ERR_BUBBLE}"><b>Error:</b> {escaped}</span>'
            f'</div>'
        )

    def _render_view(self) -> None:
        """Rebuild the full QTextBrowser HTML from committed history + streaming text."""
        body = self._history_html
        if self._streaming_text:
            body += self._render_assistant_bubble(self._streaming_text, cursor=True)
        self._chat_view.setHtml(
            f'<html><body style="background:#121212;color:#eceff1;'
            f'font-family:system-ui,sans-serif;font-size:13px;">'
            f'{body}</body></html>'
        )
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(0, lambda: self._chat_view.verticalScrollBar().setValue(
            self._chat_view.verticalScrollBar().maximum()
        ))

    # ------------------------------------------------------------------
    # Conversation actions
    # ------------------------------------------------------------------

    def _clear_conversation(self) -> None:
        self._history.clear()
        self._history_html = ""
        self._streaming_text = ""
        ack = "Conversation cleared.  How can I help?"
        self._history_html = self._render_assistant_bubble(ack)
        self._render_view()

    def _send(self) -> None:
        user_text = self._input.toPlainText().strip()
        if not user_text:
            return
        api_key = load_api_key()
        if not api_key:
            self._history_html += self._render_error_bubble(
                "No API key configured.  Use File → Claude API Key to set one."
            )
            self._render_view()
            return
        if self._worker_thread and self._worker_thread.isRunning():
            return  # already processing — ignore double-click

        self._input.clear()
        self._send_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)

        # Append user bubble to committed HTML
        self._history_html += self._render_user_bubble(user_text)

        # Show a placeholder so the user sees immediate feedback
        self._streaming_text = "Connecting…"
        self._first_chunk = True
        messages = list(self._history) + [{"role": "user", "content": user_text}]
        context  = self._context_fn()
        system   = _SYSTEM_TEMPLATE.format(context=context)

        self._worker_thread = QThread()
        self._worker = ClaudeWorker(
            api_key=api_key,
            system=system,
            messages=messages,
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.chunk.connect(self._on_chunk)
        self._worker.finished.connect(lambda full: self._on_finished(user_text, full))
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

        self._render_view()

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    def _on_chunk(self, text: str) -> None:
        if self._first_chunk:
            self._streaming_text = text  # drop the "Connecting…" placeholder
            self._first_chunk = False
        else:
            self._streaming_text += text
        self._render_view()

    def _on_finished(self, user_msg: str, full_response: str) -> None:
        # Commit the turn to conversation history
        self._history.append({"role": "user",      "content": user_msg})
        self._history.append({"role": "assistant", "content": full_response})

        # Trim to last 40 messages (20 turns) to avoid token overflow
        if len(self._history) > 40:
            self._history = self._history[-40:]

        # Commit the streaming bubble to permanent HTML
        self._history_html += self._render_assistant_bubble(full_response)
        self._streaming_text = ""
        self._first_chunk    = False
        self._send_btn.setEnabled(True)
        self._clear_btn.setEnabled(True)
        self._render_view()

    def _on_error(self, message: str) -> None:
        self._history_html += self._render_error_bubble(message)
        self._streaming_text = ""
        self._first_chunk    = False
        self._send_btn.setEnabled(True)
        self._clear_btn.setEnabled(True)
        self._render_view()
