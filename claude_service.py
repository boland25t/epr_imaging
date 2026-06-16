"""claude_service.py — Anthropic API wrapper for the EPR Imaging assistant.

Provides:
  - load_api_key() / save_api_key(): persist the key at ~/.config/epr_imaging/config.json
  - ClaudeWorker(QObject): runs a streaming chat completion in a QThread
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal


_CONFIG_PATH = Path.home() / ".config" / "epr_imaging" / "config.json"
_DEFAULT_MODEL = "claude-sonnet-4-6"


def load_api_key() -> str:
    """Return the Anthropic API key from the config file, or '' if absent."""
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
        return cfg.get("api_key", "")
    except Exception:
        return ""


def save_api_key(key: str) -> None:
    """Write key to the config file, creating parent directories if needed."""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict = {}
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
    except Exception:
        pass
    cfg["api_key"] = key
    _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


class ClaudeWorker(QObject):
    """Runs a single streaming Claude API call in a QThread.

    Instantiate, move to a QThread, and connect started → run.

    Signals:
        chunk(str)      — incremental text as it arrives
        finished(str)   — complete response text on clean completion
        error(str)      — error description if the call fails
    """

    chunk    = Signal(str)
    finished = Signal(str)
    error    = Signal(str)

    def __init__(
        self,
        api_key: str,
        system: str,
        messages: list[dict],
        model: str = _DEFAULT_MODEL,
    ) -> None:
        super().__init__()
        self._api_key  = api_key
        self._system   = system
        self._messages = messages
        self._model    = model

    def run(self) -> None:
        try:
            try:
                import anthropic
            except ImportError:
                self.error.emit(
                    "The 'anthropic' package is not installed. "
                    "Run: pip install anthropic"
                )
                return
            # timeout=30 s: short enough to fail fast offline, long enough for slow connections.
            # For streaming the timeout applies between chunks, not to the entire response.
            client = anthropic.Anthropic(api_key=self._api_key, timeout=30.0)
            accumulated = ""
            with client.messages.stream(
                model=self._model,
                max_tokens=2048,
                system=self._system,
                messages=self._messages,
            ) as stream:
                for text in stream.text_stream:
                    accumulated += text
                    self.chunk.emit(text)
            self.finished.emit(accumulated)
        except Exception as exc:
            self.error.emit(str(exc))
