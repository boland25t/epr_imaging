#!/usr/bin/env python3
"""
lint_qt_threading.py — guard the two whole-app invariants that static tools miss.

Run from the repository root:

    python3 tools/lint_qt_threading.py            # check, exit 1 on violation
    python3 tools/lint_qt_threading.py --list     # show what it considers a worker signal

CHECK 1 — cross-thread slot affinity
------------------------------------
PySide6 decides where a slot runs from the TYPE of the connected object:

    bound method of a QObject   -> queued onto the receiver's thread   (safe)
    real C++ slot (lbl.setText) -> queued onto the receiver's thread   (safe)
    lambda / plain function     -> runs on the EMITTING thread         (UNSAFE)

A worker signal is emitted on the worker thread, so a lambda connected to it
executes there too.  If that lambda touches a widget — or constructs a QObject
parented to a widget — Qt aborts the process with:

    QObject: Cannot create children for a parent that is in a different thread.

This actually shipped: the per-channel sensor output queue chained itself with a
lambda on `worker.finished`, which then created a QThread and updated widgets
from the worker thread.

Only signals DECLARED ON WORKER CLASSES are flagged.  Connecting a lambda to a
widget signal (`button.clicked`) is same-thread and perfectly fine.

CHECK 2 — deprecated naive-UTC datetime APIs
--------------------------------------------
`datetime.utcnow()` / `datetime.utcfromtimestamp()` are deprecated and slated
for removal.  The obvious replacements return AWARE datetimes, which cannot be
compared or subtracted against this project's NAIVE ones.  Use
`timeutil.utc_now()` / `timeutil.utc_from_timestamp()`, which preserve the naive
semantics exactly.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "__pycache__", "archive", "BioIntervals", ".pytest_cache",
             "tests", "tools", "3dvistool"}

# A class is treated as a worker if it declares Signals and its name says so, or
# it subclasses QObject.  Signals declared there are cross-thread by assumption.
WORKER_HINTS = ("Worker", "Runner", "Task")


def python_files() -> list[Path]:
    out: list[Path] = []
    for path in REPO.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.relative_to(REPO).parts):
            continue
        out.append(path)
    return sorted(out)


def collect_worker_signals(trees: dict) -> set:
    """Signal attribute names declared on QObject/worker classes."""
    signals: set = set()
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                for b in node.bases
            }
            is_worker = ("QObject" in bases) or any(h in node.name for h in WORKER_HINTS)
            if not is_worker:
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                    fn = stmt.value.func
                    name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                    if name == "Signal":
                        for tgt in stmt.targets:
                            if isinstance(tgt, ast.Name):
                                signals.add(tgt.id)
    return signals


def check_slot_affinity(trees: dict, signals: set) -> list:
    """Flag `<...>.<worker signal>.connect(lambda | plain function)`."""
    problems: list = []
    for path, tree in trees.items():
        # Names bound to a `def` in this module are plain functions unless they
        # are methods (which we cannot see from a bare Name anyway).
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "connect"
                    and node.args):
                continue
            signal_expr = node.func.value            # the <...>.<signal> part
            if not isinstance(signal_expr, ast.Attribute):
                continue
            if signal_expr.attr not in signals:
                continue
            # QThread.finished is emitted as the thread winds down; the codebase
            # uses it only for teardown bookkeeping (clearing refs, deleteLater),
            # which is safe.  Skip receivers whose owner is clearly the thread —
            # this check is about WORKER signals.
            owner = signal_expr.value
            owner_name = (owner.id if isinstance(owner, ast.Name)
                          else getattr(owner, "attr", ""))
            if "thread" in owner_name.lower():
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Lambda):
                kind = "lambda"
            elif isinstance(arg, ast.Name):
                kind = f"plain function '{arg.id}'"
            else:
                continue                              # bound method / attribute: safe
            problems.append((
                path, node.lineno,
                f".{signal_expr.attr}.connect({kind}) — runs on the WORKER thread; "
                f"use a bound method of a QObject instead",
            ))
    return problems


def check_deprecated_datetime(trees: dict) -> list:
    problems: list = []
    for path, tree in trees.items():
        if path.name == "timeutil.py":
            continue                                  # documents the old names
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and node.attr in ("utcnow", "utcfromtimestamp")
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "datetime"):
                problems.append((
                    path, node.lineno,
                    f"datetime.{node.attr}() is deprecated — use timeutil."
                    f"{'utc_now' if node.attr == 'utcnow' else 'utc_from_timestamp'}() "
                    "(naive-UTC preserving)",
                ))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="print the discovered worker signal names and exit")
    args = ap.parse_args()

    trees = {}
    for path in python_files():
        try:
            trees[path] = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            print(f"{path.relative_to(REPO)}:{exc.lineno}: SYNTAX ERROR: {exc.msg}")
            return 1

    signals = collect_worker_signals(trees)
    if args.list:
        print("worker signals considered cross-thread:")
        for s in sorted(signals):
            print("   ", s)
        return 0

    problems = check_slot_affinity(trees, signals) + check_deprecated_datetime(trees)
    problems.sort(key=lambda p: (str(p[0]), p[1]))

    for path, line, msg in problems:
        print(f"{path.relative_to(REPO)}:{line}: {msg}")

    print(f"\nscanned {len(trees)} files, {len(signals)} worker signals — "
          f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
