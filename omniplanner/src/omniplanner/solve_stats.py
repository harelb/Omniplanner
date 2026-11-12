"""What the solvers did for the plan being made, for the planner's event log.

The solvers sit deep under ``full_planning_pipeline``; their timings and the
files they wrote cannot travel back through the plan objects without changing
every planner's signature. Instead each solver records here, and the node that
asked for the plan collects the records afterwards. Recording is per thread
(the node plans in one thread per goal), and a no-op unless a collection was
started: library users and tests see nothing.

    solve_stats.start()
    plans = full_planning_pipeline(...)
    records, files = solve_stats.finish()
"""

from __future__ import annotations

import threading
from typing import Dict, List, Tuple

_local = threading.local()


def start() -> None:
    """Begin collecting for the plan this thread is about to make."""
    _local.records = []
    _local.files = {}


def record(engine: str, **stats) -> None:
    """One solver call: its engine (FD, GRSTAPS, ...) and what it reports."""
    records = getattr(_local, "records", None)
    if records is not None:
        records.append({"engine": engine, **stats})


def note_file(kind: str, path: str) -> None:
    """A file written for this plan (a PDDL dump), by kind; the latest wins."""
    files = getattr(_local, "files", None)
    if files is not None and path:
        files[kind] = path


def finish() -> Tuple[List[dict], Dict[str, str]]:
    """The records and files since start(); collection stops."""
    records = getattr(_local, "records", None) or []
    files = getattr(_local, "files", None) or {}
    _local.records = None
    _local.files = None
    return records, files
