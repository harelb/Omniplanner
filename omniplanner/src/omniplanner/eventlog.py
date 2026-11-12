"""A mission event log: one JSON object per line, appended to
``$ADT4_RUN_DIR/events/<source>.jsonl``.

Every line carries the same envelope, so the logs of every component (planner,
goal manager, console, executors) can be merged on time and joined on ids::

    {"v": 1, "t_wall": ..., "t_mono": ..., "t_ros": ..., "machine": "willow",
     "robot": "willow", "run_id": "...", "session_id": "...", "mission_id": "...",
     "goal_id": "", "plan_id": "...", "source": "omniplanner",
     "type": "plan.solved", "seq": 12, "payload": {...}}

The writer is deliberately small and defensive:

* off unless ``ADT4_RUN_DIR`` (or ``ADT4_EVENTS_DIR``) is set, so a planner run
  outside ``run-adt4`` writes nothing;
* line-buffered append, fsync at most every ``fsync_s`` seconds;
* rotation at ``max_bytes``: the full file becomes ``<source>.<n>.jsonl``;
* values under secret-looking keys (api_key, password, token, ...) are
  replaced by ``"<redacted>"``;
* it never raises: a failed write is counted and logged once, and planning
  goes on.

No ROS here: ``t_ros`` comes from a clock callable a ROS node may set, and the
mission id is pushed in by whoever knows it (see omniplanner_ros.eventlog_ros).
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_BYTES = 200 * 1024 * 1024
FSYNC_S = 2.0
REDACTED = "<redacted>"

# Keys whose values never reach a log (compared lowercased, '-' read as '_').
REDACT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "openai_api_key",
        "deepgram_api_key",
        "anthropic_api_key",
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
    }
)
REDACT_SUFFIXES = ("_api_key", "_password", "_secret", "_token")


def _is_secret_key(key) -> bool:
    k = str(key).lower().replace("-", "_")
    return k in REDACT_KEYS or k.endswith(REDACT_SUFFIXES)


def redact(value: Any) -> Any:
    """A copy of value with every secret-keyed entry, at any depth, redacted."""
    if isinstance(value, dict):
        return {
            k: (REDACTED if _is_secret_key(k) else redact(v)) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


def _json_default(obj):
    """numpy arrays and scalars, sets and anything else JSON cannot take."""
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(obj, (set, frozenset)):
        try:
            return sorted(obj)
        except TypeError:
            return list(obj)
    return str(obj)


def _finite(value):
    """NaN and inf are not JSON: they become None (also inside numpy arrays)."""
    if callable(getattr(value, "tolist", None)):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def resolve_events_dir(env=None) -> Optional[str]:
    """``$ADT4_EVENTS_DIR``, else ``$ADT4_RUN_DIR/events``, else None (off)."""
    env = os.environ if env is None else env
    explicit = env.get("ADT4_EVENTS_DIR", "").strip()
    if explicit:
        return os.path.expanduser(explicit)
    run_dir = env.get("ADT4_RUN_DIR", "").strip()
    if run_dir:
        return os.path.join(os.path.expanduser(run_dir), "events")
    return None


class EventLog:
    """Appends enveloped events to ``<events dir>/<source>.jsonl``."""

    def __init__(
        self,
        source: str,
        events_dir: Optional[str] = None,
        *,
        env=None,
        max_bytes: int = MAX_BYTES,
        fsync_s: float = FSYNC_S,
        machine: Optional[str] = None,
        robot: Optional[str] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        mission_id: str = "",
        wall_clock: Callable[[], float] = time.time,
        mono_clock: Callable[[], float] = time.monotonic,
    ):
        env = os.environ if env is None else env
        self.source = source
        self.events_dir = events_dir or resolve_events_dir(env)
        self.path = (
            os.path.join(self.events_dir, f"{source}.jsonl")
            if self.events_dir
            else None
        )
        self.max_bytes = int(max_bytes)
        self.fsync_s = float(fsync_s)
        run_dir = env.get("ADT4_RUN_DIR", "").strip()
        self.machine = machine or env.get("ADT4_MACHINE", "") or socket.gethostname()
        self.robot = robot if robot is not None else env.get("ADT4_ROBOT_NAME", "")
        self.run_id = (
            run_id
            if run_id is not None
            else env.get("ADT4_RUN_ID", "")
            or (os.path.basename(os.path.normpath(run_dir)) if run_dir else "")
        )
        self.session_id = (
            session_id if session_id is not None else env.get("ADT4_SESSION_ID", "")
        )
        self.mission_id = mission_id or env.get("ADT4_MISSION_ID", "")
        # Set by a ROS node to its clock (sim time aware); None -> t_ros null.
        self.ros_clock: Optional[Callable[[], float]] = None
        self._wall = wall_clock
        self._mono = mono_clock
        self._lock = threading.Lock()
        self._fh = None
        self._size = 0
        self._last_fsync = 0.0
        self.seq = 0
        self.errors = 0
        self._warned = False

    @property
    def enabled(self) -> bool:
        return self.path is not None

    # ------------- writing -------------

    def emit(
        self,
        type: str,
        payload: Optional[dict] = None,
        *,
        goal_id: str = "",
        plan_id: str = "",
    ) -> Optional[dict]:
        """Write one event. Returns the event written, or None (off or failed).

        Never raises."""
        if self.path is None:
            return None
        try:
            with self._lock:
                event = self._envelope(type, payload, goal_id, plan_id)
                line = (
                    json.dumps(
                        _finite(event),
                        default=_json_default,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                self._write(line)
                return event
        except Exception as exc:  # never into the planner
            self.errors += 1
            if not self._warned:
                self._warned = True
                logger.warning(f"event log {self.path}: cannot write ({exc})")
            return None

    def _envelope(self, type, payload, goal_id, plan_id) -> dict:
        t_ros = None
        if self.ros_clock is not None:
            try:
                t_ros = float(self.ros_clock())
            except Exception:
                t_ros = None
        self.seq += 1
        return {
            "v": SCHEMA_VERSION,
            "t_wall": self._wall(),
            "t_mono": self._mono(),
            "t_ros": t_ros,
            "machine": self.machine,
            "robot": self.robot,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "mission_id": self.mission_id or "",
            "goal_id": goal_id or "",
            "plan_id": plan_id or "",
            "source": self.source,
            "type": type,
            "seq": self.seq,
            "payload": redact(payload or {}),
        }

    def _write(self, line: str) -> None:
        data = line.encode("utf-8")
        if self._fh is None:
            self._open()
        elif self._size and self._size + len(data) > self.max_bytes:
            self._rotate()
        self._fh.write(line)
        self._size += len(data)
        now = self._mono()
        if now - self._last_fsync >= self.fsync_s:
            self._sync(now)

    def _open(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # line-buffered append: each event reaches the OS as it is written
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
        self._size = self._fh.tell()
        if self._size + 1 > self.max_bytes and self._size:
            self._rotate()

    def _rotate(self) -> None:
        """The full file becomes <source>.<n>.jsonl (the first free n)."""
        self._close_fh()
        base = os.path.join(self.events_dir, self.source)
        n = 1
        while os.path.exists(f"{base}.{n}.jsonl"):
            n += 1
        os.replace(self.path, f"{base}.{n}.jsonl")
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
        self._size = 0

    def _sync(self, now=None) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._last_fsync = self._mono() if now is None else now

    def _close_fh(self) -> None:
        if self._fh is not None:
            try:
                self._sync()
            finally:
                self._fh.close()
                self._fh = None

    def flush(self) -> None:
        try:
            with self._lock:
                self._sync()
        except Exception:
            pass

    def close(self) -> None:
        try:
            with self._lock:
                self._close_fh()
        except Exception:
            pass


class _NullLog:
    """What a component without an event log writes to: nothing."""

    enabled = False
    mission_id = ""
    ros_clock = None

    def emit(self, *args, **kwargs):
        return None

    def flush(self):
        pass

    def close(self):
        pass


NULL_LOG = _NullLog()
