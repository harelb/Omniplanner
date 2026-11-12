"""Where PDDL debug dumps go, and how they are named so none is overwritten.

Every dump gets a timestamp plus a random suffix; a ``*_latest`` name is only
ever a symlink to the newest dump, never a file that is rewritten.

Directories, first set wins:

* problem/domain/plan dumps of every solve: ``$OMNIPLANNER_DUMP_DIR``, then
  ``$ADT4_RUN_DIR/omniplanner``, then ``~/adt4_output/omniplanner``;
* the multi-robot region problem: ``$PDDL_DUMP_DIR``, ``$OMNIPLANNER_DUMP_DIR``,
  ``$ADT4_RUN_DIR/omniplanner``, then ``<cwd>/pddl_dumps`` (as before);
* the problem of a failed solve: the solve dump directory when
  ``$ADT4_RUN_DIR`` is set, else ``$ADT4_OUTPUT_DIR`` (or the cwd), as before.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def run_dump_dir(env=None) -> Optional[str]:
    """``$ADT4_RUN_DIR/omniplanner`` when run-adt4 set a run dir, else None."""
    env = os.environ if env is None else env
    run_dir = env.get("ADT4_RUN_DIR", "").strip()
    return os.path.join(os.path.expanduser(run_dir), "omniplanner") if run_dir else None


def env_dump_dir(*names, env=None) -> Optional[str]:
    """The first of the env dirs named, then the run dir; None if none is set."""
    env = os.environ if env is None else env
    for name in names:
        value = env.get(name, "").strip()
        if value:
            return os.path.expanduser(value)
    return run_dump_dir(env)


def dump_stamp(fmt: str = "%Y-%m-%d_%H_%M_%S") -> str:
    """``<time>_<uuid8>``: unique even for two dumps in the same second."""
    return f"{datetime.now().strftime(fmt)}_{uuid.uuid4().hex[:8]}"


def point_latest(link_path: str, target_path: str) -> None:
    """Make link_path a symlink to target_path (relative when they share a
    directory), replacing whatever was there atomically. Never raises."""
    try:
        directory = os.path.dirname(os.path.abspath(link_path))
        target = os.path.abspath(target_path)
        if os.path.dirname(target) == directory:
            target = os.path.basename(target)
        tmp = os.path.join(directory, f".{os.path.basename(link_path)}.{os.getpid()}")
        if os.path.lexists(tmp):
            os.remove(tmp)
        os.symlink(target, tmp)
        os.replace(tmp, link_path)
    except OSError as exc:
        logger.warning(f"Could not point {link_path} at {target_path}: {exc}")
