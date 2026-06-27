import logging
import os
import subprocess
import tempfile
import time
import uuid
from datetime import datetime

from dsg_pddl.pddl_grounding import GroundedPddlProblem
from dsg_pddl.pddl_utils import lisp_string_to_ast

logger = logging.getLogger(__name__)

# Hard wall-clock limits (seconds) passed to Fast Downward so a pathological
# problem can't hang the planner. None disables a limit. Overridable via env.
FD_SEARCH_TIME_LIMIT = os.getenv("OMNIPLANNER_FD_SEARCH_TIME_LIMIT", "60")
FD_TRANSLATE_TIME_LIMIT = os.getenv("OMNIPLANNER_FD_TRANSLATE_TIME_LIMIT", "60")


def solve_pddl(problem: GroundedPddlProblem):
    """Use fast-downward to solve the given pddl problem"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        now = datetime.now()
        key = str(uuid.uuid4())[:8]
        formatted_str = now.strftime("%Y-%m-%d_%H_%M_%S")

        problem_fn = os.path.join(tmpdirname, "problem.pddl")
        debug_problem_fn = os.path.expanduser(
            f"~/omniplanner_problem_{formatted_str}_{key}.pddl"
        )
        domain_fn = os.path.join(tmpdirname, "domain.pddl")
        debug_domain_fn = os.path.expanduser(
            f"~/omniplanner_domain_{formatted_str}_{key}.pddl"
        )
        plan_fn = os.path.join(tmpdirname, "plan.txt")
        debug_plan_fn = os.path.expanduser(
            f"~/omniplanner_plan_{formatted_str}_{key}.pddl"
        )

        with open(problem_fn, "w") as fo:
            fo.write(problem.problem_str)

        with open(debug_problem_fn, "w") as fo:
            fo.write(problem.problem_str)

        with open(domain_fn, "w") as fo:
            fo.write(problem.domain.to_string())

        with open(debug_domain_fn, "w") as fo:
            fo.write(problem.domain.to_string())

        command = ["fast-downward"]
        if FD_TRANSLATE_TIME_LIMIT not in (None, "", "0"):
            command += ["--translate-time-limit", str(FD_TRANSLATE_TIME_LIMIT)]
        if FD_SEARCH_TIME_LIMIT not in (None, "", "0"):
            command += ["--search-time-limit", str(FD_SEARCH_TIME_LIMIT)]
        command += ["--plan-file", plan_fn]
        command += [domain_fn]
        command += [problem_fn]
        command += [
            "--search",
            "let(hff, ff(), let(hcea, cea(), lazy_greedy([hff, hcea], preferred=[hff, hcea])))",
        ]

        logger.info(f"Calling: {command}")
        fd_start = time.perf_counter()
        # Capture FD output instead of letting it stream to the terminal/log;
        # surface it only on failure or at DEBUG.
        proc = subprocess.run(command, capture_output=True, text=True)
        fd_elapsed = time.perf_counter() - fd_start
        logger.info(
            f"fast-downward finished in {fd_elapsed:.3f}s (return code {proc.returncode})"
        )
        logger.debug("fast-downward stdout:\n%s", proc.stdout)
        if proc.stderr:
            logger.debug("fast-downward stderr:\n%s", proc.stderr)

        if os.path.exists(plan_fn):
            with open(plan_fn, "r") as fo:
                lines = fo.readlines()
            with open(debug_plan_fn, "w") as fo:
                fo.writelines(lines)
        else:
            output_dir = os.getenv("ADT4_OUTPUT_DIR", "")
            debug_fn = os.path.join(output_dir, "pddl_problem_debugging.pddl")
            logger.warning(
                f"Planning failed. Please see {debug_fn} for the failed problem file."
            )
            logger.warning("fast-downward stdout:\n%s", proc.stdout)
            logger.warning("fast-downward stderr:\n%s", proc.stderr)
            with open(debug_fn, "w") as fo:
                fo.write(problem.problem_str)
            raise Exception(
                f"Planning failed, please see {debug_fn} for failed problem file."
            )

    plan = [lisp_string_to_ast(line) for line in lines[:-1]]
    return plan
