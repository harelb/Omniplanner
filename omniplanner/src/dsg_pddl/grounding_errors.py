"""Typed errors for PDDL grounding and solving.

Two failure modes used to be indistinguishable to callers:

  * a goal that references a symbol which is simply *not in the scene graph
    yet* (the robot has to go look for it), and
  * a well-formed goal that the planner cannot achieve.

The first one was silently dropped during grounding (a warning + ``continue``),
which left the dangling symbol in the goal string; Fast Downward then died on
the undeclared object (rc 31) and the caller saw a bare ``Exception``. The
second one surfaced as the same bare ``Exception``. A mission harness needs to
tell them apart -- "explore more" vs. "abort" -- so both seams now raise the
typed errors below.

Every type derives from ``Exception`` (not ``BaseException``), so the blanket
``except Exception`` handlers in ``omniplanner_node.py`` keep logging and
continuing exactly as before; callers that care can catch the specific type.
"""

from dataclasses import dataclass

__all__ = [
    "MissingSymbol",
    "GroundingError",
    "MissingSymbolError",
    "PddlSolverError",
    "PddlUnsolvableError",
    "PddlTimeoutError",
    "PddlMalformedError",
    "kind_hint_for_symbol",
    "error_for_fd_returncode",
]


@dataclass
class MissingSymbol:
    name: str  # e.g. "o123" as it appeared in the goal
    kind_hint: str = ""  # "object" | "region" | "place" | "" when unknown


class GroundingError(Exception): ...


class MissingSymbolError(GroundingError):
    """A PDDL symbol in the goal has no counterpart in the scene graph.

    Two grounding seams raise this, and they know different amounts about the
    failure, so the one class supports two construction forms:

    *Plural* -- ``generate_goal_relevant_pddl`` resolves the whole goal at once
    and can report every unresolved name together::

        MissingSymbolError([MissingSymbol("o42", "object")], goal_string)

    *Singular* -- ``add_symbol_positions`` fails on one specific symbol and
    knows its full DSG identity (the NodeSymbol, the layer char, the index, and
    the underlying lookup exception)::

        MissingSymbolError(symbol=ns, pddl_symbol="o42", dsg_char="O",
                           index=42, original=exc)

    ``missing`` and ``goal`` are always populated (the singular form
    synthesises a one-element ``missing`` list), so a caller can uniformly ask
    "which names are missing?". The singular attributes -- ``symbol``,
    ``pddl_symbol``, ``dsg_char``, ``index``, ``original`` -- are ``None``/``""``
    when only the plural form's information was available.

    The singular form's message deliberately keeps the historical
    "Could not find node <ns> in DSG" wording: downstream consumers
    (agentic_navigation's ``pddl_tools.evaluate_pddl_goal`` and
    ``omniplanner_bridge``) parse it to recognise a missing scene symbol.
    """

    def __init__(
        self,
        missing: list[MissingSymbol] | None = None,
        goal: str = "",
        *,
        symbol=None,
        pddl_symbol: str | None = None,
        dsg_char: str = "",
        index: int | None = None,
        original: Exception | None = None,
    ):
        self.symbol = symbol  # spark_dsg.NodeSymbol, singular form only
        self.pddl_symbol = pddl_symbol  # e.g. "o3"
        self.dsg_char = dsg_char  # e.g. "O"
        self.index = index  # e.g. 3
        self.original = original
        self.goal = goal

        singular = symbol is not None or pddl_symbol is not None
        if missing is None:
            name = pddl_symbol if pddl_symbol is not None else str(symbol)
            missing = [MissingSymbol(name, kind_hint_for_symbol(name or ""))]
        self.missing = missing

        if singular:
            message = (
                f"Could not find node {symbol} in DSG "
                f"(pddl symbol '{pddl_symbol}')"
            )
        else:
            message = (
                "goal references symbols absent from the scene graph: "
                f"{[m.name for m in missing]}"
            )
        super().__init__(message)


class PddlSolverError(Exception):
    def __init__(self, message: str, returncode: int | None = None):
        self.returncode = returncode
        super().__init__(message)


class PddlUnsolvableError(PddlSolverError): ...  # FD rc 10/11/12


class PddlTimeoutError(PddlSolverError): ...  # FD rc 21/23/24 (no plan file)


class PddlMalformedError(PddlSolverError): ...  # FD rc 31


def kind_hint_for_symbol(name: str) -> str:
    """Guess the DSG layer a PDDL symbol name refers to from its leading char.

    PDDL symbols mirror the DSG node symbol prefix: ``o12`` is an object,
    ``r3`` a region/room, ``p57`` a place. Anything else is unknown ("").
    """
    if not name:
        return ""
    match name[0]:
        case "o":
            return "object"
        case "r":
            return "region"
        case "p":
            return "place"
        case _:
            return ""


# Fast Downward driver exit codes (driver/returncodes.py upstream). Only the
# codes we can act on are classified; anything else stays a plain
# PddlSolverError so an unexpected failure is never mistaken for "unsolvable".
FD_UNSOLVABLE_RETURNCODES = frozenset({10, 11, 12})
FD_TIMEOUT_RETURNCODES = frozenset({21, 23, 24})
FD_MALFORMED_RETURNCODES = frozenset({31})


def error_for_fd_returncode(
    returncode: int | None, message: str
) -> PddlSolverError:
    """Map a Fast Downward exit code onto the matching typed solver error."""
    if returncode in FD_UNSOLVABLE_RETURNCODES:
        return PddlUnsolvableError(message, returncode)
    if returncode in FD_TIMEOUT_RETURNCODES:
        return PddlTimeoutError(message, returncode)
    if returncode in FD_MALFORMED_RETURNCODES:
        return PddlMalformedError(message, returncode)
    return PddlSolverError(message, returncode)
