"""Who has an arm, and which objects are too heavy to lift.

Shared by both planners -- Fast Downward's multirobot grounding (static
``(has-arm r)`` / ``(heavy o)`` facts) and GRSTAPS-X (species without a
manipulator, drag tasks instead of relocations) -- so "heavy" and "has an arm"
mean the same thing whichever planner the operator picked.

Arms: a robot has one unless its type says otherwise (husky, phoenix, jackal,
uav, drone, quadrotor) or ``OMNIPLANNER_NO_ARM_ROBOTS`` names it. With no robot
types at all every robot has an arm, which is the behaviour before this module.

Heavy objects are dragged rather than picked up: an object is heavy when its
class (the node's name, or its labelspace category) is one of
``heavy_classes()``, when ``OMNIPLANNER_HEAVY_SYMBOLS`` names it, or when the
goal lists it explicitly.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# robot_type (omniplanner_plugins.yaml) of platforms without a manipulator
NO_ARM_ROBOT_TYPES = frozenset(
    {"husky", "phoenix", "jackal", "uav", "drone", "quadrotor"}
)
NO_ARM_ROBOTS_ENV = "OMNIPLANNER_NO_ARM_ROBOTS"

DEFAULT_HEAVY_CLASSES = (
    "bag",
    "sandbag",
    "box",
    "trash",
    "trash can",
    "suitcase",
    "duffel bag",
    "backpack",
)
HEAVY_CLASSES_ENV = "OMNIPLANNER_HEAVY_CLASSES"
HEAVY_SYMBOLS_ENV = "OMNIPLANNER_HEAVY_SYMBOLS"


def _env_list(name):
    raw = os.environ.get(name)
    if raw is None:
        return None
    return [s.strip().lower() for s in raw.split(",") if s.strip()]


def robot_has_arm(robot_name, robot_types=None):
    """Whether ``robot_name`` can grasp: pick, place and drag."""
    no_arm = set(_env_list(NO_ARM_ROBOTS_ENV) or [])
    name = str(robot_name or "").strip().lower()
    if name in no_arm:
        return False
    types = {str(k).lower(): v for k, v in (robot_types or {}).items()}
    robot_type = str(types.get(name) or "").strip().lower()
    return robot_type not in NO_ARM_ROBOT_TYPES


def robots_with_arms(robot_ids, robot_types=None):
    """The robots of ``robot_ids`` that have an arm, in order."""
    return [r for r in robot_ids if robot_has_arm(r, robot_types)]


def _normalise_label(label):
    return " ".join(re.split(r"[\s_\-]+", str(label or "").strip().lower())).strip()


def heavy_classes():
    """Object classes that are dragged, lowercase (OMNIPLANNER_HEAVY_CLASSES overrides)."""
    override = _env_list(HEAVY_CLASSES_ENV)
    classes = DEFAULT_HEAVY_CLASSES if override is None else override
    return tuple(_normalise_label(c) for c in classes if _normalise_label(c))


def label_is_heavy(label, classes=None):
    """``label`` names a heavy class: equal to one, or ending in one ("red bag").

    A trailing plural "s" is tolerated ("sandbags").
    """
    words = _normalise_label(label).split()
    if not words:
        return False
    candidates = [words]
    if words[-1].endswith("s") and len(words[-1]) > 1:
        candidates.append(words[:-1] + [words[-1][:-1]])
    for cls in classes if classes is not None else heavy_classes():
        cw = cls.split()
        for w in candidates:
            if len(w) >= len(cw) and w[-len(cw) :] == cw:
                return True
    return False


def node_labels(G, node):
    """The names an object node goes by: attributes.name and its labelspace category."""
    labels = []
    name = getattr(getattr(node, "attributes", None), "name", "")
    if name:
        labels.append(str(name))
    try:
        labelspace = G.get_labelspace(node.layer.layer, node.layer.partition)
        category = labelspace.get_node_category(node) if labelspace else ""
        if category:
            labels.append(str(category))
    except Exception:
        pass
    return labels


def _find_node(G, symbol):
    from dsg_pddl.pddl_utils import pddl_symbol_to_node_symbol

    try:
        ns = pddl_symbol_to_node_symbol(G, str(symbol).lower())
    except (ValueError, IndexError):
        return None
    if G is None or not G.has_node(ns):
        return None
    return G.get_node(ns)


def explicit_heavy_symbols(extra=None):
    """Symbols declared heavy outright: OMNIPLANNER_HEAVY_SYMBOLS plus ``extra``."""
    out = set(_env_list(HEAVY_SYMBOLS_ENV) or [])
    out |= {str(s).strip().lower() for s in (extra or []) if str(s).strip()}
    return out


def is_heavy(G, symbol, extra=None, classes=None):
    """Whether the object ``symbol`` (``o12``) must be dragged rather than carried."""
    sym = str(symbol).strip().lower()
    if sym in explicit_heavy_symbols(extra):
        return True
    if not sym.startswith("o"):
        return False
    node = _find_node(G, sym)
    if node is None:
        return False
    classes = heavy_classes() if classes is None else classes
    return any(label_is_heavy(lbl, classes) for lbl in node_labels(G, node))


def heavy_objects(G, symbols, extra=None):
    """The heavy objects among ``symbols`` (lowercase PDDL names), sorted."""
    classes = heavy_classes()
    explicit = explicit_heavy_symbols(extra)
    out = set()
    for s in symbols:
        sym = str(s).strip().lower()
        if sym in explicit or is_heavy(G, sym, classes=classes):
            out.add(sym)
    return sorted(out)
