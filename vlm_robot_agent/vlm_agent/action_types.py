# ---------------------------------------------------------------------------
# vlm_robot_agent/vlm_agent/action_types.py
# ---------------------------------------------------------------------------

from enum import Enum

class NavigationDirection(str, Enum):
    FORWARD = "forward"
    FORWARD_LEFT = "forward_left"
    FORWARD_RIGHT = "forward_right"
    LEFT = "left"
    RIGHT = "right"

class ActionKind(str, Enum):
    NAVIGATION = "navigation"
    INTERACTION = "interaction"

class InteractionType(str, Enum):
    TALK = "talk"
    GESTURE = "gesture"
