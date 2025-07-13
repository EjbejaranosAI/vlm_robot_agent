from enum import Enum, auto

class AgentState(Enum):
    NAVIGATING = auto()
    INTERACTING = auto()
    TALKING = auto()
    WAITING_REPLY = auto()
    FINISHED = auto()

class StateTracker:
    def __init__(self) -> None:
        self.state = AgentState.NAVIGATING

    def update(self, last_action, last_obs) -> None:
        if last_obs.get("goal_observed") and self.state != AgentState.FINISHED:
            if last_action.kind == "interaction":
                # Si la persona se ha movido ya, volvemos a navegar
                if "person" not in last_obs["obstacles"]:
                    self.state = AgentState.NAVIGATING
            elif last_action.kind == "navigation" and last_obs.get("room_entered"):
                self.state = AgentState.FINISHED
        # Otros cambios de estado según reglas …
