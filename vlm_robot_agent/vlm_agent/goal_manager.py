# ---------------------------------------------------------------------------
# vlm_robot_agent/vlm_agent/goal_manager.py
# ---------------------------------------------------------------------------


from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

@dataclass
class Goal:
    description: str                 # “Entrar en la oficina X”
    location_hint: Optional[str] = ""  # “coordenadas puerta oficina X”
    finished: bool = False

class GoalManager:
    """
    Divide un goal grande en micro-metas y las actualiza.
    """
    def __init__(self, initial_goal: Goal) -> None:
        self.goal_stack: Deque[Goal] = deque([initial_goal])

    @property
    def current(self) -> Goal:
        return self.goal_stack[-1]

    def update_from_observation(self, obs: dict) -> None:
        """
        Analiza la observación y marca metas como finalizadas o crea nuevas.
        """
        if "room_entered" in obs and obs["room_entered"]:
            self.current.finished = True

    def push_subgoal(self, goal: Goal) -> None:
        self.goal_stack.append(goal)

    def pop_finished(self) -> None:
        while self.goal_stack and self.goal_stack[-1].finished:
            self.goal_stack.pop()


