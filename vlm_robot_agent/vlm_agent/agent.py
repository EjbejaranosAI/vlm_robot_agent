# ---------------------------------------------------------------------------
# # vlm_robot_agent/vlm_agent/agent.py
# ---------------------------------------------------------------------------
# vlm_robot_agent/vlm_agent/agent.py

from __future__ import annotations

from enum import Enum, auto
from typing import Union
from pathlib import Path

from PIL import Image
import numpy as np

from .goal_manager import Goal, GoalManager
from .memory import Memory
from .planner import Planner
from .state_tracker import StateTracker, AgentState
from .actions import Action
from .perception import Perception, Observation

try:
    # ConversationManager is optional – import lazily.
    from .conversation import ConversationManager  # type: ignore
except ImportError:
    ConversationManager = None  # type: ignore

__all__ = ["RobotAgent"]


class RobotAgent:
    """Public-facing façade of the whole agent stack.

    Usage
    -----
    >>> agent = RobotAgent(goal_text="Entrar en la oficina 12")
    >>> while True:
    ...     img = camera.read()
    ...     action = agent.step(img)
    ...     robot.execute(action)
    ...     if agent.finished:
    ...         break
    """

    def __init__(
        self,
        *,
        goal_text: str,
        provider: str = "openai",
        history_size: int = 10,
    ) -> None:
        # 1) Perception engines
        self.perception = Perception(
            goal_text=goal_text,
            provider=provider,
            history_size=history_size,
        )

        # 2) Cognition / memory / planning
        self.goal_manager = GoalManager(Goal(goal_text))
        self.memory = Memory(size=history_size)
        self.planner = Planner()
        self.state_tracker = StateTracker()

        # 3) Optional conversation manager
        self.conversation = ConversationManager(goal_text) if ConversationManager else None

        # 4) Decompose high-level goal into sub-goals
        if hasattr(self.planner, "decompose"):
            subgoals = self.planner.decompose(goal_text)
            for sg in subgoals:
                self.goal_manager.push_subgoal(Goal(sg))

        # 5) Print initial plan
        print("┌ Plan inicial de sub-goals ─────────────────────────")
        for idx, g in enumerate(self.goal_manager.goal_stack, start=1):
            print(f"│ {idx}. {g.description}")
        print("└──────────────────────────────────────────────────")

    @property
    def finished(self) -> bool:
        """Whether the agent has completed its top-level mission."""
        return self.state_tracker.state == AgentState.FINISHED

    def step(self, img: Union[str, Path, Image.Image, np.ndarray]) -> Action:
        """One control tick:
        1. Perceive with the right mode.
        2. Update goal_manager & state_tracker.
        3. Decide next action via planner.
        4. If INTERACTION, run ConversationManager.
        5. Log to memory.
        """

        # 1) Run perception
        mode = self._current_mode()
        obs: Observation = self.perception.perceive(img, mode=mode)

        # 2) Update goals + state
        self.goal_manager.update_from_observation(obs)
        self.goal_manager.pop_finished()
        self.state_tracker.update_last_observation(obs)

        # 3) Plan next action
        action = self.planner.decide(obs)

        # 4) If interaction, trigger conversation turn
        if action.kind.name.lower() == "interaction" and self.conversation:
            utterance = self.conversation.robot_turn()
            action.params["utterance"] = utterance

        # 5) Record into memory
        self.memory.add(obs, action)

        return action

    def _current_mode(self) -> str:
        """Choose 'navigation' vs 'interaction' based on the agent's state."""
        if self.state_tracker.state in {
            AgentState.INTERACTING,
            AgentState.TALKING,
            AgentState.WAITING_REPLY,
        }:
            return "interaction"
        return "navigation"




#TODO: Este script debe llamar las distintas partes o sublibrerias, tener el goal como input, asi como una imagen, crear subgoals de lo que tiene que hacer, imprimir el plan con los subgoals, devolver las acciones de navegacion e interacion. En las acciones de navegacion se devolvera movimientos forward, left, right, con distancia. EN el caso de interaction se puede comenzar una conversacion y se usan los tts y stt para hablar con el humano, con lo que tenemos que entender como devolver esto en la libreria, ya que luego lo usare en ROS. 
#TODO: Cada vez que acabe la interacion se debe analizar el entorno, con una imagen para verificar si el camino esta libre y cambiar a acciones de navegacion. 
#TODO: Como esta libreria con todas sus sublibrerias puede ser usada de manera que sea facil en ROS2. 

