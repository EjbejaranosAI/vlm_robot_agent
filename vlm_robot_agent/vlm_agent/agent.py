# ---------------------------------------------------------------------------
# agent.py
# ---------------------------------------------------------------------------
from __future__ import annotations

from enum import Enum, auto
from typing import Union

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
    """Public‑facing façade of the whole agent stack.

    Usage
    -----
    >>> agent = RobotAgent(goal_text="Entrar en la oficina 12")
    >>> while True:
    ...     img = camera.read()
    ...     action = agent.step(img)
    ...     robot.execute(action)
    ...     if agent.finished:
    ...         break
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        goal_text: str,
        provider: str = "openai",
        history_size: int = 10,
    ) -> None:
        # Perception (two prompts under the hood)
        self.perception = Perception(goal_text=goal_text, provider=provider, history_size=history_size)

        # Cognition / memory / planning
        self.goal_manager = GoalManager(Goal(goal_text))
        self.memory = Memory(size=history_size)
        self.planner = Planner()
        self.state_tracker = StateTracker()

        # Optional conversation manager (only if you created conversation.py)
        self.conversation = ConversationManager(goal_text) if ConversationManager else None

    # ------------------------------------------------------------------
    def step(self, img: Union[str, Path, Image.Image, np.ndarray]) -> Action:
        """One control tick.

        1. Run *Perception* in the correct mode.
        2. Update internal goal & FSM.
        3. Ask *Planner* for the next action.
        4. Record to *Memory* and return the chosen action.
        """
        mode = self._current_mode()
        obs = self.perception.perceive(img, mode=mode)

        # --------------- internal bookkeeping -----------------------
        self.goal_manager.update_from_observation(obs)
        self._maybe_switch_state(obs)

        # ------------------ decide next action ----------------------
        action = self._decide_action(obs)

        # ------------------- logging / memory -----------------------
        self.memory.add(obs, action)
        return action

    # ------------------------------------------------------------------
    @property
    def finished(self) -> bool:  # handy convenience accessor
        return self.state_tracker.state == AgentState.FINISHED

    # ========================= helpers ==============================
    def _current_mode(self) -> str:
        return "interaction" if self.state_tracker.state in {
            AgentState.INTERACTING,
            AgentState.TALKING,
            AgentState.WAITING_REPLY,
        } else "navigation"

    def _maybe_switch_state(self, obs: Observation) -> None:
        """Very thin wrapper that delegates to StateTracker."""
        self.state_tracker.update_last_observation(obs)

    def _decide_action(self, obs: Observation) -> Action:
        """Consult planner *and* conversation manager when needed."""
        action = self.planner.decide(obs)

        if action.kind.name.lower() == "interaction" and action.params.get("interaction_type") == "talk":
            if self.conversation is not None:
                utterance = self.conversation.ask_sync()
                action.params["utterance"] = utterance
        return action
