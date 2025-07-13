from typing import Union
from PIL import Image
import numpy as np

from .goal_manager import Goal, GoalManager
from .memory import Memory
from .perception import Perception
from .planner import Planner
from .state_tracker import StateTracker
from .actions import Action

class RobotAgent:
    """
    API pública: RobotAgent.step(image) -> Action
    """
    def __init__(self, goal_text: str, vlm_provider: str = "openai"):
        # 1. Inference engine
        from .vlm_inference import VLMInference
        self.perception = Perception(
            VLMInference(goal=goal_text, provider=vlm_provider)
        )
        # 2. Componentes internos
        self.goal_manager = GoalManager(Goal(goal_text))
        self.memory = Memory(size=10)
        self.planner = Planner()
        self.state_tracker = StateTracker()

    def step(self, img: Union[str, Image.Image, np.ndarray]) -> Action:
        obs = self.perception.perceive(img)
        current_goal = self.goal_manager.current
        self.goal_manager.update_from_observation(obs)

        # Decidir
        action = self.planner.decide(obs)
        self.memory.add(obs, action)
        self.state_tracker.update(action, obs)
        return action

    # Métodos auxiliares para serializar memoria, logs, etc.
