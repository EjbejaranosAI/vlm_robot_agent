from typing import Dict, Any
from .actions import Action
from .action_types import ActionKind, NavigationDirection, InteractionType

class Planner:
    """
    Traducir percepción + meta → próxima acción.
    Mantén aquí heurística o llama a otro LLM si prefieres “chain-of-thought”.
    """
    def decide(self, observation: Dict[str, Any]) -> Action:
        status = observation["status"]
        obstacles = observation["obstacles"]

        # 1. Si bloqueado por persona: INTERACTION
        if "person" in obstacles:
            return Action(
                kind=ActionKind.INTERACTION,
                params={"interaction_type": InteractionType.TALK, "target": "person"}
            )

        # 2. Si despejado: NAVIGATE hacia adelante
        return Action(
            kind=ActionKind.NAVIGATION,
            params={"direction": NavigationDirection.FORWARD, "angle": 0.0, "distance": 0.5}
        )
