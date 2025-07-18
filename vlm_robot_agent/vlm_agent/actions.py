# ---------------------------------------------------------------------------
# vlm_robot_agent/vlm_agent/actions.py
# ---------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from .action_types import ActionKind, NavigationDirection, InteractionType

@dataclass
class Action:
    kind: ActionKind
    params: Dict[str, Any] = field(default_factory=dict)
    # Ejemplos de params:
    #   NAVIGATION: {"direction": NavigationDirection, "angle": float, "distance": float}
    #   INTERACTION: {"interaction_type": InteractionType, "target": str}

    def is_done(self, observation: Dict[str, Any]) -> bool:
        """
        Lógica rápida para saber si la acción terminó, basada en la observación más reciente
        (p. ej. si estoy hablando y la puerta se despejó => terminada).
        """
        # ★ Implementar según tus sensores o outputs del VLM
        return observation.get("status") == "DONE"
