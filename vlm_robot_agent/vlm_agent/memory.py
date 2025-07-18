# ---------------------------------------------------------------------------
# vlm_robot_agent/vlm_agent/memory.py
# ---------------------------------------------------------------------------

from collections import deque
from typing import Deque, Dict, Any
from .actions import Action

class Memory:
    """
    Almacena (máx N) pares (observación, acción) para dar contexto al VLM.
    """
    def __init__(self, size: int = 10) -> None:
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=size)

    def add(self, observation: Dict[str, Any], action: Action) -> None:
        self.buffer.append({"obs": observation, "action": action})

    def summary(self) -> str:
        # Genera resumen tipo bullet list para añadir al prompt
        lines = []
        for idx, item in enumerate(self.buffer):
            act = item["action"]
            lines.append(f"{idx+1}. {act.kind} → {act.params}")
        return "\n".join(lines)
