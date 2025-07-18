# ---------------------------------------------------------------------------
# vlm_robot_agent/vlm_agent/planner.py
# ---------------------------------------------------------------------------
# vlm_robot_agent/vlm_agent/planner.py

from typing import Dict, Any, List
from .actions import Action
from .action_types import ActionKind, NavigationDirection, InteractionType

class Planner:
    """
    Traducir percepción + meta → próxima acción y descomponer meta en sub-goals.
    """

    def decompose(self, goal_text: str) -> List[str]:
        """
        Divide un goal de alto nivel en una lista de sub-goals.
        Puedes reemplazar esta implementación por una llamada a un LLM
        para generar dinámicamente los pasos intermedios.
        """
        lower = goal_text.lower()
        # Ejemplo estático para metas que contengan "oficina"
        if "oficina" in lower:
            return [
                "Ir al pasillo principal",
                "Orientarse hacia la puerta de la oficina",
                "Avanzar hasta la entrada de la oficina",
                "Entrar en la oficina"
            ]
        # Por defecto, un único paso igual al goal completo
        return [goal_text]

    def decide(self, observation: Dict[str, Any]) -> Action:
        """
        Basado en la observación actual, devuelve la siguiente acción.
        - Si detecta una persona bloqueando, genera una interacción.
        - En caso contrario, genera un comando de navegación hacia adelante.
        """
        status = observation.get("status", "")
        obstacles = observation.get("obstacles", [])

        # 1) Si hay persona en el camino → interacción hablada
        if "person" in obstacles:
            return Action(
                kind=ActionKind.INTERACTION,
                params={
                    "interaction_type": InteractionType.TALK,
                    "target": "person"
                }
            )

        # 2) Si no hay obstáculo → navegar hacia adelante
        return Action(
            kind=ActionKind.NAVIGATION,
            params={
                "direction": NavigationDirection.FORWARD,
                "angle": 0.0,
                "distance": 0.5
            }
        )




#TODO: # Este script debe tomar el goal y partirlo teniendo en cuenta las observaciones actuales que detecte del VLM, con lo que debe hacer una llamada al modelo, ver que ocurre, observar si tiene obstaculos y si es un humano, que sea el primer goal acercarse, interactuar con el humano, convencerlo , verificar que el path esta limpio, y pasar. En el caso de que alguno de los subgoals no se cumplan se debe recalcular los goal o el plar