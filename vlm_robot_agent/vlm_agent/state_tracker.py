# ---------------------------------------------------------------------------
# vlm_robot_agent/vlm_agent/state_tracker.py
# ---------------------------------------------------------------------------

from enum import Enum, auto

class AgentState(Enum):
    IDLE = auto()
    PLANNING = auto()
    MOVING = auto()
    NAVIGATING = auto()     # Estado para navegación activa
    INTERACTING = auto()
    TALKING = auto()        # Estado para interacción hablada
    WAITING_REPLY = auto()  # Esperando respuesta humana
    FINISHED = auto()       # Estado finalizado

class StateTracker:
    def __init__(self) -> None:
        self.state = AgentState.NAVIGATING
        self.last_observation = None  # Guarda la última observación para referencia interna

    def update_last_observation(self, observation: dict) -> None:
        """
        Actualiza el estado basado en la última observación recibida.
        """
        self.last_observation = observation
        # Aquí puedes decidir cambiar estado o mantenerlo
        # Ejemplo simple:
        if observation.get("goal_observed") and self.state != AgentState.FINISHED:
            # Si estoy interactuando y ya no hay persona bloqueando, cambio a navegar
            if self.state in {AgentState.INTERACTING, AgentState.TALKING, AgentState.WAITING_REPLY}:
                if "person" not in observation.get("obstacles", []):
                    self.state = AgentState.NAVIGATING
            # Si ya entré en la sala marco estado finalizado
            if observation.get("room_entered"):
                self.state = AgentState.FINISHED

    def update(self, last_action, last_obs) -> None:
        """
        Método adicional para actualizar estado usando última acción y observación,
        puede ser llamado desde fuera si se prefiere.
        """
        self.update_last_observation(last_obs)
