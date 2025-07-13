from typing import Dict, Any, Union
from PIL import Image
import numpy as np
from .vlm_inference import VLMInference  # importa tu clase existente

class Perception:
    def __init__(self, inference: VLMInference) -> None:
        self.inference = inference

    def perceive(self, img: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        Ejecuta el VLM y devuelve dict estandarizado:
        {
            "status": "OK" | "BLOCKED" | ...,
            "description": ...,
            "obstacles": [...],
            "goal_observed": bool,
            "actions": [...],          # acciones sugeridas por el VLM
        }
        """
        result = self.inference.infer(img)
        # Adaptar lo que devuelve VLMInference al formato que uses
        return {
            "status": result["status"].value,
            "description": result["description"],
            "obstacles": result["obstacles"],
            "goal_observed": any(a.get("Goal_observed") == "TRUE" for a in result["actions"]),
            "suggested_actions": result["actions"],
        }
