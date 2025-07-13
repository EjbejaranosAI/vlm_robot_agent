# ---------------------------------------------------------------------------
# perception.py
# ---------------------------------------------------------------------------
from __future__ import annotations

from typing import Dict, Any, Union
from pathlib import Path

from PIL import Image
import numpy as np

try:
    from ..vlm_inference.inference import VLMInference  # type: ignore
except ImportError as exc:
    # Helpful error if package layout is wrong.
    raise ImportError(
        "Cannot import VLMInference. Expected it under 'vlm_robot_agent/vlm_inference/'.\n"
        "Verify your folder structure matches the one documented in the module header."
    ) from exc

__all__ = ["Perception", "Observation"]

# ---------------------------------------------------------------------------
# Typed alias for an observation returned by Perception.perceive()
Observation = Dict[str, Any]


class Perception:
    """High‑level perception interface.

    Parameters
    ----------
    goal_text : str
        The current high‑level goal – is baked into the prompt of each
        *VLMInference* instance.
    provider : str, default "openai"
        Inference backend; passed straight to :class:`VLMInference`.
    history_size : int, default 10
        Length of the circular buffer inside each :class:`VLMInference`.
    """

    def __init__(self, *, goal_text: str, provider: str = "openai", history_size: int = 10):
        self._navigation_engine = VLMInference(
            goal=goal_text,
            provider=provider,
            history_size=history_size,
            prompt_name="navigation",
        )
        self._interaction_engine = VLMInference(
            goal=goal_text,
            provider=provider,
            history_size=history_size,
            prompt_name="interaction",
        )

    # ------------------------------------------------------------------
    def perceive(
        self,
        img: Union[str, Path, Image.Image, np.ndarray],
        *,
        mode: str = "navigation",  # "navigation" | "interaction"
    ) -> Observation:
        """Run VLM inference and standardise its output.

        *No* domain logic lives here – we simply normalise the JSON so downstream
        modules need not worry about slight variations in the model output.
        """
        engine = self._navigation_engine if mode == "navigation" else self._interaction_engine
        result = engine.infer(img)

        # Flatten / coerce the TypedDict coming from VLMInference into a simple dict.
        observation: Observation = {
            "status": getattr(result["status"], "value", result["status"]),
            "description": result.get("description", ""),
            "obstacles": result.get("obstacles", []),
            "current_environment_type": result.get("current_environment_type", "UNKNOWN_ENV"),
            "suggested_actions": result.get("actions", []),
            # quick boolean for convenience ↓
            "goal_observed": any(
                str(a.get("Goal_observed", "False")).lower() == "true" for a in result.get("actions", [])
            ),
        }
        return observation
