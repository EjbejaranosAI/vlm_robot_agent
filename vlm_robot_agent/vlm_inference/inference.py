"""# vlm_robot_agent/vlm_inference/inference.py
================================
Vision‑Language‑Model (VLM) Inference engine specialised for indoor robot
navigation & interaction.  Works either against the OpenAI Vision model
(“GPT‑4o” family) via API or against a pluggable local model (future).

The class **VLMInference** receives a single image (or path / ndarray), a
high‑level *goal* string (e.g. "Enter office 42"), and optional action
history.  It returns a *structured JSON‐like dict* with:

    • actions   – list[Action] (Navigation or Interaction)
    • description – short scene summary
    • obstacles – list[str]
    • current_environment_type – enum str
    • status – enum Status
    • error – diagnostic field (empty if OK)

The file is 100 % self‑contained; no external project imports besides
`openai`, `Pillow`, `numpy`, and standard lib.  Can be used stand‑alone as:

>>> from vlm_inference.inference import VLMInference
>>> engine = VLMInference(goal="Find the bathroom")
>>> result = engine.infer("/path/to/image.jpg")
>>> print(result["actions"][0])
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, TypedDict, Union
from importlib import resources

import numpy as np
import yaml  # optional: if you want YAML config files
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw

__all__ = [
    "Status",
    "ActionParameters",
    "Action",
    "InferenceResult",
    "VLMInference",
]


PROMPT_FILE_JSON = "navigation_prompts.json"


###############################################################################
# Logging & utility helpers                                                   #
###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vlm_inference")


def annotate_tercios(img: Image.Image, color: tuple[int, int, int, int] = (255, 0, 255, 80)) -> Image.Image:
    """Overlay two vertical lines that split the image in thirds (Left/Center/Right).
    Returns a *new* RGBA image (does not modify original)."""
    w, h = img.size
    x1, x2 = w / 3, 2 * w / 3

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.line([(x1, 0), (x1, h)], fill=color, width=5)
    draw.line([(x2, 0), (x2, h)], fill=color, width=5)

    return Image.alpha_composite(img.convert("RGBA"), overlay)


def pil_to_data_url(img: Image.Image, fmt: str = "JPEG") -> str:
    """Convert a PIL Image to a data‑URL (base64).  Auto‑converts RGBA→RGB for JPEG."""
    if fmt.upper() == "JPEG" and img.mode == "RGBA":
        img = img.convert("RGB")
    buff = io.BytesIO()
    img.save(buff, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buff.getvalue()).decode()}"

###############################################################################
# TypedDicts & Enums                                                          #
###############################################################################


class Status(str, Enum):
    OK = "OK"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"
    NEED_HELP = "NEED_HELP"
    FINISHED = "FINISHED"


class ActionParameters(TypedDict, total=False):
    # Navigation
    direction: str
    angle: float
    distance: float

    # Interaction specific
    interaction_type: str  # "talk|gesture|wait"
    utterance: str
    gesture: str
    target: str


class Action(TypedDict):
    type: str  # "Navigation" | "Interaction"
    parameters: ActionParameters
    Goal_observed: str
    where_goal: str
    obstacle_avoidance_strategy: str


class InferenceResult(TypedDict):
    actions: List[Action]
    description: str
    obstacles: List[str]
    current_environment_type: str
    status: Status
    error: str


class HistoryItem(TypedDict):
    action: Action
    description: str
    current_environment_type: str
    status: Status


###############################################################################
# Settings & Exceptions                                                       #
###############################################################################


class VLMSettings:
    """Simple .env‑based settings (only OpenAI‑key for now)."""

    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")


class VLMInferenceError(Exception):
    """Generic wrapper so caller can catch VLM‑specific errors."""


###############################################################################
# Main class                                                                  #
###############################################################################


class VLMInference:
    """High‑level wrapper that embeds: prompt → image → call LLM → parse JSON."""

    def __init__(
        self,
        goal: str,
        provider: str = "openai",
        prompt_path: str | Path | None = None,
        history_size: int = 6,
        settings: VLMSettings | None = None,
    ) -> None:
        self.goal = goal
        self.provider = provider
        self.settings = settings or VLMSettings()
        self.action_history: deque[HistoryItem] = deque(maxlen=history_size)

        if provider != "openai":
            raise NotImplementedError("Only provider='openai' implemented for now")
        if not self.settings.api_key:
            raise VLMInferenceError("OPENAI_API_KEY env‑var missing")

        self.client = OpenAI(api_key=self.settings.api_key)
        self.base_prompt_template = self._load_prompt(prompt_path)
        logger.info("VLMInference ready (goal=%s)", goal)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def infer(self, image: Union[str, Path, Image.Image, np.ndarray]) -> InferenceResult:
        """Run full cycle and return parsed dict.  Handles any exception -> status ERROR."""
        try:
            data_url = self._prepare_image(image)
            prompt = self._format_prompt()
            raw = self._call_llm(data_url, prompt)
            parsed = self._parse_response(raw)
            self._maybe_store_history(parsed)
            return parsed
        except Exception as exc:
            logger.exception("Inference failed: %s", exc)
            return {
                "actions": [],
                "description": f"Error: {exc}",
                "obstacles": [],
                "current_environment_type": "UNKNOWN",
                "status": Status.ERROR,
                "error": str(exc),
            }

    # ---------------------------------------------------------------------
    # Prompt management
    # ---------------------------------------------------------------------
    def _load_prompt(self, path: str | Path | None, prompt_key: str = "default") -> str:
        """Return the prompt string either from an explicit file or from the packaged JSON."""
        # 1️⃣  explicit text file wins
        if path is not None:
            return Path(path).read_text(encoding="utf-8")

        # 2️⃣  fallback to packaged JSON
        
        with resources.files("vlm_robot_agent.prompts").joinpath(PROMPT_FILE_JSON).open("r", encoding="utf-8") as f:
            print(f)
            print("-------------------")
            data = json.load(f)
        try:
            return data[prompt_key]["system"]
        except KeyError as exc:
            raise VLMInferenceError(f"Prompt key '{prompt_key}' not found in {PROMPT_FILE_JSON}") from exc


    def _format_prompt(self) -> str:
        history_lines: list[str] = []
        for i, h in enumerate(self.action_history):
            a = h["action"]
            history_lines.append(
                f"{i+1}. {a['type']} params={a['parameters']} status={h['status'].value} desc={h['description']}"
            )
        history = "\n".join(history_lines) if history_lines else "(none)"
        return self.base_prompt_template.format(goal=self.goal, action_history=history)

    # ---------------------------------------------------------------------
    # Image helpers
    # ---------------------------------------------------------------------

    def _prepare_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> str:
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        annotated = annotate_tercios(img)
        return pil_to_data_url(annotated)

    # ---------------------------------------------------------------------
    # OpenAI call
    # ---------------------------------------------------------------------

    def _call_llm(self, data_url: str, prompt: str) -> str:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        txt = resp.choices[0].message.content
        logger.debug("LLM latency %.2fs", time.time() - t0)
        return txt

    # ---------------------------------------------------------------------
    # Parse & history
    # ---------------------------------------------------------------------

    def _parse_response(self, raw: str) -> InferenceResult:
        clean = raw.strip()

        # --- elimina bloque ``` … ``` si existe
        if clean.startswith("```"):
            clean = clean.split("```", 2)[1].strip()

        # --- elimina prefijo opcional “json” o similar
        if clean.lower().startswith("json"):
            idx = clean.find("{")
            if idx != -1:
                clean = clean[idx:]          # nos quedamos desde “{” en adelante

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            raise VLMInferenceError(
                f"Invalid JSON from model: {e}; text={clean[:200]}"
            ) from e
        
        # status
        status = Status(data.get("status", "ERROR"))

        # actions normalisation
        actions: list[Action] = []
        for a in data.get("actions", []):
            actions.append(
                Action(
                    type=a.get("type", "Navigation"),
                    parameters=a.get("parameters", {}),
                    Goal_observed=a.get("Goal_observed", "False"),
                    where_goal=a.get("where_goal", "FALSE"),
                    obstacle_avoidance_strategy=a.get("obstacle_avoidance_strategy", ""),
                )
            )

        return {
            "actions": actions,
            "description": data.get("description", ""),
            "obstacles": data.get("obstacles", []),
            "current_environment_type": data.get("current_environment_type", "UNKNOWN"),
            "status": status,
            "error": "",
        }

    def _maybe_store_history(self, result: InferenceResult) -> None:
        if result["actions"] and result["status"] != Status.ERROR:
            self.action_history.append(
                HistoryItem(
                    action=result["actions"][0],
                    description=result["description"],
                    current_environment_type=result["current_environment_type"],
                    status=result["status"],
                )
            )


#/home/edison/Desktop/PhD/vlm_robot_agent/img/1_center.jpg
# -------------------------------------------------------------------------
# Stand-alone demo
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, glob, pprint, sys
    from pathlib import Path

    # ---------- helper defaults ----------
    def _default_image() -> str:
        # Coge la primera JPG/PNG encontrada dentro de ./img (si existe)
        here = Path(__file__).parent.parent.parent  # …/vlm_robot_agent
        imgs = glob.glob(str(here / "img" / "*.[jp][pn]g"))
        return imgs[0] if imgs else ""

    # ---------- CLI ----------
    parser = argparse.ArgumentParser(
        description="Quick test for VLMInference (OpenAI Vision)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--image",
        metavar="PATH",
        default=_default_image(),
        help="path to RGB image file",
    )
    parser.add_argument(
        "-g", "--goal",
        default="Entrar en la oficina 12",
        help="navigation goal text",
    )
    args = parser.parse_args()

    # ---------- sanity checks ----------
    if not args.image:
        sys.exit("❌  No image found/provided. Use --image PATH.")
    if not Path(args.image).exists():
        sys.exit(f"❌  Image not found: {args.image}")

    # ---------- run inference ----------
    engine = VLMInference(goal=args.goal)
    result = engine.infer(args.image)

    pprint.pprint(result, compact=True, width=120)

