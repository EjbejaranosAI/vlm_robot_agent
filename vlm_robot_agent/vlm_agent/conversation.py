# vlm_agent/conversation.py
from dataclasses import dataclass, field
from typing import List, Dict, Any
import asyncio
from openai import OpenAI  # o tu proveedor LLM

@dataclass
class Utterance:
    role: str            # "robot" | "human"
    text: str

class ConversationManager:
    """
    Lleva el hilo del diálogo y decide la próxima respuesta.
    Se usa sólo cuando el Planner devolvió una acción Interaction(talk|gesture).
    """
    def __init__(self, goal: str, model: str = "gpt-4o-mini") -> None:
        self.history: List[Utterance] = []
        self.client = OpenAI()
        self.model = model
        self.goal = goal

    def _build_prompt(self) -> List[Dict[str, str]]:
        system_msg = (
            "You are a polite service robot trying to clear the path to "
            f"'{self.goal}'. Keep sentences short and respectful."
        )
        messages = [{"role": "system", "content": system_msg}]
        for u in self.history[-10:]:               # máx 10 turnos
            messages.append({"role": u.role, "content": u.text})
        return messages

    async def ask(self) -> str:
        """Genera la primera frase (o la siguiente) para persuadir al humano."""
        messages = self._build_prompt()
        resp = self.client.chat.completions.create(model=self.model,
                                                   messages=messages,
                                                   max_tokens=60)
        text = resp.choices[0].message.content.strip()
        self.history.append(Utterance("robot", text))
        return text

    def reply(self, human_text: str) -> None:
        """Agrega el turno humano al historial. Se llama desde STT."""
        self.history.append(Utterance("human", human_text))

    def save_log(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for u in self.history:
                f.write(f"{u.role}: {u.text}\n")
