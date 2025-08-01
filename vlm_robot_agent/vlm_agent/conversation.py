"""
    vlm_robot_agent/vlm_agent/conversation.py
----------------------------------------
• El LLM crea la charla y decide cuándo insertar
  #HUMANO_DESPEJO_PASO | #HUMANO_RECHAZO | #HUMANO_SIN_RESPUESTA.
• El script:
    – cuenta negativas / silencio,
    – ordena al LLM despedirse con la etiqueta adecuada,
    – termina al detectar la etiqueta.
• NUEVO: imprime la clasificación de cada respuesta y el resultado final.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import os, json, time, re, unicodedata
from importlib import resources

from openai import OpenAI
from vlm_robot_agent.vlm_agent.io import speech_io

PROMPT_FILE = "conversation_prompts.json"
TAG_RE = re.compile(r"#HUMANO_(?:DESPEJO_PASO|RECHAZO|SIN_RESPUESTA)", re.I)

POS_TAG = "Clear"
NEG_TAG = "Not_Clear"
SIL_TAG = "Silence"


# ---------- utils --------------------------------------------------
def _load_prompt(key: str) -> Dict[str, str]:
    with resources.files("vlm_robot_agent.prompts").joinpath(PROMPT_FILE).open(
        "r", encoding="utf-8"
    ) as f:
        data = json.load(f)
    e = data.get(key) or {}
    return {"system": e.get("system", ""), "examples": e.get("examples", [])}


def _clean(t: str) -> str:
    return t.replace("¿", "").replace("¡", "")


@dataclass
class Turn:
    role: str
    text: str


# ---------- manager ------------------------------------------------
class ConversationManager:
    def __init__(
        self,
        goal: str,
        *,
        prompt_key="default",
        model="gpt-4o-mini",
        max_history=40,
        silence_limit=10,
        max_negative=5,
        max_elapsed_neg=40,
        openai_api_key: str | None = None,
    ):
        self.goal = goal
        self.model = model
        self.max_history = max_history
        self.silence_limit = silence_limit
        self.max_negative = max_negative
        self.max_elapsed_neg = max_elapsed_neg

        self.silence_elapsed = 0
        self.negatives = 0
        self.neg_start: float | None = None
        self.greeted = False
        self.final_result: str | None = None  # <-- nuevo

        prm = _load_prompt(prompt_key)
        self._system_tpl = prm["system"]
        self._examples = prm["examples"]

        self.history: List[Turn] = []
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))

    # ----- prompt builders --------------
    def _system_prompt(self) -> str:
        return self._system_tpl.format(goal=self.goal)

    def _msgs(self) -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": self._system_prompt()}]
        msgs.extend(self._examples)
        msgs.extend(
            {"role": "assistant" if t.role == "robot" else "user", "content": t.text}
            for t in self.history[-self.max_history :]
        )
        return msgs

    # ----- IO ---------------------------
    def _speak(self, txt: str):
        speech_io.speak(_clean(txt))
        print(f"[Robot] {txt}")

    # ----- greeting ---------------------
    def greet(self):
        if not self.greeted:
            self._speak(
                f"Hola, soy Tiago, el robot asistente del edificio. Mi misión es {self.goal}."
            )
            self.greeted = True

    # ----- generic LLM call -------------
    def _ask_llm(self, messages: List[Dict[str, str]], max_tokens=60) -> str:
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )
        return r.choices[0].message.content.strip()

    # ----- main robot turn --------------
    def robot_turn(self):
        txt = self._ask_llm(self._msgs())
        self.history.append(Turn("robot", txt))
        self._speak(txt)
        return txt

    # ----- listen -----------------------
    def listen(self, secs=5) -> str | None:
        txt = speech_io.listen(seconds=secs) or ""
        if txt:
            txt = txt.strip()
            self.history.append(Turn("human", txt))
            print(f"[Humano] {txt}")
            return txt
        print("[Humano] (silencio)")
        return None

    # ----- classify ---------------------
    def _classify(self, human: str) -> str:
        cls = self._ask_llm(
            [
                {
                    "role": "system",
                    "content": (
                        "Responde solo positive, negative o neutral según la persona "
                        "acepte moverse (positive) o se niegue (negative)."
                    ),
                },
                {"role": "user", "content": human},
            ],
            max_tokens=1,
        )
        return cls.lower().strip()

    # ----- despedida via LLM ------------
    def _farewell_with_tag(self, tag: str):
        txt = self._ask_llm(
            [
                {
                    "role": "system",
                    "content": "Di una frase corta de despedida en tono cordial "
                    f"y termina EXACTAMENTE con {tag}.",
                }
            ],
            max_tokens=30,
        )
        if tag.lower() not in txt.lower():
            txt += f" {tag}"
        self._speak(txt)

    # ----- main loop --------------------
    def interactive_turn(self, listen_secs=5) -> bool:

        if not self.greeted:
            self.greet()

        robot_reply = self.robot_turn()

        if TAG_RE.search(robot_reply):
            self.final_result = TAG_RE.search(robot_reply).group().upper()
            print(f"--> Resultado final: {self.final_result}")
            return False

        human = self.listen(listen_secs)

        if human:
            cls = self._classify(human)
            print(f"[Clasificación LLM] → {cls}")  # <-- nuevo print
            if cls.startswith("pos"):
                self.final_result = POS_TAG
                self._farewell_with_tag(POS_TAG)
                print(f"--> Resultado final: {self.final_result}")
                return False

            if cls.startswith("neg"):
                self.negatives += 1
                if self.negatives == 1:
                    self.neg_start = time.time()
                if self.negatives >= self.max_negative:
                    self.final_result = NEG_TAG
                    self._farewell_with_tag(NEG_TAG)
                    print(f"--> Resultado final: {self.final_result}")
                    return False

            self.silence_elapsed = 0
            return True

        # silencio
        self.silence_elapsed += listen_secs
        if self.silence_elapsed >= self.silence_limit:
            self.final_result = SIL_TAG
            self._farewell_with_tag(SIL_TAG)
            print(f"--> Resultado final: {self.final_result}")
            return False

        if self.neg_start and time.time() - self.neg_start >= self.max_elapsed_neg:
            self.final_result = NEG_TAG
            self._farewell_with_tag(NEG_TAG)
            print(f"--> Resultado final: {self.final_result}")
            return False

        return True

    # ----- dump -------------------------
    def dump(self) -> str:
        return "\n".join(f"{t.role}: {t.text}" for t in self.history)


# --------------- CLI -------------------
if __name__ == "__main__":
    cm = ConversationManager(goal="entrar en la oficina doce", prompt_key="default", silence_limit=5)

    while cm.interactive_turn(listen_secs=5):
        pass

    print("\n=== Historial completo ===")
    print(cm.dump())
    if cm.final_result:
        print(f"\n### Conversación concluida con etiqueta: {cm.final_result}")


