# vlm_agent/io/speech_io.py
import asyncio
from some_stt_lib import transcribe_async
from some_tts_lib import speak_async

async def speak(text: str) -> None:
    await speak_async(text)               # tu motor TTS

async def listen(timeout: float = 10.0) -> str | None:
    """
    Devuelve la transcripción o None si se agotó el tiempo.
    """
    try:
        return await asyncio.wait_for(transcribe_async(), timeout)
    except asyncio.TimeoutError:
        return None
