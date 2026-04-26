import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable

from google import genai
from google.genai import types


AudioCallback = Callable[[bytes], Awaitable[None]]
EventCallback = Callable[[dict], Awaitable[None]]


SYSTEM_PROMPT = (
    "Tu es Her, une présence vocale bienveillante, calme et précise. "
    "Détecte automatiquement si Arthur parle français ou anglais et réponds dans la même langue. "
    "Garde des réponses courtes, naturelles et chaleureuses. "
    "Si l'utilisateur t'interrompt, arrête-toi et écoute. "
    "Ne mentionne pas les détails techniques de ton fonctionnement."
)


class GeminiLiveBridge:
    def __init__(self, input_sample_rate: int = 16000) -> None:
        self.model = os.getenv("MODEL", "gemini-live-2.5-flash-native-audio")
        self.location = os.getenv("LOCATION", "us-central1")
        self.project_id = os.getenv("PROJECT_ID")
        self.input_sample_rate = input_sample_rate
        self.client = self._build_client()

    def _build_client(self) -> genai.Client:
        if os.getenv("GOOGLE_API_KEY"):
            return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        if self.project_id:
            return genai.Client(vertexai=True, project=self.project_id, location=self.location)
        raise RuntimeError("Set GOOGLE_API_KEY or PROJECT_ID before starting a Live session.")

    def _config(self) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=os.getenv("VOICE_NAME", "Puck"))
                )
            ),
            system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

    async def run(
        self,
        audio_queue: asyncio.Queue[bytes | None],
        text_queue: asyncio.Queue[str | None],
        on_audio: AudioCallback,
        on_event: EventCallback,
    ) -> None:
        async with self.client.aio.live.connect(model=self.model, config=self._config()) as session:
            tasks = [
                asyncio.create_task(self._send_audio(session, audio_queue)),
                asyncio.create_task(self._send_text(session, text_queue)),
                asyncio.create_task(self._receive(session, on_audio, on_event)),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in pending:
                task.cancel()
            for task in done:
                task.result()

    async def _send_audio(self, session, audio_queue: asyncio.Queue[bytes | None]) -> None:
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                try:
                    await session.send_realtime_input(audio_stream_end=True)
                except TypeError:
                    pass
                return
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={self.input_sample_rate}")
            )

    async def _send_text(self, session, text_queue: asyncio.Queue[str | None]) -> None:
        while True:
            text = await text_queue.get()
            if text is None:
                return
            await session.send(input=text, end_of_turn=True)

    async def _receive(self, session, on_audio: AudioCallback, on_event: EventCallback) -> None:
        async for response in session.receive():
            server_content = getattr(response, "server_content", None)
            if not server_content:
                continue

            model_turn = getattr(server_content, "model_turn", None)
            if model_turn:
                for part in model_turn.parts or []:
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data and inline_data.data:
                        await on_audio(inline_data.data)

            input_tx = getattr(server_content, "input_transcription", None)
            if input_tx and input_tx.text:
                await on_event({"type": "transcript", "role": "user", "text": input_tx.text})

            output_tx = getattr(server_content, "output_transcription", None)
            if output_tx and output_tx.text:
                await on_event({"type": "transcript", "role": "her", "text": output_tx.text})

            if getattr(server_content, "interrupted", False):
                await on_event({"type": "interrupted"})

            if getattr(server_content, "turn_complete", False):
                await on_event({"type": "turn_complete"})
