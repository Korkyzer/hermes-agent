import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .auth import require_request_auth, require_websocket_auth
from .gemini_live import GeminiLiveBridge


load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("voice-her")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Voice Her Prototype")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/")
async def index(_: None = Depends(require_request_auth)) -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    if not await require_websocket_auth(websocket):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=24)
    text_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)

    async def send_audio(data: bytes) -> None:
        await websocket.send_bytes(data)

    async def send_event(event: dict) -> None:
        await websocket.send_json(event)

    async def receive_client() -> None:
        try:
            while True:
                message = await websocket.receive()
                if message.get("bytes") is not None:
                    await audio_queue.put(message["bytes"])
                    continue

                if message.get("text"):
                    payload = json.loads(message["text"])
                    message_type = payload.get("type")
                    if message_type == "stop_audio":
                        await audio_queue.put(None)
                    elif message_type == "text" and payload.get("text"):
                        await text_queue.put(payload["text"])
        except WebSocketDisconnect:
            await audio_queue.put(None)
            await text_queue.put(None)

    receiver = asyncio.create_task(receive_client())
    bridge = GeminiLiveBridge()

    try:
        await send_event({"type": "ready", "sampleRate": 24000})
        await bridge.run(audio_queue, text_queue, send_audio, send_event)
    except Exception as exc:
        logger.exception("Live session failed")
        try:
            await send_event({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        receiver.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8760")),
        reload=False,
    )
