import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .frame_bus import frame_bus
from .signaling import handle_signaling
from .turn_credentials import get_ice_servers

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

MJPEG_BOUNDARY = "frame"
TARGET_FPS = 12
MIN_INTERVAL = 1.0 / TARGET_FPS

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/broadcast")
async def broadcast_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "broadcast.html")


@app.get("/view")
async def view_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "view.html")


@app.get("/ice-servers")
async def ice_servers() -> dict:
    return {"iceServers": get_ice_servers()}


@app.websocket("/ws/signal")
async def ws_signal(websocket: WebSocket) -> None:
    await handle_signaling(websocket)


async def _mjpeg_generator():
    last_version = -1
    while True:
        jpeg, version = frame_bus.latest()
        if jpeg is not None and version != last_version:
            last_version = version
            yield (
                b"--" + MJPEG_BOUNDARY.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )
        await asyncio.sleep(MIN_INTERVAL)


@app.get("/stream.mjpg")
async def stream_mjpg() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
    )
