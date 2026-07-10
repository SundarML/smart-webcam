import asyncio
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, Request, WebSocket
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from . import auth
from .frame_bus import frame_bus
from .signaling import handle_signaling
from .turn_credentials import get_ice_servers


def _safe_next(next_url: str) -> str:
    if next_url.startswith("/") and not next_url.startswith("//"):
        return next_url
    return "/"

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

MJPEG_BOUNDARY = "frame"
TARGET_FPS = 12
MIN_INTERVAL = 1.0 / TARGET_FPS
SESSION_MAX_AGE = 14 * 24 * 3600

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=auth.SESSION_SECRET, max_age=SESSION_MAX_AGE)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/login")
async def login_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "login.html")


@app.post("/login")
async def login_submit(
    request: Request, password: str = Form(...), next: str = Form("")
) -> RedirectResponse:
    safe_next = _safe_next(next)
    if not auth.check_password(password):
        return RedirectResponse(f"/login?error=1&next={quote(safe_next)}", status_code=303)
    request.session["authed"] = True
    return RedirectResponse(safe_next, status_code=303)


@app.get("/logout")
async def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/broadcast")
async def broadcast_page(request: Request):
    if not auth.is_authed(request):
        return RedirectResponse("/login?next=/broadcast", status_code=303)
    return FileResponse(STATIC_DIR / "broadcast.html")


@app.get("/view")
async def view_page(request: Request):
    if not auth.is_authed(request):
        return RedirectResponse("/login?next=/view", status_code=303)
    return FileResponse(STATIC_DIR / "view.html")


@app.get("/ice-servers")
async def ice_servers(request: Request):
    if not auth.is_authed(request):
        return Response(status_code=401)
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
async def stream_mjpg(request: Request):
    if not auth.is_authed(request):
        return Response(status_code=401)
    return StreamingResponse(
        _mjpeg_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
    )
