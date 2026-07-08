import json

from aiortc import RTCSessionDescription
from fastapi import WebSocket, WebSocketDisconnect

from .inference import MODEL_MODES
from .state import DEFAULT_MODEL_MODE, state
from .webrtc_session import build_peer_connection, wait_for_ice_gathering_complete


def _log_candidates(label: str, sdp: str) -> None:
    lines = [line for line in sdp.splitlines() if line.startswith("a=candidate")]
    if not lines:
        print(f"[webrtc] {label}: no candidates in SDP")
        return
    for line in lines:
        print(f"[webrtc] {label}: {line}")


async def handle_signaling(websocket: WebSocket) -> None:
    await websocket.accept()

    if not state.try_acquire_broadcaster():
        await websocket.send_json({"type": "error", "message": "A broadcast is already in progress."})
        await websocket.close()
        return

    pc = build_peer_connection()
    try:
        async for raw in websocket.iter_text():
            message = json.loads(raw)
            msg_type = message.get("type")

            if msg_type == "hello":
                model = message.get("model", DEFAULT_MODEL_MODE)
                if model not in MODEL_MODES:
                    await websocket.send_json(
                        {"type": "error", "message": f"Unknown model mode: {model!r}"}
                    )
                    continue
                state.set_model_mode(model)

            elif msg_type == "offer":
                offer = RTCSessionDescription(sdp=message["sdp"], type="offer")
                _log_candidates("offer (client)", offer.sdp)
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await wait_for_ice_gathering_complete(pc)
                _log_candidates("answer (server)", pc.localDescription.sdp)
                await websocket.send_json({"type": "answer", "sdp": pc.localDescription.sdp})

            elif msg_type == "bye":
                break
    except WebSocketDisconnect:
        pass
    finally:
        await pc.close()
        state.release_broadcaster()
