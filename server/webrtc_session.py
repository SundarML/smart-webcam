import asyncio
from concurrent.futures import ThreadPoolExecutor

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection
from aiortc.mediastreams import MediaStreamError

from . import inference
from .frame_bus import frame_bus
from .state import state

# Public test TURN service (Open Relay Project) — needed because a phone
# behind carrier-grade NAT often can't establish a direct/STUN-only ICE path
# to a PaaS container, which typically has no reachable public UDP port either.
ICE_SERVERS = [
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    RTCIceServer(
        urls=["turn:openrelay.metered.ca:80"],
        username="openrelayproject",
        credential="openrelayproject",
    ),
    RTCIceServer(
        urls=["turn:openrelay.metered.ca:443", "turn:openrelay.metered.ca:443?transport=tcp"],
        username="openrelayproject",
        credential="openrelayproject",
    ),
    RTCIceServer(
        urls=["turns:openrelay.metered.ca:443?transport=tcp"],
        username="openrelayproject",
        credential="openrelayproject",
    ),
]

_executor = ThreadPoolExecutor(max_workers=1)


def build_peer_connection() -> RTCPeerConnection:
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ICE_SERVERS))

    @pc.on("track")
    def on_track(track) -> None:
        print(f"[webrtc] track received: kind={track.kind}")
        if track.kind == "video":
            asyncio.ensure_future(_consume_video(track))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        print(f"[webrtc] connectionState={pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        print(f"[webrtc] iceConnectionState={pc.iceConnectionState}")

    return pc


async def wait_for_ice_gathering_complete(pc: RTCPeerConnection, timeout: float = 10.0) -> None:
    if pc.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _on_change() -> None:
        if pc.iceGatheringState == "complete":
            done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[webrtc] ICE gathering timed out after {timeout}s, state={pc.iceGatheringState}")


async def _consume_video(track) -> None:
    loop = asyncio.get_event_loop()
    inference_busy = False

    def _on_inference_done(fut: "asyncio.Future") -> None:
        nonlocal inference_busy
        inference_busy = False
        exc = fut.exception()
        if exc is not None:
            print(f"Inference error: {exc!r}")
            return
        frame_bus.publish(fut.result())

    while True:
        try:
            frame = await track.recv()
        except MediaStreamError:
            break

        if inference_busy:
            continue

        inference_busy = True
        img = frame.to_ndarray(format="bgr24")
        mode = state.get_model_mode()
        future = loop.run_in_executor(_executor, inference.run, img, mode)
        future.add_done_callback(_on_inference_done)
