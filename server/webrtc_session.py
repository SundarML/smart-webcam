import asyncio
from concurrent.futures import ThreadPoolExecutor

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection
from aiortc.mediastreams import MediaStreamError

from . import inference
from .frame_bus import frame_bus
from .state import state
from .turn_credentials import get_ice_servers

_executor = ThreadPoolExecutor(max_workers=1)


async def build_peer_connection() -> RTCPeerConnection:
    raw_ice_servers = await asyncio.to_thread(get_ice_servers)
    ice_servers = [
        RTCIceServer(urls=s["urls"], username=s.get("username"), credential=s.get("credential"))
        for s in raw_ice_servers
    ]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))

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
