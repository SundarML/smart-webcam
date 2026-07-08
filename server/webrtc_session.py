import asyncio
from concurrent.futures import ThreadPoolExecutor

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection
from aiortc.mediastreams import MediaStreamError

from . import inference
from .frame_bus import frame_bus
from .state import state

ICE_SERVERS = [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]

_executor = ThreadPoolExecutor(max_workers=1)


def build_peer_connection() -> RTCPeerConnection:
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ICE_SERVERS))

    @pc.on("track")
    def on_track(track) -> None:
        if track.kind == "video":
            asyncio.ensure_future(_consume_video(track))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()

    return pc


async def wait_for_ice_gathering_complete(pc: RTCPeerConnection, timeout: float = 3.0) -> None:
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
        pass


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
