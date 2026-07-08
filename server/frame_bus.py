import threading

import cv2
import numpy as np

JPEG_QUALITY = 80


class FrameBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._version = 0

    def publish(self, frame_bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._version += 1

    def latest(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._jpeg, self._version


frame_bus = FrameBus()
