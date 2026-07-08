import threading

DEFAULT_MODEL_MODE = "ppe"


class AppState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._broadcaster_active = False
        self._model_mode = DEFAULT_MODEL_MODE

    def try_acquire_broadcaster(self) -> bool:
        with self._lock:
            if self._broadcaster_active:
                return False
            self._broadcaster_active = True
            return True

    def release_broadcaster(self) -> None:
        with self._lock:
            self._broadcaster_active = False

    def set_model_mode(self, mode: str) -> None:
        with self._lock:
            self._model_mode = mode

    def get_model_mode(self) -> str:
        with self._lock:
            return self._model_mode


state = AppState()
