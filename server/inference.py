import numpy as np
from ultralytics import YOLO

MODEL_PATHS = {
    "ppe": "models/ppe.pt",
    "fire": "models/fire_smoke.pt",
}

MODEL_MODES = {"ppe", "fire", "both"}

_models: dict[str, YOLO] = {}


def _get_model(name: str) -> YOLO:
    if name not in _models:
        _models[name] = YOLO(MODEL_PATHS[name])
    return _models[name]


def run(frame_bgr: np.ndarray, mode: str) -> np.ndarray:
    if mode not in MODEL_MODES:
        raise ValueError(f"Unknown model mode: {mode!r}")

    if mode == "both":
        ppe_results = _get_model("ppe")(frame_bgr, verbose=False)
        annotated = ppe_results[0].plot()
        fire_results = _get_model("fire")(frame_bgr, verbose=False)
        return fire_results[0].plot(img=annotated)

    results = _get_model(mode)(frame_bgr, verbose=False)
    return results[0].plot()
