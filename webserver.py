import argparse
import os
import time

import cv2
from ultralytics import YOLO

MODEL_PATHS = {
    "ppe": "models/ppe.pt",
    "fire": "models/fire_smoke.pt",
}

WINDOW_TITLES = {
    "ppe": "Smartphone Bodycam - PPE",
    "fire": "Smartphone Bodycam - Fire/Smoke",
    "both": "Smartphone Bodycam - PPE + Fire/Smoke",
}

parser = argparse.ArgumentParser(description="Smartphone bodycam detection prototype")
parser.add_argument(
    "--model",
    choices=["ppe", "fire", "both"],
    default="ppe",
    help="Which model(s) to run: ppe (default), fire, or both",
)
args = parser.parse_args()

if args.model == "both":
    ppe_model = YOLO(MODEL_PATHS["ppe"])
    fire_model = YOLO(MODEL_PATHS["fire"])
else:
    model = YOLO(MODEL_PATHS[args.model])

# Replace this with the EXACT RTSP URL shown on your phone's screen
# smartphone_rtsp_url = "rtsp://192.168.1.5:5540/live.sdp"

# Convert your HTTP link into the hidden RTSP stream address
# smartphone_url = "rtsp://172.20.10.3:8080/h264_pcs.sdp"
smartphone_url = 'http://192.168.29.234:8080/video'

# Initialize video capture
cap = cv2.VideoCapture(smartphone_url)

# PRO TIP: Force OpenCV to use TCP instead of UDP to prevent gray screen/artifacts
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

window_title = WINDOW_TITLES[args.model]

RETRY_DELAY_SECONDS = 1.0
RECONNECT_AFTER_FAILURES = 10

consecutive_failures = 0

while True:
    ret, frame = cap.read()
    if not ret:
        consecutive_failures += 1
        if consecutive_failures >= RECONNECT_AFTER_FAILURES:
            print(f"Lost smartphone stream, reconnecting to {smartphone_url}...")
            cap.release()
            time.sleep(RETRY_DELAY_SECONDS)
            cap = cv2.VideoCapture(smartphone_url)
            consecutive_failures = 0
        else:
            print("Waiting for smartphone stream...")
            time.sleep(RETRY_DELAY_SECONDS)
        continue

    consecutive_failures = 0

    if args.model == "both":
        ppe_results = ppe_model(frame)
        annotated_frame = ppe_results[0].plot()
        fire_results = fire_model(frame)
        annotated_frame = fire_results[0].plot(img=annotated_frame)
    else:
        results = model(frame)
        annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow(window_title, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
