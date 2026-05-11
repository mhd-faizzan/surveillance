import streamlit as st
import cv2
import torch
import numpy as np
import yaml
from ultralytics import YOLO
from collections import deque

from src.models.lstm_classifier import LSTMClassifier, get_device
from src.features.normalize_keypoints import normalize_keypoints
import pandas as pd

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

SEQUENCE_LENGTH = config["features"]["sequence_length"]
INPUT_SIZE = config["features"]["input_size"]
CONFIDENCE_THRESHOLD = config["inference"]["confidence_alert_threshold"]
DEVICE = get_device()
CLASSES = config["data"]["classes"]


@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n-pose.pt")
    lstm = LSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=config["model"]["lstm"]["hidden_size"],
        num_layers=config["model"]["lstm"]["num_layers"],
        num_classes=len(CLASSES),
        dropout=config["model"]["lstm"]["dropout"]
    ).to(DEVICE)

    try:
        lstm.load_state_dict(torch.load(
            config["model"]["lstm"]["save_path"],
            map_location=DEVICE
        ))
        lstm.eval()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False

    return yolo, lstm, model_loaded


def extract_keypoints(yolo: YOLO, frame: np.ndarray) -> np.ndarray:
    img_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC)
    results = yolo(img_resized, verbose=False)

    for result in results:
        if result.keypoints is None:
            return None
        keypoints = result.keypoints.xy.cpu().numpy()
        confidences = result.keypoints.conf.cpu().numpy()
        if len(keypoints) == 0:
            return None
        best = confidences.mean(axis=1).argmax()
        return keypoints[best].flatten()

    return None


def predict_activity(lstm: LSTMClassifier, sequence: list) -> tuple:
    kp_cols = [f"kp_{i}" for i in range(INPUT_SIZE)]
    df = pd.DataFrame(sequence, columns=kp_cols)
    df = normalize_keypoints(df)
    tensor = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = lstm(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        confidence = probs[pred_idx]

    return CLASSES[pred_idx], confidence


st.title("Surveillance Activity Detection")
st.markdown("Real-time suspicious activity detection using YOLOv8 + LSTM")

yolo, lstm, model_loaded = load_models()

if not model_loaded:
    st.warning("LSTM model not found. Train the model first then rerun the app. Running in pose-only mode.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
confidence_threshold = st.slider("Alert threshold", 0.5, 1.0, float(CONFIDENCE_THRESHOLD))

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")
    frame_placeholder = st.empty()
    alert_placeholder = st.empty()
    keypoint_buffer = deque(maxlen=SEQUENCE_LENGTH)
    alert_log = []

    # temporal smoothing — track consecutive predictions
    prediction_buffer = deque(maxlen=config["inference"]["temporal_smoothing_frames"])

    st.markdown("### Live Feed")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_keypoints(yolo, frame)

        if keypoints is not None:
            keypoint_buffer.append(keypoints)

            if len(keypoint_buffer) == SEQUENCE_LENGTH and model_loaded:
                activity, confidence = predict_activity(lstm, list(keypoint_buffer))
                prediction_buffer.append((activity, confidence))

                # only alert if same activity persists for N frames
                if len(prediction_buffer) == config["inference"]["temporal_smoothing_frames"]:
                    activities = [p[0] for p in prediction_buffer]
                    most_common = max(set(activities), key=activities.count)
                    avg_confidence = np.mean([p[1] for p in prediction_buffer if p[0] == most_common])

                    color = (0, 255, 0)
                    if most_common not in ["NormalVideos", "Normal"] and avg_confidence > confidence_threshold:
                        color = (0, 0, 255)
                        alert_msg = f"ALERT: {most_common} ({avg_confidence:.0%})"
                        if not alert_log or alert_log[-1] != alert_msg:
                            alert_log.append(alert_msg)

                    cv2.putText(frame, f"{most_common} {avg_confidence:.0%}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        if alert_log:
            alert_placeholder.error(alert_log[-1])

    cap.release()

    if alert_log:
        st.markdown("### Alert Log")
        for alert in alert_log:
            st.error(alert)
    else:
        st.success("No suspicious activity detected")