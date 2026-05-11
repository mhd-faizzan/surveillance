import streamlit as st
import cv2
import torch
import numpy as np
import yaml
from ultralytics import YOLO
from collections import deque
import pandas as pd

from src.models.lstm_classifier import LSTMClassifier, get_device
from src.models.autoencoder import Autoencoder, get_anomaly_score
from src.features.normalize_keypoints import normalize_keypoints

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

SEQUENCE_LENGTH = config["features"]["sequence_length"]
INPUT_SIZE = config["features"]["input_size"]
CONFIDENCE_THRESHOLD = config["inference"]["confidence_alert_threshold"]
ANOMALY_THRESHOLD = config["model"]["autoencoder"]["anomaly_threshold"]
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
        lstm_loaded = True
    except FileNotFoundError:
        lstm_loaded = False

    ae_model = Autoencoder(
        input_size=INPUT_SIZE,
        hidden_size=config["model"]["autoencoder"]["hidden_size"],
        sequence_length=SEQUENCE_LENGTH
    ).to(DEVICE)

    try:
        ae_model.load_state_dict(torch.load(
            config["model"]["autoencoder"]["save_path"],
            map_location=DEVICE
        ))
        ae_model.eval()
        ae_loaded = True
    except FileNotFoundError:
        ae_loaded = False

    return yolo, lstm, lstm_loaded, ae_model, ae_loaded


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


# UI
st.title("Surveillance Activity Detection")
st.markdown("Real-time suspicious activity detection using YOLOv8 + LSTM + Anomaly Detection")

yolo, lstm, lstm_loaded, ae_model, ae_loaded = load_models()

col1, col2 = st.columns(2)
with col1:
    if lstm_loaded:
        st.success("LSTM model loaded")
    else:
        st.warning("LSTM model not found")

with col2:
    if ae_loaded:
        st.success("Anomaly detector loaded")
    else:
        st.warning("Anomaly detector not found")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
confidence_threshold = st.slider("Alert threshold", 0.5, 1.0, float(CONFIDENCE_THRESHOLD))
anomaly_threshold = st.slider("Anomaly threshold", 0.0, 0.2, float(ANOMALY_THRESHOLD))

if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")
    frame_placeholder = st.empty()
    alert_placeholder = st.empty()
    status_placeholder = st.empty()
    keypoint_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction_buffer = deque(maxlen=config["inference"]["temporal_smoothing_frames"])
    alert_log = []

    st.markdown("### Live Feed")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_keypoints(yolo, frame)

        if keypoints is not None:
            keypoint_buffer.append(keypoints)

            if len(keypoint_buffer) == SEQUENCE_LENGTH:
                sequence_list = list(keypoint_buffer)

                # LSTM prediction
                if lstm_loaded:
                    activity, confidence = predict_activity(lstm, sequence_list)
                    prediction_buffer.append((activity, confidence))
                else:
                    activity, confidence = "Unknown", 0.0

                # anomaly detection
                anomaly_score = 0.0
                if ae_loaded:
                    kp_cols = [f"kp_{i}" for i in range(INPUT_SIZE)]
                    df_seq = pd.DataFrame(sequence_list, columns=kp_cols)
                    df_seq = normalize_keypoints(df_seq)
                    seq_tensor = torch.tensor(df_seq.values, dtype=torch.float32).to(DEVICE)
                    anomaly_score = get_anomaly_score(ae_model, seq_tensor)

                # temporal smoothing
                if len(prediction_buffer) == config["inference"]["temporal_smoothing_frames"]:
                    activities = [p[0] for p in prediction_buffer]
                    most_common = max(set(activities), key=activities.count)
                    avg_confidence = np.mean([p[1] for p in prediction_buffer if p[0] == most_common])

                    is_anomaly = anomaly_score > anomaly_threshold
                    is_suspicious = most_common not in ["NormalVideos", "Normal"] and avg_confidence > confidence_threshold

                    color = (0, 255, 0)
                    if is_suspicious or is_anomaly:
                        color = (0, 0, 255)
                        reason = []
                        if is_suspicious:
                            reason.append(f"{most_common} ({avg_confidence:.0%})")
                        if is_anomaly:
                            reason.append(f"anomaly score: {anomaly_score:.4f}")
                        alert_msg = "ALERT: " + " | ".join(reason)
                        if not alert_log or alert_log[-1] != alert_msg:
                            alert_log.append(alert_msg)

                    cv2.putText(frame, f"{most_common} {avg_confidence:.0%}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"anomaly: {anomaly_score:.4f}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    status_placeholder.info(f"Activity: {most_common} | Confidence: {avg_confidence:.0%} | Anomaly Score: {anomaly_score:.4f}")

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