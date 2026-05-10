import os
import logging
import glob

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def get_frame_paths(data_dir: str, split: str) -> list:
    split_dir = os.path.join(data_dir, split)
    frame_paths = []

    for class_dir in sorted(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        frames = glob.glob(os.path.join(class_path, "*.png"))
        for frame in frames:
            frame_paths.append((frame, class_dir))

    logger.info("Found %d frames in %s", len(frame_paths), split)
    return frame_paths


def extract_keypoints_from_frame(model: YOLO, frame_path: str) -> np.ndarray:
    img = cv2.imread(frame_path)
    if img is None:
        return None

    # upscale so yolo can actually detect people in low-res frames
    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_CUBIC)

    results = model(img_resized, verbose=False)

    for result in results:
        if result.keypoints is None:
            return None

        keypoints = result.keypoints.xy.cpu().numpy()
        confidences = result.keypoints.conf.cpu().numpy()

        if len(keypoints) == 0:
            return None

        # pick most confident person
        best_person_idx = confidences.mean(axis=1).argmax()
        kps = keypoints[best_person_idx].flatten()
        return kps

    return None


def extract_and_save_keypoints(data_dir: str, output_path: str, frame_skip: int) -> None:
    model = YOLO("yolov8n-pose.pt")
    rows = []

    for split in ["Train", "Test"]:
        frame_paths = get_frame_paths(data_dir, split)
        frame_paths = frame_paths[::frame_skip]

        for i, (frame_path, label) in enumerate(frame_paths):
            keypoints = extract_keypoints_from_frame(model, frame_path)

            if keypoints is None:
                continue

            row = {"frame_path": frame_path, "label": label, "split": split}
            for j, val in enumerate(keypoints):
                row[f"kp_{j}"] = val

            rows.append(row)

            if i % 500 == 0:
                logger.info("Processed %d / %d frames in %s", i, len(frame_paths), split)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)