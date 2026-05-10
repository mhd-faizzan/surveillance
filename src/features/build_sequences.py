import numpy as np
import pandas as pd


def build_sequences(df: pd.DataFrame, sequence_length: int) -> tuple:
    kp_cols = [col for col in df.columns if col.startswith("kp_")]
    label_map = {label: i for i, label in enumerate(sorted(df["label"].unique()))}

    X, y = [], []

    for label, group in df.groupby("label"):
        keypoints = group[kp_cols].values

        # sliding window over frames for each class
        for i in range(len(keypoints) - sequence_length):
            sequence = keypoints[i:i + sequence_length]
            X.append(sequence)
            y.append(label_map[label])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), label_map