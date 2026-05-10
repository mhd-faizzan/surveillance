import numpy as np
import pandas as pd


def normalize_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes keypoint x,y coordinates to [0,1] relative to bounding box.
    This removes position bias — a fight in top-left looks same as bottom-right.
    """
    kp_cols = [col for col in df.columns if col.startswith("kp_")]
    x_cols = kp_cols[0::2]  # even indices are x
    y_cols = kp_cols[1::2]  # odd indices are y

    x_vals = df[x_cols].values
    y_vals = df[y_cols].values

    x_min = x_vals.min(axis=1, keepdims=True)
    x_max = x_vals.max(axis=1, keepdims=True)
    y_min = y_vals.min(axis=1, keepdims=True)
    y_max = y_vals.max(axis=1, keepdims=True)

    # avoid division by zero for frames where all keypoints are 0
    x_range = np.where((x_max - x_min) == 0, 1, x_max - x_min)
    y_range = np.where((y_max - y_min) == 0, 1, y_max - y_min)

    df[x_cols] = (x_vals - x_min) / x_range
    df[y_cols] = (y_vals - y_min) / y_range

    return df