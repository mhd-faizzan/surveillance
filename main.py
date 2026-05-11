import logging
import torch
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.lstm_classifier import LSTMClassifier, get_device
from src.models.autoencoder import train_autoencoder
from src.features.normalize_keypoints import normalize_keypoints
from src.features.build_sequences import build_sequences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["data"]["keypoints_path"])
df = normalize_keypoints(df)

X, y, label_map = build_sequences(df, config["features"]["sequence_length"])

# train autoencoder on Normal class only
normal_label = label_map["NormalVideos"]
normal_mask = y == normal_label
normal_sequences = torch.tensor(X[normal_mask]).to(get_device())

logger = logging.getLogger(__name__)
logger.info("Training autoencoder on %d Normal sequences", len(normal_sequences))

train_autoencoder(normal_sequences, config)
logger.info("Autoencoder training complete")