import logging
import torch
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.lstm_classifier import LSTMClassifier, get_device
from src.models.evaluate import evaluate
from src.features.normalize_keypoints import normalize_keypoints
from src.features.build_sequences import build_sequences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# load and prepare data
df = pd.read_csv(config["data"]["keypoints_path"])
df = normalize_keypoints(df)

X, y, label_map = build_sequences(df, config["features"]["sequence_length"])

_, X_val, _, y_val = train_test_split(
    X, y,
    test_size=config["data"]["test_size"],
    random_state=config["project"]["random_seed"]
)

X_val = torch.tensor(X_val).to(get_device())
y_val = torch.tensor(y_val).to(get_device())

# load trained model
model = LSTMClassifier(
    input_size=config["features"]["input_size"],
    hidden_size=config["model"]["lstm"]["hidden_size"],
    num_layers=config["model"]["lstm"]["num_layers"],
    num_classes=len(label_map),
    dropout=config["model"]["lstm"]["dropout"]
).to(get_device())

model.load_state_dict(torch.load(
    config["model"]["lstm"]["save_path"],
    map_location=get_device()
))

import os
os.makedirs("assets/results", exist_ok=True)

evaluate(model, X_val, y_val, label_map)