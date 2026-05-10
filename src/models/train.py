import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from src.models.lstm_classifier import LSTMClassifier, get_device
from src.features.normalize_keypoints import normalize_keypoints
from src.features.build_sequences import build_sequences

logger = logging.getLogger(__name__)


def load_keypoints(keypoints_path: str) -> pd.DataFrame:
    if not __import__('os').path.exists(keypoints_path):
        raise FileNotFoundError(f"keypoints.csv not found at {keypoints_path}. Run keypoint extraction first.")
    df = pd.read_csv(keypoints_path)
    logger.info("Loaded %d rows from %s", len(df), keypoints_path)
    return df


def train(config: dict) -> None:
    device = get_device()
    logger.info("Using device: %s", device)

    df = load_keypoints(config["data"]["keypoints_path"])
    df = normalize_keypoints(df)

    sequence_length = config["features"]["sequence_length"]
    X, y, label_map = build_sequences(df, sequence_length)
    logger.info("Built %d sequences across %d classes", len(X), len(label_map))
    logger.info("Label map: %s", label_map)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["project"]["random_seed"]
    )

    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(device)
    X_val = torch.tensor(X_val).to(device)
    y_val = torch.tensor(y_val).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config["model"]["lstm"]["batch_size"],
        shuffle=True
    )

    model = LSTMClassifier(
        input_size=config["features"]["input_size"],
        hidden_size=config["model"]["lstm"]["hidden_size"],
        num_layers=config["model"]["lstm"]["num_layers"],
        num_classes=len(label_map),
        dropout=config["model"]["lstm"]["dropout"]
    ).to(device)

    # weighted loss to handle class imbalance 
    class_counts = np.bincount(y_train.cpu().numpy())
    weights = 1.0 / class_counts
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["lstm"]["learning_rate"]
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stopping_patience = 5

    for epoch in range(config["model"]["lstm"]["epochs"]):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()
            val_preds = val_output.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        logger.info(
            "Epoch %d/%d — train loss: %.4f val loss: %.4f val acc: %.4f",
            epoch + 1, config["model"]["lstm"]["epochs"],
            total_loss / len(train_loader), val_loss, val_acc
        )

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            __import__('os').makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), config["model"]["lstm"]["save_path"])
            logger.info("Saved best model to %s", config["model"]["lstm"]["save_path"])
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break