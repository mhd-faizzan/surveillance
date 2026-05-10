import logging
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.lstm_classifier import LSTMClassifier, get_device

logger = logging.getLogger(__name__)


def evaluate(model: LSTMClassifier, X_val: torch.Tensor, y_val: torch.Tensor, label_map: dict) -> None:
    device = get_device()
    model.eval()

    with torch.no_grad():
        output = model(X_val.to(device))
        preds = output.argmax(dim=1).cpu().numpy()

    y_true = y_val.cpu().numpy()
    labels = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

    print(classification_report(y_true, preds, target_names=labels))

    # confusion matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("assets/results/confusion_matrix.png")
    logger.info("Saved confusion matrix to assets/results/confusion_matrix.png")