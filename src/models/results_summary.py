import torch
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.models.lstm_classifier import LSTMClassifier, get_device
from src.features.normalize_keypoints import normalize_keypoints
from src.features.build_sequences import build_sequences

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# load data
df = pd.read_csv(config["data"]["keypoints_path"])
df = normalize_keypoints(df)
X, y, label_map = build_sequences(df, config["features"]["sequence_length"])

_, X_val, _, y_val = train_test_split(
    X, y,
    test_size=config["data"]["test_size"],
    random_state=config["project"]["random_seed"]
)

device = get_device()
X_val = torch.tensor(X_val).to(device)
y_val = torch.tensor(y_val).to(device)

# load model
model = LSTMClassifier(
    input_size=config["features"]["input_size"],
    hidden_size=config["model"]["lstm"]["hidden_size"],
    num_layers=config["model"]["lstm"]["num_layers"],
    num_classes=len(label_map),
    dropout=config["model"]["lstm"]["dropout"]
).to(device)

model.load_state_dict(torch.load(
    config["model"]["lstm"]["save_path"],
    map_location=device
))
model.eval()

with torch.no_grad():
    output = model(X_val)
    preds = output.argmax(dim=1).cpu().numpy()

y_true = y_val.cpu().numpy()
labels = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

report = classification_report(y_true, preds, target_names=labels, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report = df_report.loc[labels]
df_report = df_report[["precision", "recall", "f1-score", "support"]]
df_report = df_report.round(2)

# plot results table
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")

colors = []
for idx, row in df_report.iterrows():
    f1 = row["f1-score"]
    if f1 >= 0.90:
        colors.append(["#c8f7c5", "#c8f7c5", "#c8f7c5", "#f0f0f0"])
    elif f1 >= 0.75:
        colors.append(["#fef9c3", "#fef9c3", "#fef9c3", "#f0f0f0"])
    else:
        colors.append(["#fde8e8", "#fde8e8", "#fde8e8", "#f0f0f0"])

table = ax.table(
    cellText=df_report.values,
    rowLabels=df_report.index,
    colLabels=["Precision", "Recall", "F1 Score", "Samples"],
    cellLoc="center",
    loc="center",
    cellColours=colors
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# legend
green = mpatches.Patch(color="#c8f7c5", label="F1 ≥ 0.90  excellent")
yellow = mpatches.Patch(color="#fef9c3", label="F1 ≥ 0.75  good")
red = mpatches.Patch(color="#fde8e8", label="F1 < 0.75   needs improvement")
ax.legend(handles=[green, yellow, red], loc="lower right", fontsize=9)

plt.title("Per-Class Model Performance", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("assets/results/performance_table.png", dpi=150, bbox_inches="tight")
print("Saved to assets/results/performance_table.png")