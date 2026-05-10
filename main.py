import logging
import yaml

from src.models.train import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

train(config)