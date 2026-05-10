import logging
import yaml

from src.data.extract_keypoints import extract_and_save_keypoints

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

extract_and_save_keypoints(
    data_dir=config["data"]["raw_path"],
    output_path=config["data"]["keypoints_path"],
    confidence_threshold=config["features"]["confidence_threshold"],
    frame_skip=config["features"]["frame_skip"]
)