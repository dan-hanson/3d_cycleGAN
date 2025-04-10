import kagglehub
from pathlib import Path

# Define the local data directory
local_data_dir = Path("/data/")

# Ensure the directory exists
local_data_dir.mkdir(parents=True, exist_ok=True)

# Download the latest version to the local data directory
path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation", path=local_data_dir)

print("Path to dataset files:", path)