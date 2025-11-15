#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
readonly ENV_NAME="pbip"
readonly PYTHON_VERSION="3.9"
readonly REPO_DIR="PBIP"

# Google Drive File IDs
readonly DATA_ZIP_GDRIVE_ID="1ExQsVCF3AE_mF5aO9tgeK5LN6-10KIQ6"
readonly PRETRAINED_ZIP_GDRIVE_ID="1RIuYLnVijOXeUVP0hXMTlNEBHWbd9iv-"
readonly BEST_MODEL_WEIGHT_GDRIVE_ID="1X-Bd7JCgIPq2JXh1WHHm5Gzv42WInbqw"

# --- Helper Functions ---
usage() {
  echo "Usage: $0 {train|test}"
  echo "  train: Set up the environment, download data, and run the training process."
  echo "  test:  Set up the environment, download data, and run the testing process."
  exit 1
}


# --- Script Start ---

# 1. Validate User Input and Prerequisites
if [[ "$1" != "train" && "$1" != "test" ]]; then
  echo "Error: Invalid mode specified."
  usage
fi
readonly MODE=$1

# Check for required commands
if ! command -v conda &> /dev/null; then
    echo "Error: conda could not be found. Please install Anaconda or Miniconda."
    exit 1
fi
if ! command -v gdown &> /dev/null; then
    echo "Error: gdown could not be found. Please install it by running 'pip install gdown'."
    exit 1
fi

echo "Starting PBIP script in '$MODE' mode..."


# 2. Create and Prepare Conda Environment
echo "----------------------------------------"
echo "STEP 1: Setting up the '$ENV_NAME' Conda environment..."
if ! conda env list | grep -q "$ENV_NAME"; then
  echo "Creating new Conda environment '$ENV_NAME'..."
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
else
  echo "Environment '$ENV_NAME' already exists. Skipping creation."
fi


# 3. Clone Repository
echo "----------------------------------------"
echo "STEP 2: Cloning the PBIP repository..."
if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/tom1209-netizen/PBIP.git
else
  echo "Repository '$REPO_DIR' already exists. Skipping clone."
fi
cd "$REPO_DIR"


# 4. Download and Extract Data
echo "----------------------------------------"
echo "STEP 3: Downloading data from Google Drive..."

# Download and extract main dataset
echo "Downloading dataset..."
gdown --id "$DATA_ZIP_GDRIVE_ID" -O data.zip
echo "Extracting dataset to ./data ..."
unzip -qo data.zip -d ./data
rm data.zip
echo "Dataset extracted."

# Download and extract pretrained models
echo "Downloading pretrained models..."
mkdir -p pretrained
gdown --id "$PRETRAINED_ZIP_GDRIVE_ID" -O pretrained.zip
echo "Extracting models to ./pretrained ..."
unzip -qo pretrained.zip -d ./pretrained
rm pretrained.zip
echo "Pretrained models extracted."


# 5. Install Python Dependencies
echo "----------------------------------------"
echo "STEP 4: Installing Python requirements..."
conda run -n "$ENV_NAME" pip install -r requirements.txt
echo "Requirements installed."


# 6. Execute Train or Test Logic
echo "----------------------------------------"
if [ "$MODE" == "train" ]; then
  echo "STEP 5: Starting Training..."
  conda run -n "$ENV_NAME" pip install icecream
  conda run -n "$ENV_NAME" python train_stage_1.py --config ./work_dirs/bcss/classification/config.yaml --gpu 0
  echo "Training finished."

elif [ "$MODE" == "test" ]; then
  echo "STEP 5: Starting Testing..."

  # Download the fine-tuned model weights required for testing
  echo "Downloading best model weights..."
  mkdir -p ./work_dirs/bcss/checkpoints
  gdown --id "$BEST_MODEL_WEIGHT_GDRIVE_ID" -O ./work_dirs/bcss/checkpoints/best_cam.pth
  echo "Best model weights downloaded."

  # Install test-specific dependencies and run the test
  conda run -n "$ENV_NAME" pip install git+https://github.com/lucasb-eyer/pydensecrf.git
  conda run -n "$ENV_NAME" python test_and_visualize.py --config ./work_dirs/bcss/classification/config.yaml --checkpoint ./work_dirs/bcss/checkpoints/best_cam.pth

  # Zip the results
  echo "Zipping visualization results..."
  zip -rj ../test_visualizations.zip ./work_dirs/bcss/checkpoints/test_visualizations
  echo "Testing finished. Visualizations are saved in 'test_visualizations.zip' in the parent directory."
fi

echo "All done!"