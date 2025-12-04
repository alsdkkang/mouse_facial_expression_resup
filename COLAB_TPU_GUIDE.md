# Google Colab TPU Training Guide (Folds 2-4)

This guide helps you train the **ReSupModel** for Folds 2, 3, and 4 using Google Colab's TPU runtime.

## 1. Runtime Setup
1.  Go to **Runtime** > **Change runtime type**.
2.  Select **TPU v2** (or available TPU) as the Hardware accelerator.
3.  Click **Save**.

## 2. Install Dependencies
Copy and run this cell to install necessary libraries.
```python
# 1. Install general dependencies (ignoring blinker to avoid Colab conflict)
!pip install lightning mlflow torchmetrics python-dotenv --ignore-installed blinker

# 2. Install Torch XLA for TPU (Universal method)
# This automatically finds the compatible version for the current Colab runtime
!pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

# 3. Verify XLA installation
import torch_xla.core.xla_model as xm
print(f"TPU Device: {xm.xla_device()}")
```

## 3. Connect Google Drive & Setup Data
Mount Google Drive, copy files to the local runtime (for speed), and unzip.

```python
from google.colab import drive
import os
import shutil

# 1. Mount Drive
drive.mount('/content/drive')

# 2. Define your Drive path (Change this to where your files are!)
DRIVE_PATH = '/content/drive/MyDrive/mouse-facial-expressions' 

# 3. Setup Workspace in /content (Faster I/O than Drive)
%cd /content

if os.path.exists(DRIVE_PATH):
    print(f"Copying files from {DRIVE_PATH}...")
    # Copy zip files
    shutil.copy(f'{DRIVE_PATH}/project_code.zip', '/content/project_code.zip')
    shutil.copy(f'{DRIVE_PATH}/data.zip', '/content/data.zip')
    
    print("Unzipping...")
    !unzip -q -o project_code.zip # -o to overwrite if exists
    !unzip -q -o data.zip
    print("Done! Workspace is ready in /content")
    
    # Create .env file for Colab paths
    print("Creating .env file for Colab...")
    with open('.env', 'w') as f:
        f.write("MFE_RAW_VIDEO_FOLDER=/content/data/raw_videos\n")
        f.write("MFE_PROCESSED_VIDEO_FOLDER=/content/data/processed_videos\n")
        f.write("MFE_EXTRACTED_FRAMES_FOLDER=/content/data/extracted_frames\n")
        f.write("MFE_DLC_FACIAL_LABELS_FOLDER=/content/data/dlc_labels\n")
        f.write("MFE_DLC_FACIAL_PROJECT_PATH=/content/data/dlc_project\n")
        f.write("MFE_TASKS=/content/data/processed\n")
        f.write("MFE_RAW_CSV_FOLDER=/content/data/raw_csvs\n")
        f.write("MFE_VERSION=1.1\n")
    print(".env file created successfully!")
    
else:
    print(f"Error: Path {DRIVE_PATH} not found. Please check your Drive path.")
```

## 4. Run Training (Folds 2-4)
Run the following command to start training.
We add `PYTHONPATH=.` to ensure Python finds the `mouse_facial_expressions` package.

```python
import sys
import os

# Verify we are in the right directory (should see 'mouse_facial_expressions' folder)
if not os.path.exists('mouse_facial_expressions'):
    print("Error: 'mouse_facial_expressions' folder not found in current directory!")
    print("Current directory contents:", os.listdir('.'))
else:
    print("Found package. Starting training...")
    
    # Run training module directly with PYTHONPATH set
    !PYTHONPATH=. python3 -m mouse_facial_expressions.models.train_task1_resup_model \
        --folds "2,3,4" \
        --epochs 20 \
        --learning_rate 0.0001 \
        --num_frames 10 \
        --frame_stride 2 \
        --train_batch_size 128 \
        --test_batch_size 128 \
        --use_soft_labels True \
        --label_smoothing 0.2 \
        --dropout 0.5 \
        --dataset_version "1.1" \
        --train_augmentation "TrivialAugmentWide" \
        --seed 97531 \
        --accelerator "auto"
```

> [!TIP]
> **Batch Size**: TPUs have large memory (High RAM). I increased the batch size to `128` (from 12) to speed up training significantly. If you encounter errors, try reducing it to `64`.

## 5. Download Results
After training, download the checkpoints.
```python
from google.colab import files
!zip -r checkpoints_folds234.zip models/checkpoints_resup
files.download('checkpoints_folds234.zip')
```
