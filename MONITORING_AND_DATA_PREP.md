# Training Progress Monitoring & Chronic Data Preparation Guide

## Issue: Training Crashes on Network Drive

The training keeps crashing because reading images from Google Drive over the network is unreliable. Here are your options:

### Solution 1: Copy Frames Locally (Recommended)

**Copy a subset of frames to local storage:**

```bash
# Create local frames directory
mkdir -p data/local_frames

# Copy frames for a few mice to test (adjust based on your disk space)
# This copies frames for mice m1-m5 and f1-f5 (10 mice total)
cd "/Users/minakang/Library/CloudStorage/GoogleDrive-minakang46@gmail.com/Other computers/My Laptop/Desktop/mouse-facial-expressions-2023-main/mouse-facial-expressions-2023-main/raw/20230627"

for mouse in m1 m2 m3 m4 m5 f1 f2 f3 f4 f5; do
    cp -r ${mouse}_* /Users/minakang/Desktop/mouse-facial-expressions-2023-main/data/local_frames/
done
```

**Update .env to use local frames:**
```bash
MFE_EXTRACTED_FRAMES_FOLDER=/Users/minakang/Desktop/mouse-facial-expressions-2023-main/data/local_frames
```

**Then regenerate dataset and train:**
```bash
# Regenerate dataset with local frames
python mouse_facial_expressions/data/make_datasets.py task1 --version "1.1" --frameset_size 1

# Train
python mouse_facial_expressions/models/train_task1_baseline_model.py --epochs 5 --dataset_version "1.1"
```

### Solution 2: Use Pre-trained ImageNet Weights

Since full training is problematic, you can use the ImageNet pre-trained ResNet34 as your starting point:

```python
# The model already loads ImageNet weights by default:
model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
```

This is actually a good baseline for transfer learning!

---

## Monitoring Training Progress

### Method 1: Check Terminal Output

The training script prints progress every batch:
```bash
# Look for lines like:
# Epoch 0:  10%|█     | 50/500 [02:30<22:30,  0.33it/s]
# 2025-11-22 18:12:25 - __main__ - INFO - Train epoch 0, loss 0.449, accuracy 0.900
```

### Method 2: Check Saved Checkpoints

```bash
# List checkpoints (shows file size and timestamp)
ls -lh models/checkpoints/

# Example output:
# task1-fold0-epoch=0-val_acc=0.85.ckpt
# task1-fold0-epoch=1-val_acc=0.90.ckpt
# task1-fold0-epoch=2-val_acc=0.92.ckpt  <- Best model so far
```

The filename tells you:
- **fold0**: Which cross-validation fold
- **epoch=2**: Which epoch
- **val_acc=0.92**: Validation accuracy (92%)

### Method 3: Use MLflow UI (if available)

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
# View metrics, parameters, and artifacts
```

### Method 4: Check Process Status

```bash
# Check if training is still running
ps aux | grep train_task1

# Monitor CPU/memory usage
top -pid $(pgrep -f train_task1)
```

---

## Preparing Your Chronic Data

### Step 1: Organize Your Video Files

Create a directory structure:
```
chronic_data/
├── videos/
│   ├── c1_baseline.mp4
│   ├── c1_week1.mp4
│   ├── c2_baseline.mp4
│   └── c2_week1.mp4
└── metadata/
    ├── treatments_chronic.csv
    └── raw_videos_chronic.csv
```

### Step 2: Create Metadata CSVs

**treatments_chronic.csv:**
```csv
mouse,date_of_birth,treatment,injection_time,notes
c1,2024-01-15,control,10:00,sham surgery
c2,2024-01-15,chronic_pain,10:15,CCI model left sciatic
c3,2024-01-16,control,10:30,sham surgery
c4,2024-01-16,chronic_pain,10:45,CCI model left sciatic
c5,2024-01-17,control,11:00,sham surgery
c6,2024-01-17,chronic_pain,11:15,CCI model left sciatic
```

**raw_videos_chronic.csv:**
```csv
animal,recording,camera,year,month,day,hour,minutes,seconds,start,end,discard
c1,1,cam1,2024,2,1,10,0,0,0,-1,0
c1,7,cam1,2024,2,8,10,0,0,0,-1,0
c1,14,cam1,2024,2,15,10,0,0,0,-1,0
c2,1,cam1,2024,2,1,10,15,0,0,-1,0
c2,7,cam1,2024,2,8,10,15,0,0,-1,0
c2,14,cam1,2024,2,15,10,15,0,0,-1,0
```

**Key columns explained:**
- `animal`: Mouse ID (e.g., c1, c2, c3...)
- `recording`: Time point (1=baseline, 7=week 1, 14=week 2, etc.)
- `camera`: Camera identifier
- `year,month,day,hour,minutes,seconds`: Video timestamp for matching
- `start,end`: Trim points in MM:SS format (use "0" and "-1" for full video)
- `discard`: Set to 1 to exclude bad videos

### Step 3: Extract Frames from Your Videos

**Option A: If you have DeepLabCut model**
```bash
# Process videos
python mouse_facial_expressions/data/raw_video_processing.py rename \
  --input_folder chronic_data/videos \
  --output_folder chronic_data/processed \
  --meta_csv chronic_data/metadata/raw_videos_chronic.csv

# Extract facial landmarks
python mouse_facial_expressions/data/raw_video_processing.py dlc_process_videos \
  --dlc_project /path/to/your/dlc/project \
  --input_folder chronic_data/processed \
  --output_folder chronic_data/dlc

# Extract aligned frames
python mouse_facial_expressions/data/raw_video_processing.py extract_frames \
  --processed_videos_folder chronic_data/processed \
  --dlc_facial_labels_folder chronic_data/dlc \
  --extracted_frames_folder chronic_data/frames
```

**Option B: If you have pre-cropped frames**

Organize frames like this:
```
chronic_data/frames/
├── c1_rec1_baseline/
│   ├── frame00001.png
│   ├── frame00002.png
│   └── ...
├── c1_rec7_week1/
│   ├── frame00001.png
│   └── ...
└── c2_rec1_baseline/
    └── ...
```

### Step 4: Update Environment for Chronic Data

Create `chronic_data/.env`:
```bash
MFE_RAW_VIDEO_FOLDER=/path/to/chronic_data/videos
MFE_RAW_CSV_FOLDER=/path/to/chronic_data/metadata
MFE_VERSION=chronic_v1
MFE_EXTRACTED_FRAMES_FOLDER=/path/to/chronic_data/frames
MFE_TASKS=/Users/minakang/Desktop/mouse-facial-expressions-2023-main/data/processed
```

### Step 5: Modify Dataset Creation Script

Edit `mouse_facial_expressions/data/make_datasets.py`:

**Find this section (around line 91-100):**
```python
logger.info('Assigning labels')
combined_df['label'] = np.nan
combined_df.loc[(combined_df.recording == 1) & (combined_df.treatment == 'saline'), 'label'] = 0
combined_df.loc[(combined_df.recording == 1) & (combined_df.treatment == 'low'), 'label'] = 0
# ... etc
```

**Replace with your chronic labeling:**
```python
logger.info('Assigning labels for chronic pain study')
combined_df['label'] = np.nan

# Baseline (recording 1) for all groups = control (0)
combined_df.loc[(combined_df.recording == 1), 'label'] = 0

# Week 1 (recording 7) for chronic_pain group = pain (1)
combined_df.loc[(combined_df.recording == 7) & (combined_df.treatment == 'chronic_pain'), 'label'] = 1

# Optional: Week 1 for control group = control (0)
combined_df.loc[(combined_df.recording == 7) & (combined_df.treatment == 'control'), 'label'] = 0

# Drop unlabeled samples
combined_df = combined_df.dropna(subset='label')
```

### Step 6: Generate Chronic Dataset

```bash
# Set environment
export $(cat chronic_data/.env | xargs)

# Generate dataset
python mouse_facial_expressions/data/make_datasets.py task1 \
  --version "chronic_v1" \
  --frameset_size 1 \
  --train_size 5000 \
  --test_size 500 \
  --kfold_splits 5
```

### Step 7: Verify Dataset

```bash
# Check generated files
ls -lh data/processed/task-chronic_v1/

# Should see:
# - dataset_df.pkl
# - fold0.pkl, fold1.pkl, ..., fold4.pkl
# - eval.pkl
# - README.txt
```

---

## Quick Start: Minimal Chronic Data Setup

If you just want to test with minimal data:

1. **Prepare 2-3 mice minimum** (1 control, 1-2 chronic)
2. **2 time points** (baseline + 1 week)
3. **100-200 frames per mouse** per time point
4. **Create simple CSVs** as shown above
5. **Run dataset creation** with smaller sizes:
   ```bash
   python mouse_facial_expressions/data/make_datasets.py task1 \
     --version "chronic_test" \
     --frameset_size 1 \
     --train_size 500 \
     --test_size 100 \
     --kfold_splits 2
   ```

---

## Troubleshooting

**Q: "No such file or directory" error**
- Check that `MFE_EXTRACTED_FRAMES_FOLDER` points to correct location
- Verify frame folder names match pattern: `{mouse}_rec{recording}_{label}/`

**Q: "No frames found"**
- Ensure frames are named `frameXXXXX.png` (5 digits)
- Check that folders contain `.png` files

**Q: Dataset generation is slow**
- The parallel optimization should help
- If still slow, reduce number of mice or frames

**Q: Not enough data for cross-validation**
- Reduce `--kfold_splits` to 2 or 3
- Ensure you have at least 4-5 mice per treatment group
