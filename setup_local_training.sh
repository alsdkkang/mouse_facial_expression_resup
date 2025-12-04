#!/bin/bash
# Script to copy LPS data locally and train the model

set -e  # Exit on error

echo "========================================="
echo "Local Training Setup for LPS Model"
echo "========================================="
echo ""

# Configuration
GOOGLE_DRIVE_PATH="/Users/minakang/Library/CloudStorage/GoogleDrive-minakang46@gmail.com/Other computers/My Laptop/Desktop/mouse-facial-expressions-2023-main/mouse-facial-expressions-2023-main/raw/20230627"
LOCAL_FRAMES_PATH="data/local_frames"
PROJECT_DIR="/Users/minakang/Desktop/mouse-facial-expressions-2023-main"

cd "$PROJECT_DIR"

# Step 1: Create local directory
echo "Step 1: Creating local frames directory..."
mkdir -p "$LOCAL_FRAMES_PATH"

# Step 2: Copy frames from Google Drive to local
echo "Step 2: Copying frames from Google Drive (119 MB)..."
echo "This may take a few minutes..."
rsync -av --progress "$GOOGLE_DRIVE_PATH/" "$LOCAL_FRAMES_PATH/"

echo ""
echo "✓ Frames copied successfully!"
echo "  Location: $PROJECT_DIR/$LOCAL_FRAMES_PATH"
echo "  Size: $(du -sh $LOCAL_FRAMES_PATH | cut -f1)"
echo ""

# Step 3: Update .env file
echo "Step 3: Updating .env file to use local frames..."
cat > .env << EOF
MFE_RAW_VIDEO_FOLDER=$PROJECT_DIR/data/raw
MFE_RAW_CSV_FOLDER=$PROJECT_DIR/data/raw
MFE_VERSION=20230627
MFE_PROCESSED_VIDEO_FOLDER=$PROJECT_DIR/data/processed
MFE_EXTRACTED_FRAMES_FOLDER=$PROJECT_DIR/$LOCAL_FRAMES_PATH
MFE_TASKS=$PROJECT_DIR/data/processed
EOF

echo "✓ Environment configured!"
echo ""

# Step 4: Regenerate dataset with local frames
echo "Step 4: Regenerating dataset with local frames..."
python mouse_facial_expressions/data/make_datasets.py task1 \
  --version "1.1" \
  --frameset_size 1

echo ""
echo "✓ Dataset generated!"
echo ""

# Step 5: Ready to train
echo "========================================="
echo "Setup Complete! Ready to train."
echo "========================================="
echo ""
echo "To start training, run:"
echo ""
echo "  python mouse_facial_expressions/models/train_task1_baseline_model.py \\"
echo "    --epochs 10 \\"
echo "    --dataset_version \"1.1\""
echo ""
echo "Training will save checkpoints to: models/checkpoints/"
echo "Estimated time on CPU: ~7 hours for 10 epochs"
echo ""
echo "To monitor progress:"
echo "  - Watch terminal output for accuracy/loss"
echo "  - Check models/checkpoints/ for saved models"
echo ""
