import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import glob
import sys

# Add project root to path
sys.path.append(os.path.abspath('.'))

from mouse_facial_expressions.models.train_task1_resup_model import ReSupModel
from mouse_facial_expressions.data.dataset_chronic import ChronicStressDataset

def train_chronic_finetune(fold=0):
    print("=" * 80)
    print(f"Fine-Tuning ReSup Model on Chronic Stress Data (Fold {fold})")
    print("=" * 80)
    
    # Configuration
    # Auto-detect data path
    possible_paths = [
        "data/chronic_stress_frames",
        "chronic_stress_frames",
        "/content/data/chronic_stress_frames",
        "/content/chronic_stress_frames"
    ]
    DATA_DIR = None
    for p in possible_paths:
        if os.path.exists(p):
            DATA_DIR = p
            print(f"Found data directory at: {DATA_DIR}")
            break
            
    if DATA_DIR is None:
        print("ERROR: Could not find data directory. Checked:", possible_paths)
        print("Current directory:", os.getcwd())
        print("Directory contents:", os.listdir('.'))
        return

    METADATA_FILE = "chronic_stress_metadata.csv"
    CHECKPOINT_DIR = "models/checkpoints_resup" # Source of pre-trained model
    OUTPUT_DIR = "models/checkpoints_chronic_finetune"
    
    BATCH_SIZE = 8
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-5 # Low LR for fine-tuning
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), # Slightly smaller for speed/memory? Or keep 300? Let's stick to 224 or 256 standard. Original was 300?
        # Let's check original transform. It was 300.
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = ChronicStressDataset(
        DATA_DIR, METADATA_FILE, 
        transform=train_transform, 
        fold=fold, is_train=True
    )
    
    val_dataset = ChronicStressDataset(
        DATA_DIR, METADATA_FILE, 
        transform=val_transform, 
        fold=fold, is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Load Pre-trained Model
    # Search for checkpoints in multiple locations
    possible_ckpt_dirs = [
        "models/checkpoints_resup",
        ".",
        "/content",
        "/content/drive/MyDrive/Mouse_Project",
        "checkpoints"
    ]
    
    checkpoints = []
    for d in possible_ckpt_dirs:
        if os.path.exists(d):
            found = glob.glob(os.path.join(d, "*.ckpt"))
            if found:
                print(f"Found {len(found)} checkpoints in {d}")
                checkpoints.extend(found)
                
    if not checkpoints:
        print("Error: No pre-trained checkpoints found. Checked:", possible_ckpt_dirs)
        print("Please upload .ckpt files or resup_result.zip")
        return
        
    # Helper to parse accuracy
    def get_val_acc(path):
        try:
            part = path.split("val_acc=")[-1]
            acc = float(part.replace(".ckpt", ""))
            return acc
        except:
            return -1.0
            
    best_ckpt = max(checkpoints, key=get_val_acc)
    print(f"Initializing from: {best_ckpt}")
    
    # Load model
    # We need to override the config to use hard labels for fine-tuning (since we have binary labels)
    # Or we can keep soft labels if we want, but our dataset returns hard labels (0/1).
    # The ReSupModel handles this: if target is 1D (batch,), it uses CrossEntropy.
    
    config = {
        'learning_rate': LEARNING_RATE,
        'weight_decay': 1e-4,
        'label_smoothing': 0.0, # No smoothing for fine-tuning with hard labels
        'use_soft_labels': False, # We have hard labels
        'num_frames': 10,
        'frame_stride': 2,
        'dropout': 0.5
    }
    
    model = ReSupModel.load_from_checkpoint(best_ckpt, config=config, strict=False)
    
    # Update config in model instance
    model.config = config
    model.use_soft_labels = False
    
    # Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename=f'chronic-finetune-fold{fold}-{{epoch}}-{{val_acc:.2f}}',
        save_top_k=1,
        monitor='val_acc',
        mode='max'
    )
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        logger=True,
        log_every_n_steps=10
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training complete for Fold {fold}")

if __name__ == "__main__":
    train_chronic_finetune(fold=0)
