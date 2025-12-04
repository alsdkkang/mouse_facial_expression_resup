import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

# Add project root to path
sys.path.append(os.path.abspath('.'))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from mouse_facial_expressions.models.train_task1_resup_model import ReSupModel
from mouse_facial_expressions.data.temporal_datasets import Task1TemporalFolds

def evaluate_fold0():
    print("=" * 80)
    print("Evaluating ReSupModel on Fold 0 Test Set")
    print("=" * 80)

    # Configuration (Must match training)
    config = {
        'dataset_version': '1.1',
        'num_frames': 10,
        'frame_stride': 2,
        'train_augmentation': 'TrivialAugmentWide',
        'use_soft_labels': True,
        'test_batch_size': 32,
        # Model params (needed for init)
        'dropout': 0.5,
        'label_smoothing': 0.2
    }
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Find best checkpoint for Fold 0
    checkpoint_dir = "models/checkpoints_resup"
    # Pattern: resup-fold0-epoch=*-val_acc=*.ckpt
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "resup-fold0-*.ckpt"))
    
    if not checkpoints:
        print(f"No checkpoints found for Fold 0 in {checkpoint_dir}")
        return

    # Sort by validation accuracy (extracted from filename) or modification time
    # Filename format: resup-fold0-epoch={epoch}-val_acc={val_acc:.2f}.ckpt
    # We can try to parse val_acc, or just take the latest file if we trust the callback (save_top_k=1)
    # The callback saves the best model, so if there are multiple, the latest might be the best or just the last saved.
    # Actually, ModelCheckpoint with save_top_k=1 usually keeps only one file per monitor unless filename differs.
    # Let's pick the one with highest val_acc in filename.
    
    def get_val_acc(path):
        try:
            # Extract val_acc from string like "...val_acc=0.95.ckpt"
            part = path.split("val_acc=")[-1]
            acc = float(part.replace(".ckpt", ""))
            return acc
        except:
            return -1.0

    best_ckpt = max(checkpoints, key=get_val_acc)
    print(f"Loading best checkpoint: {best_ckpt}")

    # 2. Load Model
    model = ReSupModel.load_from_checkpoint(best_ckpt, config=config)
    model.to(device)
    model.eval()

    # 3. Prepare Data (Fold 0 Test Set)
    print("Preparing Fold 0 Test Dataset...")
    cv = Task1TemporalFolds(
        version=config['dataset_version'],
        num_frames=config['num_frames'],
        frame_stride=config['frame_stride'],
        train_augmentation=config['train_augmentation'],
        use_soft_labels=config['use_soft_labels']
    )
    
    # Get Fold 0
    # cv yields (train_dataset, test_dataset)
    # We just need the first yield (Fold 0)
    for fold, (train_dataset, test_dataset) in enumerate(cv):
        if fold == 0:
            break
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['test_batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    
    # 4. Inference
    print("Starting inference...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x, y = batch
            x = x.to(device)
            
            # Forward pass (ReSupModel forward returns average of both nets)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Handle targets (soft or hard)
            if y.dim() == 2:
                targets = torch.argmax(y, dim=1)
            else:
                targets = y
                
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 5. Metrics
    accuracy = accuracy_score(all_targets, all_preds)
    print("\n" + "="*40)
    print(f"Evaluation Results (Fold 0)")
    print("="*40)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=['Class 0', 'Class 1']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    print("="*40)

if __name__ == "__main__":
    evaluate_fold0()
