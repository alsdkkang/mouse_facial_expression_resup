import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys
import glob

# Add project root to path
sys.path.append(os.path.abspath('.'))

from mouse_facial_expressions.models.train_task1_resup_model import ReSupModel
from mouse_facial_expressions.data.dataset_chronic import ChronicStressDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_chronic_finetune(fold=0):
    print("=" * 80)
    print(f"Evaluating Fine-Tuned Model (Fold {fold})")
    print("=" * 80)
    
    DATA_DIR = "data/chronic_stress_frames"
    METADATA_FILE = "chronic_stress_metadata.csv"
    CHECKPOINT_DIR = "models/checkpoints_chronic_finetune"
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transform (same as validation)
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation Dataset
    val_dataset = ChronicStressDataset(
        DATA_DIR, METADATA_FILE, 
        transform=transform, 
        fold=fold, is_train=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Load best checkpoint
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, f"chronic-finetune-fold{fold}-*.ckpt"))
    if not checkpoints:
        print("No checkpoints found.")
        return
        
    def get_val_acc(path):
        try:
            part = path.split("val_acc=")[-1]
            acc = float(part.replace(".ckpt", ""))
            return acc
        except:
            return -1.0
            
    best_ckpt = max(checkpoints, key=get_val_acc)
    print(f"Loading checkpoint: {best_ckpt}")
    
    # Load Model
    # Config must match training
    config = {
        'num_frames': 10,
        'frame_stride': 2,
        'dropout': 0.5,
        'use_soft_labels': False
    }
    
    model = ReSupModel.load_from_checkpoint(best_ckpt, config=config, strict=False)
    model.to(device)
    model.eval()
    
    # Inference
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running inference...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # Probability of Stress
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print("\n" + "="*40)
    print(f"Results for Fold {fold}")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 40)
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    # Save results
    df = pd.DataFrame({
        'True_Label': all_labels,
        'Predicted_Label': all_preds,
        'Prob_Stress': all_probs
    })
    df.to_csv(f"chronic_finetune_fold{fold}_results.csv", index=False)
    print(f"Saved results to chronic_finetune_fold{fold}_results.csv")

if __name__ == "__main__":
    evaluate_chronic_finetune(fold=0)
