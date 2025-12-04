import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Add project root to path
sys.path.append(os.path.abspath('.'))

from mouse_facial_expressions.models.train_task1_resup_model import ReSupModel

class ChronicStressTemporalDataset(Dataset):
    """
    Dataset for Chronic Stress data that creates temporal sequences
    """
    def __init__(self, root_dir, num_frames=10, frame_stride=2, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        
        # Find all images
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True))
        
        # Group by video
        self.video_frames = {}
        for img_path in self.image_paths:
            # Assuming structure: .../video_name/frame.png
            video_name = os.path.basename(os.path.dirname(img_path))
            if video_name not in self.video_frames:
                self.video_frames[video_name] = []
            self.video_frames[video_name].append(img_path)
            
        # Sort frames within each video
        for video in self.video_frames:
            self.video_frames[video].sort()
            
        # Create sequences
        self.sequences = self._create_sequences()
        print(f"Found {len(self.video_frames)} videos, created {len(self.sequences)} sequences.")

    def _create_sequences(self):
        sequences = []
        
        for video_name, frames in self.video_frames.items():
            num_available = len(frames)
            sequence_length = self.num_frames * self.frame_stride
            
            # Create overlapping sequences
            if num_available >= sequence_length:
                for start_idx in range(0, num_available - sequence_length + 1, self.frame_stride):
                    # Get frames for this sequence
                    seq_frames = [frames[i] for i in range(start_idx, start_idx + sequence_length, self.frame_stride)]
                    seq_frames = seq_frames[:self.num_frames]
                    
                    sequences.append({
                        'video_name': video_name,
                        'frames': seq_frames,
                        'center_frame': seq_frames[len(seq_frames)//2] # Representative frame for visualization
                    })
            else:
                # Handle short videos if necessary (pad or repeat)
                # For now, skip if shorter than required sequence
                if num_available >= self.num_frames:
                     # Just take first num_frames
                     seq_frames = frames[:self.num_frames]
                     sequences.append({
                        'video_name': video_name,
                        'frames': seq_frames,
                        'center_frame': seq_frames[len(seq_frames)//2]
                    })
        
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        frame_paths = seq_info['frames']
        
        images = []
        for p in frame_paths:
            try:
                img = Image.open(p).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading {p}: {e}")
                images.append(torch.zeros(3, 300, 300))
                
        # Stack: (num_frames, C, H, W)
        images = torch.stack(images)
        
        return images, seq_info['video_name'], seq_info['center_frame']

def evaluate_chronic_stress(folds=[0, 1]):
    print("=" * 80)
    print(f"Evaluating ReSupModel Ensemble (Folds {folds}) on Chronic Stress Data")
    print("=" * 80)

    # Configuration
    DATA_DIR = "data/chronic_stress_frames"
    CHECKPOINT_DIR = "models/checkpoints_resup"
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    # Config for model (same for all folds)
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'label_smoothing': 0.2,
        'use_soft_labels': True,
        'num_frames': 10,
        'frame_stride': 2,
        'dropout': 0.5
    }

    dataset = ChronicStressTemporalDataset(
        DATA_DIR, 
        num_frames=config['num_frames'], 
        frame_stride=config['frame_stride'], 
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Store predictions from each fold
    ensemble_probs = []
    
    # Helper to parse accuracy from filename
    def get_val_acc(path):
        try:
            part = path.split("val_acc=")[-1]
            acc = float(part.replace(".ckpt", ""))
            return acc
        except:
            return -1.0

    # Load models from all 5 folds
    folds = [0, 1, 2, 3, 4]
    models = []
    
    print(f"Loading models for folds: {folds}")

    for fold in folds:
        print(f"\n--- Processing Fold {fold} ---")
        
        # Find best checkpoint for this fold
        checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, f"resup-fold{fold}-*.ckpt"))
        
        if not checkpoints:
            print(f"Warning: No checkpoints found for Fold {fold}. Skipping.")
            continue

        best_ckpt = max(checkpoints, key=get_val_acc)
        print(f"Loading best checkpoint: {best_ckpt}")
        
        # Load model
        model = ReSupModel.load_from_checkpoint(best_ckpt, config=config)
        model.to(device)
        model.eval()
        models.append(model)
        
    # Inference
    print(f"Running inference with {len(models)} models...")
    
    all_predictions = []
    
    with torch.no_grad():
        for batch_images, _, _ in tqdm(dataloader):
            batch_images = batch_images.to(device)
            
            # Average predictions across all models for this batch
            batch_probs_sum = None
            
            for model in models:
                outputs = model(batch_images)
                probs = torch.softmax(outputs, dim=1)
                
                if batch_probs_sum is None:
                    batch_probs_sum = probs
                else:
                    batch_probs_sum += probs
            
            # Average
            avg_batch_probs = batch_probs_sum / len(models)
            all_predictions.append(avg_batch_probs.cpu().numpy())
            
    # Concatenate all batches
    if not all_predictions:
        print("No predictions generated.")
        return

    avg_probs = np.concatenate(all_predictions, axis=0)
    print(f"\nGenerated predictions for {len(avg_probs)} sequences.")
    
    # Prepare results
    results = []
    
    for i, seq_info in enumerate(dataset.sequences):
        results.append({
            'video_name': seq_info['video_name'],
            'center_frame': seq_info['center_frame'],
            'prob_non_stress': avg_probs[i][0],
            'prob_stress': avg_probs[i][1],
            'predicted_label': np.argmax(avg_probs[i])
        })
                
    # Save results
    df = pd.DataFrame(results)
    output_file = "chronic_stress_resup_ensemble_predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved frame-level ensemble predictions to {output_file}")
    
    # Video-level summary
    video_summary = df.groupby('video_name')[['prob_stress']].mean()
    video_summary['predicted_class'] = (video_summary['prob_stress'] > 0.5).astype(int)
    
    print("\nVideo-level Summary (Average Stress Probability):")
    print(video_summary)
    
    summary_file = "chronic_stress_resup_ensemble_video_summary.csv"
    video_summary.to_csv(summary_file)
    print(f"Saved video-level ensemble summary to {summary_file}")

if __name__ == "__main__":
    evaluate_chronic_stress()
