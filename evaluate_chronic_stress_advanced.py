import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import re
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.path.abspath('.'))

from mouse_facial_expressions.models.train_task1_advanced_model import AdvancedDeepSet

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
            # Sort by frame number (assuming filename contains frame number or is sortable)
            # Usually frame extraction produces frame_0001.png etc.
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
                # Return zeros if error (should be handled better in prod)
                images.append(torch.zeros(3, 300, 300))
                
        # Stack: (num_frames, C, H, W)
        images = torch.stack(images)
        
        return images, seq_info['video_name'], seq_info['center_frame']

def evaluate():
    # Configuration
    DATA_DIR = "data/chronic_stress_frames"
    CHECKPOINT_DIR = "models/checkpoints_advanced"
    
    # Find latest checkpoint
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.ckpt"))
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    # Sort by modification time (latest first)
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    print(f"Loading checkpoint: {latest_ckpt}")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    # We need to reconstruct the config used for training
    # Ideally load from hparams.yaml if available, but for now hardcode matching defaults
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'use_soft_labels': True,
        'num_frames': 10,
        'frame_stride': 2
    }
    
    model = AdvancedDeepSet.load_from_checkpoint(latest_ckpt, config=config)
    model.to(device)
    model.eval()
    
    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ChronicStressTemporalDataset(
        DATA_DIR, 
        num_frames=config['num_frames'], 
        frame_stride=config['frame_stride'], 
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Inference
    print("Starting inference...")
    results = []
    
    with torch.no_grad():
        for batch_images, video_names, center_frames in tqdm(dataloader):
            batch_images = batch_images.to(device)
            
            # Forward pass
            outputs = model(batch_images)
            
            # Get probabilities (Softmax for soft labels/LDL output)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            for i in range(len(probs)):
                results.append({
                    'video_name': video_names[i],
                    'center_frame': center_frames[i],
                    'prob_non_stress': probs[i][0],
                    'prob_stress': probs[i][1],
                    'predicted_label': np.argmax(probs[i])
                })
                
    # Save results
    df = pd.DataFrame(results)
    df.to_csv("chronic_stress_advanced_predictions.csv", index=False)
    print("Saved frame-level predictions to chronic_stress_advanced_predictions.csv")
    
    # Video-level summary
    video_summary = df.groupby('video_name')[['prob_stress']].mean()
    video_summary['predicted_class'] = (video_summary['prob_stress'] > 0.5).astype(int)
    
    print("\nVideo-level Summary (Average Stress Probability):")
    print(video_summary)
    
    video_summary.to_csv("chronic_stress_advanced_video_summary.csv")
    print("Saved video-level summary to chronic_stress_advanced_video_summary.csv")

if __name__ == "__main__":
    evaluate()
