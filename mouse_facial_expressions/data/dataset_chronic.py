import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import re

class ChronicStressDataset(Dataset):
    """
    Dataset for Chronic Stress data with Metadata Labels.
    Excludes 'Het' animals by default.
    """
    def __init__(self, root_dir, metadata_file, num_frames=10, frame_stride=2, transform=None, fold=0, num_folds=5, is_train=True, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        
        # Load Metadata
        self.meta_df = pd.read_csv(metadata_file)
        
        # Filter out 'Het'
        self.meta_df = self.meta_df[self.meta_df['Condition'] != 'Het'].copy()
        
        # Create AnimalID to Label mapping
        # Stress = 1, Control = 0
        self.meta_df['Label'] = self.meta_df['Condition'].apply(lambda x: 1 if x.lower() == 'stress' else 0)
        self.id_to_label = dict(zip(self.meta_df['AnimalID'], self.meta_df['Label']))
        self.valid_ids = set(self.meta_df['AnimalID'])
        
        # Find all videos
        # Assuming structure: root_dir/video_name/*.png
        self.video_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
        self.video_dirs = [d for d in self.video_dirs if os.path.isdir(d)]
        
        print(f"DEBUG: Root dir: {root_dir}")
        print(f"DEBUG: Found {len(self.video_dirs)} subdirectories.")
        if len(self.video_dirs) > 0:
            print(f"DEBUG: First 5 dirs: {[os.path.basename(d) for d in self.video_dirs[:5]]}")
        
        # Filter videos that match valid AnimalIDs
        self.valid_videos = []
        for v_dir in self.video_dirs:
            v_name = os.path.basename(v_dir)
            animal_id = self._extract_animal_id(v_name)
            
            if animal_id is None:
                # print(f"DEBUG: Could not extract ID from {v_name}") # Uncomment if needed
                continue
                
            if animal_id in self.valid_ids:
                self.valid_videos.append({
                    'path': v_dir,
                    'name': v_name,
                    'animal_id': animal_id,
                    'label': self.id_to_label[animal_id]
                })
            else:
                # print(f"DEBUG: ID {animal_id} from {v_name} not in metadata (or is Het)")
                pass
        
        print(f"Found {len(self.valid_videos)} valid videos matching metadata (excluding Het).")
        
        # Split by Animal ID (to avoid leakage)
        unique_animals = sorted(list(set(v['animal_id'] for v in self.valid_videos)))
        np.random.seed(seed)
        np.random.shuffle(unique_animals)
        
        # Simple K-Fold split on animals
        fold_size = len(unique_animals) // num_folds
        splits = []
        for i in range(num_folds):
            start = i * fold_size
            end = (i + 1) * fold_size if i < num_folds - 1 else len(unique_animals)
            test_animals = unique_animals[start:end]
            train_animals = [a for a in unique_animals if a not in test_animals]
            splits.append((train_animals, test_animals))
            
        train_animals, test_animals = splits[fold]
        
        target_animals = train_animals if is_train else test_animals
        self.videos_to_use = [v for v in self.valid_videos if v['animal_id'] in target_animals]
        
        print(f"Fold {fold} ({'Train' if is_train else 'Test'}): {len(self.videos_to_use)} videos (Animals: {len(target_animals)})")
        
        # Create sequences
        self.sequences = self._create_sequences()
        print(f"Created {len(self.sequences)} sequences.")

    def _extract_animal_id(self, video_name):
        # Same logic as analysis script
        match = re.search(r'_(\d+)[a-zA-Z]*_', video_name)
        if match:
            return int(match.group(1))
        match = re.search(r'Cohort\s*\d+\s*[ _]+(\d+)', video_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _create_sequences(self):
        sequences = []
        
        for video_info in self.videos_to_use:
            # Find all frames
            frames = sorted(glob.glob(os.path.join(video_info['path'], "*.png")))
            num_available = len(frames)
            sequence_length = self.num_frames * self.frame_stride
            
            if num_available >= sequence_length:
                # Create overlapping sequences
                # Stride between sequences? Let's use num_frames/2 for training to get more data
                seq_stride = self.num_frames if self.num_frames < 10 else 10 
                
                for start_idx in range(0, num_available - sequence_length + 1, seq_stride):
                    seq_frames = [frames[i] for i in range(start_idx, start_idx + sequence_length, self.frame_stride)]
                    seq_frames = seq_frames[:self.num_frames]
                    
                    sequences.append({
                        'frames': seq_frames,
                        'label': video_info['label'],
                        'video_name': video_info['name']
                    })
            else:
                 # Handle short videos: just take what we have if enough
                 if num_available >= self.num_frames:
                     seq_frames = frames[:self.num_frames]
                     sequences.append({
                        'frames': seq_frames,
                        'label': video_info['label'],
                        'video_name': video_info['name']
                    })
        
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        frame_paths = seq_info['frames']
        label = seq_info['label']
        
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
                
        images = torch.stack(images)
        return images, torch.tensor(label, dtype=torch.long)
