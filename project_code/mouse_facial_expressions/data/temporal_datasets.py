"""
Temporal Dataset for Advanced Mouse Facial Expression Recognition
Supports multi-frame sequences for temporal aggregation
"""

import torch
import numpy as np
import pandas as pd
import pickle
import re
import torchvision
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.io import imread

# Import path helpers
from mouse_facial_expressions.paths import get_task_folder, get_extracted_frames_folder


class TemporalMouseDataset(Dataset):
    """
    Dataset that returns sequences of frames instead of single frames
    for temporal aggregation models
    """
    
    def __init__(self, samples, df, num_frames=10, frame_stride=2, 
                 transform=None, use_soft_labels=True, label_smoothing=0.1):
        """
        Args:
            samples: List of sample dicts from fold pickle [{'indices': [], 'label': ...}]
            df: DataFrame containing image paths and metadata
            num_frames: Number of frames to sample per sequence
            frame_stride: Stride between sampled frames
            transform: Image transformations
            use_soft_labels: Whether to use soft labels (LDL)
            label_smoothing: Smoothing factor for soft labels
        """
        self.samples = samples
        self.df = df
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.transform = transform
        self.use_soft_labels = use_soft_labels
        self.label_smoothing = label_smoothing
        self.frame_dir = Path(get_extracted_frames_folder())
        
        # Create sequences from samples
        self.sequences = self._create_sequences()
        
    def _create_sequences(self):
        """Create all possible sequences from samples"""
        sequences = []
        
        for sample in self.samples:
            indices = sample['indices']
            label = sample['label']
            
            # Sort indices to ensure temporal order
            indices = sorted(indices)
            
            num_available = len(indices)
            sequence_length = self.num_frames * self.frame_stride
            
            # Create overlapping sequences
            if num_available >= sequence_length:
                for start_idx in range(0, num_available - sequence_length + 1, self.frame_stride):
                    # Get the subset of indices for this sequence
                    seq_indices = [indices[i] for i in range(start_idx, start_idx + sequence_length, self.frame_stride)]
                    # Take only num_frames
                    seq_indices = seq_indices[:self.num_frames]
                    
                    sequences.append({
                        'indices': seq_indices,
                        'label': label
                    })
            else:
                # If video is too short, repeat frames
                if num_available > 0:
                    # Simple resampling: linspace
                    resampled_idx_positions = np.linspace(0, num_available-1, self.num_frames, dtype=int)
                    seq_indices = [indices[i] for i in resampled_idx_positions]
                    
                    sequences.append({
                        'indices': seq_indices,
                        'label': label
                    })
        
        return sequences
    
    def _create_soft_label(self, hard_label):
        """Create soft label distribution for LDL"""
        if not self.use_soft_labels:
            return hard_label
        
        num_classes = 2
        soft_label = torch.zeros(num_classes)
        soft_label[int(hard_label)] = 1.0 - self.label_smoothing
        soft_label[1 - int(hard_label)] = self.label_smoothing
        
        return soft_label
    
    def get_image(self, imagepath):
        # Use skimage.io.imread as in original dataset, then convert to PIL for transforms
        img_array = imread(self.frame_dir / imagepath)
        return Image.fromarray(img_array)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
        indices = sequence_info['indices']
        label_val = sequence_info['label']
        
        # Load frames
        frames = []
        
        # Get image paths from df
        image_paths = self.df.loc[indices].image.tolist()
        
        for img_path in image_paths:
            image = self.get_image(img_path)
            
            if self.transform:
                image = self.transform(image)
            
            frames.append(image)
        
        # Stack frames: (num_frames, C, H, W)
        frames = torch.stack(frames)
        
        # Create soft label if needed
        if self.use_soft_labels:
            label = self._create_soft_label(label_val)
        else:
            label = int(label_val)
        
        return frames, label


class Task1TemporalFolds:
    """
    Cross-validation folds for Task 1 with temporal sequences
    """
    
    def __init__(self, version='1.0', num_frames=10, frame_stride=2,
                 train_augmentation='TrivialAugmentWide', use_soft_labels=True):
        self.version = version
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.use_soft_labels = use_soft_labels
        
        # Define transforms (using torchvision directly as in original)
        self.train_transform = self._get_train_transform(train_augmentation)
        self.test_transform = self._get_test_transform()
        
        # Load data using existing structure
        task1_path = get_task_folder(version)
        self.df = pd.read_pickle(task1_path / 'dataset_df.pkl')

        folds = task1_path.glob('fold*.pkl')
        fold_df = pd.DataFrame({'foldpath': folds})
        # Extract fold index
        fold_df['fold_index'] = fold_df.foldpath.apply(lambda x: int(re.match('.*(\d+)', x.parts[-1]).group(1)))
        fold_df = fold_df.sort_values('fold_index')
        fold_df = fold_df.set_index('fold_index')
        self.fold_df = fold_df
    
    def _get_train_transform(self, augmentation_type):
        """Get training transforms with augmentation"""
        if augmentation_type == 'TrivialAugmentWide':
            return transforms.Compose([
                transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _get_test_transform(self):
        """Get test transforms (no augmentation)"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
            ])
    
    def __iter__(self):
        for idx in range(len(self.fold_df)):
            row = self.fold_df.iloc[idx]
            with open(row.foldpath, 'rb') as fp:
                fold_data = pickle.load(fp)
                
            train_samples = fold_data['train']
            test_samples = fold_data['test']
            
            train_dataset = TemporalMouseDataset(
                samples=train_samples,
                df=self.df,
                num_frames=self.num_frames,
                frame_stride=self.frame_stride,
                transform=self.train_transform,
                use_soft_labels=self.use_soft_labels
            )
            
            test_dataset = TemporalMouseDataset(
                samples=test_samples,
                df=self.df,
                num_frames=self.num_frames,
                frame_stride=self.frame_stride,
                transform=self.test_transform,
                use_soft_labels=self.use_soft_labels
            )
            
            yield train_dataset, test_dataset
    
    def __len__(self):
        return len(self.fold_df)
