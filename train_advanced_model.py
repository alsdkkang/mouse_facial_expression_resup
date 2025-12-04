#!/usr/bin/env python
"""
Quick start script for training the advanced model
"""

import subprocess
import sys

def main():
    print("=" * 80)
    print("Advanced Mouse Facial Expression Recognition Model")
    print("Training with LDL + Temporal Aggregation")
    print("=" * 80)
    print()
    
    # Default parameters
    cmd = [
        sys.executable, "-m",
        "mouse_facial_expressions.models.train_task1_advanced_model",
        "--epochs", "20",
        "--learning_rate", "0.0001",
        "--num_frames", "10",
        "--frame_stride", "2",
        "--train_batch_size", "16",
        "--test_batch_size", "32",
        "--use_soft_labels", "True",
        "--label_smoothing", "0.1",
        "--dataset_version", "1.1",
        "--train_augmentation", "TrivialAugmentWide",
        "--seed", "97531"
    ]
    
    # Add any additional arguments passed to this script
    cmd.extend(sys.argv[1:])
    
    print("Training command:")
    print(" ".join(cmd))
    print()
    print("Starting training...")
    print("=" * 80)
    print()
    
    # Run training
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
