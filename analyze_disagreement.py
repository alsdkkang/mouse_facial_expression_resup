import os
import glob
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from mouse_facial_expressions.models.train_task1_resup_model import ReSupModel

def analyze_disagreement():
    print("=" * 80)
    print("Analyzing Model Disagreement (ReSup Ensemble)")
    print("=" * 80)

    # Configuration
    DATA_DIR = "data/chronic_stress_frames"
    CHECKPOINT_DIR = "models/checkpoints_resup"
    OUTPUT_DIR = "analysis_results/disagreement_analysis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    folds = [0, 1, 2, 3, 4]
    models = []
    config = {
        'learning_rate': 1e-4, 'weight_decay': 1e-4, 'label_smoothing': 0.2,
        'use_soft_labels': True, 'num_frames': 10, 'frame_stride': 2, 'dropout': 0.5
    }

    print("Loading models...")
    for fold in folds:
        checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, f"resup-fold{fold}-*.ckpt"))
        if not checkpoints: continue
        
        # Helper to parse accuracy
        def get_val_acc(path):
            try: return float(path.split("val_acc=")[-1].replace(".ckpt", ""))
            except: return -1.0
            
        best_ckpt = max(checkpoints, key=get_val_acc)
        model = ReSupModel.load_from_checkpoint(best_ckpt, config=config)
        model.to(device)
        model.eval()
        models.append(model)
    
    print(f"Loaded {len(models)} models.")

    # Select a few videos to analyze (e.g., first 3)
    video_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "*")))[:3]
    
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Store frames with scores
    high_disagreement_frames = []
    low_disagreement_frames = []

    for video_dir in video_dirs:
        video_name = os.path.basename(video_dir)
        print(f"Processing {video_name}...")
        
        image_paths = sorted(glob.glob(os.path.join(video_dir, "*.png")))
        # Sample every 10th frame to save time
        image_paths = image_paths[::10]
        
        for img_path in tqdm(image_paths):
            try:
                # Prepare input (single frame repeated to match model input dim if needed, 
                # but model expects sequence. We will just duplicate the frame 10 times 
                # to analyze "static" frame disagreement)
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0) # (1, C, H, W)
                
                # Create a "static sequence" of 10 identical frames
                input_seq = img_tensor.repeat(1, 10, 1, 1, 1).to(device) # (1, 10, C, H, W)
                
                # Inference
                probs_list = []
                with torch.no_grad():
                    for model in models:
                        out = model(input_seq)
                        prob = torch.softmax(out, dim=1)[0, 1].item() # Prob of Stress
                        probs_list.append(prob)
                
                # Calculate metrics
                mean_prob = np.mean(probs_list)
                std_dev = np.std(probs_list) # Disagreement
                
                frame_info = {
                    'path': img_path,
                    'mean_prob': mean_prob,
                    'std_dev': std_dev,
                    'probs': probs_list
                }
                
                high_disagreement_frames.append(frame_info)
                low_disagreement_frames.append(frame_info)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # Sort and save top 5
    high_disagreement_frames.sort(key=lambda x: x['std_dev'], reverse=True)
    low_disagreement_frames.sort(key=lambda x: x['std_dev']) # Lowest std first
    
    def save_examples(frames, prefix):
        for i, item in enumerate(frames[:5]):
            img = cv2.imread(item['path'])
            
            # Add text
            text = f"Std: {item['std_dev']:.3f}, Mean: {item['mean_prob']:.2f}"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            out_path = os.path.join(OUTPUT_DIR, f"{prefix}_{i}_{os.path.basename(item['path'])}")
            cv2.imwrite(out_path, img)
            print(f"Saved {out_path}")

    print("\nSaving High Disagreement Frames...")
    save_examples(high_disagreement_frames, "high_disagreement")
    
    print("\nSaving Low Disagreement Frames...")
    save_examples(low_disagreement_frames, "low_disagreement")

if __name__ == "__main__":
    analyze_disagreement()
