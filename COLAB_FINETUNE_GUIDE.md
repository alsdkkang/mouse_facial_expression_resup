# Colab Fine-Tuning Guide

## 1. GPU Selection
Based on your available hardware, select **A100 GPU** for the fastest training.

**Ranking (Speed):**
1.  **A100 GPU**: Fastest (Recommended). Use this if available.
2.  **L4 GPU**: Very good performance, significantly faster than T4. Good alternative.
3.  **T4 GPU**: Standard option, slowest among the GPUs but sufficient if others are unavailable.

*Note: TPUs (v6e, v5e) are also powerful but require code changes. Stick to GPU for this script.*

## 2. Files to Upload
You need to upload the following files to the Colab environment:
1.  `project_code_finetune.zip` (I will create this for you)
2.  `data.zip` (Contains `data/chronic_stress_frames`)
3.  `models/checkpoints_resup/*.ckpt` (You need at least one pre-trained checkpoint in the correct folder structure, or upload the `resup_result.zip` and unzip it)

## 3. Steps
1.  Open `Colab_FineTuning.ipynb` in Google Colab.
2.  Upload the zip files.
3.  Run the cells in order.
4.  The script `train_chronic_finetune.py` will automatically find the best pre-trained checkpoint and start fine-tuning.
5.  After training, `evaluate_chronic_finetune.py` will run and save the results.
6.  Download `chronic_finetune_results.csv` and the new model checkpoints.
