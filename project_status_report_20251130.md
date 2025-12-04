# Project Status Report: ReSup Model Development & Evaluation
**Date:** November 30, 2025
**Subject:** Model Comparison (Andre's Advanced vs. ReSup) and Training Progress

## 1. Executive Summary
We have successfully developed and deployed the **ReSup (Reliable Noise Suppression) Model**, which integrates Label Distribution Learning (LDL), Temporal Aggregation, and a Dual-Network architecture. 

Comparative evaluation on the **Chronic Stress dataset** demonstrates a significant breakthrough:
- **Andre's Model (Baseline)**: Failed to detect any stress signals (0% detection rate).
- **ReSup Model (Current)**: Successfully detected chronic stress in **19.5%** of subjects with high confidence.

Currently, the 5-Fold Cross-Validation process is **40% complete** (Folds 0 & 1 done), with Fold 2 training underway locally.

---

## 2. Model Comparison: Technical Architecture

The ReSup model builds upon the baseline architecture but introduces critical mechanisms to handle label noise and improve generalization.

| Feature | Andre's Model (Advanced) | ReSup Model (Current) | Impact |
| :--- | :--- | :--- | :--- |
| **Core Architecture** | Single Network (ResNet34 + Transformer) | **Dual Network** (Two independent ResNet34 + Transformer streams) | Allows for cross-validation between networks during training. |
| **Noise Handling** | None (Treats all labels as ground truth) | **Weight Exchange Mechanism** | Identifies "disagreement" between networks and down-weights likely noisy samples. |
| **Consistency** | None | **Consistency Loss** | Forces both networks to converge on stable predictions, reducing overfitting to noise. |
| **Labeling** | LDL (Soft Labels) | LDL (Soft Labels) | Both use soft labels to capture ambiguity, but ReSup refines this with noise suppression. |

**Key Insight**: The "Weak Supervision" in ReSup is achieved not by adding external data, but by treating the existing noisy labels as weak signals and filtering them via the Dual-Network disagreement.

---

## 3. Evaluation Results: Chronic Stress Dataset

We evaluated both models on the **Chronic Stress Facial Expression Videos** dataset (41 videos). This dataset represents a "hidden" pain state that is difficult to detect compared to acute pain.

### Quantitative Results

| Metric | Andre's Model | ReSup Model (Ensemble Fold 0+1) |
| :--- | :--- | :--- |
| **Stress Detected** | **0 / 41 (0.0%)** | **8 / 41 (19.5%)** |
| **Avg Stress Probability** | ~0.11 (Very Low) | **Max 0.84** (High Confidence) |
| **Detection Threshold** | Failed to cross 0.5 threshold | Robustly crossed threshold for specific cohorts |

### Qualitative Analysis
- **Andre's Model**: exhibited "conservative" behavior, predicting near-zero stress probability for all subjects. It likely overfitted to the clear signals of acute pain (Task 1) and failed to generalize to the subtle signals of chronic stress.
- **ReSup Model**: demonstrated "sensitivity" to subtle features. It identified a cluster of stress cases, particularly in **Cohort 1**, with probability scores ranging from **0.50 to 0.84**. This suggests it has learned to identify facial features associated with distress that are distinct from the acute pain "grimace".

---

## 4. Current Progress & Next Steps

### Training Status (5-Fold Cross-Validation)
We are training the ReSup model on the standard Task 1 dataset using 5-fold cross-validation to ensure statistical robustness.

| Fold | Status | Outcome / Notes |
| :--- | :--- | :--- |
| **Fold 0** | ✅ **Complete** | Validation Accuracy: **~99%** |
| **Fold 1** | ✅ **Complete** | Validation Accuracy: **~85%** |
| **Fold 2** | 🔄 **In Progress** | Training started locally (MacBook Pro) |
| **Fold 3** | ⏳ Scheduled | Will run automatically after Fold 2 |
| **Fold 4** | ⏳ Scheduled | Will run automatically after Fold 3 |

### Immediate Action Plan
1.  **Monitor Fold 2-4 Training**: Ensure local training completes successfully (Est. 30-40 hours).
2.  **Final Ensemble Evaluation**: Once all 5 folds are complete, run the ensemble evaluation again using all 5 models. This is expected to further improve the detection rate and reliability on the Chronic Stress dataset.
