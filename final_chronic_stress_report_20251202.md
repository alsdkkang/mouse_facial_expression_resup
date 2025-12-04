# Final Report: Chronic Stress Detection (Zero-Shot Evaluation)
**Date:** December 3, 2025
**Author:** Mina Kang

## 1. Executive Summary
This report summarizes the evaluation of the **ReSup model (trained on Acute Stress)** when applied to the **Chronic Stress** dataset. Using the newly acquired metadata, we verified the model's zero-shot generalization capabilities.

**Key Findings:**
*   **Domain Shift**: The Acute-trained model **does not generalize** well to the Chronic dataset (Accuracy: 45%).
*   **Batch Effect**: A strong batch effect was observed in **Cohort 1**, where 100% of animals were predicted as "Stressed" regardless of their actual condition. This suggests the model is picking up on environmental features (lighting/angle) specific to that cohort.
*   **Next Step**: We are proceeding with **Fine-Tuning** on the Chronic dataset to correct this bias.

## 2. Metadata & Labeling Decisions
Following consultation with the research team (Brenna/Andre), we established the following ground truth for evaluation:
*   **Excluded Animals**: "Grey" highlighted animals in the metadata are excluded from the dataset.
*   **Het (Heterozygous)**: "Yellow" highlighted animals are excluded from the binary classification analysis to maintain a clear Control vs. Stress comparison.
*   **KO Stress**: Labeled as **Class 1 (Stress)**. While the expression might differ from WT Stress, the experimental intent is to induce stress, and we aim to detect it.

## 3. Quantitative Performance (Zero-Shot)
*Model: ReSup Ensemble (Acute Trained) applied to Chronic Data.*

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **45.00%** | Near random chance due to systematic errors. |
| **Precision** | **44.44%** | High False Positive rate (driven by Cohort 1 Controls). |
| **Recall** | **19.05%** | Failed to detect stress in Cohorts 2-5. |

### Confusion Matrix
| | Predicted Control | Predicted Stress |
| :--- | :---: | :---: |
| **Actual Control** | 14 | 5 (All from Cohort 1) |
| **Actual Stress** | 17 | 4 (All from Cohort 1) |

## 4. Cohort 1 Analysis: The "False" Signal
The model assigns high stress probabilities (>0.5) to **every single animal** in Cohort 1.
*   **Cohort 1 Prediction**: 100% Stress
*   **Actual Composition**: Mixed (Control & Stress)
*   **Implication**: The model is likely reacting to a "darker" or "different" video style in Cohort 1 that resembles the Acute Stress training data, rather than the facial expression itself.

![Stress by Condition](chronic_stress_visualizations/stress_by_condition.png)

## 5. Conclusion & Remediation Plan
The Zero-Shot approach confirms that **Acute Stress features are not directly transferable** to this Chronic Stress dataset without adaptation.

**Action Plan (In Progress):**
1.  **Fine-Tuning**: We are currently fine-tuning the ReSup model on the Chronic dataset using 5-Fold Cross-Validation (split by Animal ID).
2.  **Goal**: This will force the model to learn the specific difference between Control and Stress *within* the chronic experimental setup, effectively removing the Cohort 1 batch effect.
3.  **Expected Outcome**: Improved accuracy and a model capable of distinguishing chronic stress signals independent of the recording cohort.
