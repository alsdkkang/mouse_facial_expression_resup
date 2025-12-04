# Draft Response to Andre

**Subject:** Re: Update on Mouse Facial Expression Model (ReSup)

Hi Andre,

Thanks for the great questions and feedback! Here are the details on our current approach and findings:

### 1. Alignment with Labels (LPS & Chronic Stress)
*   **LPS (Acute Pain)**: The model aligns perfectly with the high-dose LPS labels in the training set (Task 1), achieving ~99% accuracy in validation (Fold 0).
*   **Chronic Stress**: This is the most exciting part. While your previous model (baseline) predicted ~0.11 score for all chronic stress mice (0% detection), the new **ReSup model (5-Fold Ensemble)** detects stress in **22.0% (9/41)** of the chronic stress group with high confidence (scores > 0.6).
    *   **Cohort Alignment**: Interestingly, **all 9 detected mice belong to Cohort 1**. This strongly suggests that Cohort 1 might have experienced a different level of stress or expressed it more overtly compared to other cohorts. This directly answers your question about alignment with other labels/groups.

### 2. Verification, Comparison, and Signal Amplification
*   **Verification**: We have completed the full **5-Fold Cross-Validation** (Subject-Independent). The results are consistent across folds, confirming robustness.
*   **Signal Amplification (ReSup)**: We implemented a **"Reliable Noise Suppression (ReSup)"** mechanism.
    *   Instead of standard signal amplification, we use a **Dual-Network** architecture. Two networks train simultaneously and "cross-check" each other.
    *   If they **disagree** on an image, we treat it as "noisy" and down-weight it. This naturally filters out ambiguous frames and forces the model to learn only from the "cleanest" signals of distress, effectively amplifying the true signal.

### 3. DeepSet Approach?
*   **Yes**, we are still using the **DeepSet-like approach** (Temporal Aggregation).
*   Specifically, we input **10 frames** per sequence.
*   We use a **ResNet34** spatial encoder followed by a **Transformer** temporal encoder to aggregate features across these 10 frames before making a final prediction.

### 4. Addressing Your Comments
*   **Agreement/Disagreement**: We performed a detailed analysis on this.
    *   **High Disagreement**: Occurs mostly on frames with **motion blur** or **grooming behavior** (where paws occlude the face). This confirms the ReSup mechanism is correctly identifying and down-weighting "noisy" or ambiguous inputs.
    *   **High Agreement**: Occurs on clear, frontal face frames. Notably, we found high agreement on the **Cohort 1** mice (the ones identified as stressed), suggesting their facial features were distinct and unambiguous to both networks.
    *   I have attached example frames showing high vs. low disagreement for your reference.
*   **Calibration**: Noted. We are using **Label Distribution Learning (LDL)** with soft labels (e.g., target 0.8 instead of 1.0), so the outputs are indeed "scores" representing the intensity/confidence rather than strict probabilities. We will be careful with the terminology.

Best,

Mina
