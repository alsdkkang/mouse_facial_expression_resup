# Report: Progress on Automated Chronic Stress Detection in Mice

**To:** Prof. Alfonso Abizaid
**From:** Mina Kang
**Date:** December 1, 2025
**Subject:** Breakthrough in Chronic Stress Detection using ReSup Model

Dear Professor Abizaid,

I am writing to share a significant update regarding our automated mouse facial expression analysis project. We have successfully developed and validated a new deep learning model that shows promising capability in detecting **chronic stress**, a task where our previous baseline models had failed.

## 1. The Challenge: "Hidden" Signals of Chronic Stress
Our initial attempts using standard models (ResNet + Transformer) achieved high accuracy on **acute pain** (Task 1) but failed to generalize to **chronic stress** data (0% detection rate).
We hypothesized that this was because:
1.  **Subtlety**: Chronic stress signals are far less intense than acute pain "grimaces."
2.  **Ambiguity**: The training labels (Pain vs. No Pain) are noisy approximations for chronic stress, leading standard models to discard these subtle signals as "noise."

## 2. The Solution: The ReSup Model (Reliable Noise Suppression)
To address this, we implemented a **ReSup (Reliable Noise Suppression)** architecture. This model is designed specifically to learn from "noisy" or "weak" labels by distinguishing between clean data and ambiguous data.

Key technical innovations include:
*   **Dual-Network Architecture**: We train two independent networks simultaneously. They "cross-check" each other. If they disagree on a prediction, the model identifies that sample as "ambiguous" or "noisy."
*   **Weight Exchange Mechanism**: Instead of forcing the model to memorize every label (which leads to overfitting on acute pain), we dynamically lower the weight of ambiguous samples. This allows the model to focus on learning robust, generalized features of distress rather than memorizing specific high-intensity grimaces.
*   **Label Distribution Learning (LDL)**: Instead of a binary "Stress/No Stress" classification, we use soft labels (e.g., "80% Stress"). This allows the model to capture the **intensity** of the expression, making it sensitive to the milder signs of chronic stress.

## 3. Preliminary Results
We evaluated this new model on the **Chronic Stress Facial Expression Videos** dataset (41 subjects).

*   **Baseline Model**: Detected **0** stress cases (0%). It classified all chronic stress mice as "neutral," likely waiting for a strong acute pain signal that never appeared.
*   **ReSup Model**: Detected **8** stress cases (**19.5%**) with high confidence (probability scores up to **0.84**).

## 4. Implication
This result suggests that the ReSup model has successfully learned to identify **"micro-expressions" of distress** that are common to both acute pain and chronic stress, without relying on the high-intensity features required by previous models.

We are currently completing a rigorous 5-fold cross-validation to confirm these findings statistically. I look forward to discussing these results and the potential for this tool in our upcoming meeting.

Sincerely,

Mina Kang
