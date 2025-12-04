# Draft Email to Team

**Subject:** Re: Chronic dataset group info / Update on Analysis Plan

Hi everyone,

Thank you all for the valuable input!

**To answer Alfonso and Brenna:**
Yes, the model used for the initial evaluation was trained on the **Acute Stress dataset (LPS vs. Saline)** from the previous study (Task 1). My hypothesis was to see if facial features learned from Acute Stress (LPS) would transfer to the Chronic Stress context.

**Update on Analysis & Findings:**

**1. Metadata Integration (Thanks Brenna!)**
Using the group info you provided:
*   **Excluded Animals**: We confirmed that the "Grey" (excluded) animals are not present in our video dataset.
*   **Het (Yellow)**: We have excluded "Het" animals from our analysis to maintain a clear Control vs. Stress comparison.
*   **KO Stress**: Based on Brenna’s insight ("likely more stressed"), we are labeling **KO Stress as 'Stress' (Class 1)** for our evaluation.

**2. Zero-Shot Results (Acute Model -> Chronic Data)**
When applying the Acute-trained model directly to the Chronic data, we observed a **strong batch effect in Cohort 1**.
*   The model predicted **100% of Cohort 1 animals as "Stressed"**, regardless of whether they were Control or Stress.
*   This suggests that the model is picking up on specific recording conditions (e.g., lighting, angle) of Cohort 1 that resemble the Acute dataset, rather than the actual stress expression.

**3. Next Steps: Fine-Tuning**
Since the "Zero-Shot" transfer didn't work perfectly, we are now proceeding with **Fine-Tuning**:
*   We are training the model specifically on the **Chronic Stress dataset** (using 5-Fold Cross-Validation to ensure robust testing).
*   This will allow the model to learn the specific differences between Control and Stress *within* the chronic experiment environment.

I am currently running this fine-tuning process and will share the updated performance metrics (including WT vs. KO breakdown) soon.

Best regards,
Mina
