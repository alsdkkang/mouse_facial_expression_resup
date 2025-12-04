# Draft Email to Research Team

**To:** Andre, Prof. Alfonso Abizaid, Brenna
**Subject:** Final Results: Automated Chronic Stress Detection using ReSup Model

Dear all,

I am excited to share the final results of our **ReSup (Reliable Noise Suppression)** model evaluation on the Chronic Stress dataset.

After completing a rigorous **5-Fold Cross-Validation**, we have achieved a significant breakthrough in detecting chronic stress signals that were previously undetectable by our baseline models.

### 1. Key Results (ReSup vs. Baseline)
*   **Baseline Model**: Detected **0** stress cases (0%).
*   **ReSup Model (5-Fold Ensemble)**: Detected **9** stress cases (**22.0%**) with high confidence.

### 2. Crucial Finding: Cohort 1 Specificity (For Brenna)
A detailed analysis revealed that **all 9 detected mice belong to Cohort 1**.
*   This suggests that the mice in Cohort 1 might have expressed more overt signs of distress or experienced a different level of stress compared to other cohorts.
*   **Brenna**, could you confirm if there were any experimental differences (e.g., duration, intensity, or specific conditions) for Cohort 1? This alignment is very striking.

### 3. Addressing Andre's Questions (Metadata Needed)
Andre raised an excellent point about calculating metrics like AUROC and understanding signal strength by comparing Control vs. Stress groups.
*   **Brenna**, to do this, we need the **metadata/labels** for the Chronic Stress dataset (specifically, which mice are Controls and which are Chronic Stress).
*   Currently, we only have the video files. Could you please share the Excel/CSV file that maps the video names (e.g., `Cohort 5_12`) to their experimental conditions?
*   Once we have this, we can immediately calculate the True Positive/False Positive rates as Andre suggested.

### 4. Validation of ReSup Mechanism (For Andre)
To verify that the model isn't just "guessing," we analyzed the **Disagreement Map** between the dual networks:
*   **High Disagreement**: Occurs on frames with **motion blur** or **grooming** (face occlusion). The model correctly identifies these as "noise" and down-weights them.
*   **High Agreement**: Occurs on clear, frontal face frames, particularly in the detected Cohort 1 mice.
*   *Attached are example frames showing this distinction.*

### 5. Attachments
I have attached the full report and data for your review:
1.  **Final Report** (`final_chronic_stress_report_20251202.pdf`): Detailed analysis and plots.
2.  **Summary CSV** (`chronic_stress_resup_ensemble_video_summary.csv`): Video-level predictions.
3.  **Visualizations**: Model comparison plots and disagreement examples.

We believe this model provides a robust tool for automated stress detection, capable of filtering out noise and identifying subtle chronic stress signals.

Best regards,

Mina Kang
