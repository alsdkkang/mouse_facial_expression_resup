import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def extract_animal_id(video_name):
    # Try to find a standalone number or a number surrounded by underscores
    # Examples: "Cohort1_26_facial exp", "Cohort 5_12_FacialExp", "Cohort3_33i_FacialExp"
    # Match _12_ or _33i_
    match = re.search(r'_(\d+)[a-zA-Z]*_', video_name)
    if match:
        return int(match.group(1))
    
    # Fallback: Look for "CohortX_ID" pattern
    # e.g. Cohort1_26
    match = re.search(r'Cohort\s*\d+\s*[ _]+(\d+)', video_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
        
    return None

def analyze_chronic_stress():
    print("Loading data...")
    preds_df = pd.read_csv("chronic_stress_resup_ensemble_video_summary.csv")
    meta_df = pd.read_csv("chronic_stress_metadata.csv")
    
    print(f"Predictions: {len(preds_df)} videos")
    print(f"Metadata: {len(meta_df)} animals")
    
    # Extract ID from video name
    preds_df['AnimalID'] = preds_df['video_name'].apply(extract_animal_id)
    
    # Check for missing IDs
    missing_ids = preds_df[preds_df['AnimalID'].isnull()]
    if not missing_ids.empty:
        print("Warning: Could not extract AnimalID from the following videos:")
        print(missing_ids['video_name'].tolist())
        
    # Merge
    # Note: Metadata has AnimalID as int
    merged_df = pd.merge(preds_df, meta_df, on='AnimalID', how='left')
    
    # Check for unmatched videos
    unmatched = merged_df[merged_df['Condition'].isnull()]
    if not unmatched.empty:
        print(f"Warning: {len(unmatched)} videos could not be matched to metadata.")
        print(unmatched[['video_name', 'AnimalID']])
        
    # Filter out unmatched for analysis
    valid_df = merged_df.dropna(subset=['Condition']).copy()
    print(f"Matched {len(valid_df)} videos with metadata.")
    
    # Save merged data
    valid_df.to_csv("chronic_stress_metadata_analysis.csv", index=False)
    
    # --- Analysis ---
    
    # 1. Filter out "Het" for binary classification metrics
    binary_df = valid_df[valid_df['Condition'] != 'Het'].copy()
    
    # Map Condition to binary label (Stress=1, Control=0)
    # Note: Metadata uses "Stress" and "Control" (case sensitive? let's normalize)
    binary_df['Condition'] = binary_df['Condition'].str.title() # Ensure Title Case
    binary_df['True_Label'] = binary_df['Condition'].map({'Stress': 1, 'Control': 0})
    
    # Check if mapping worked
    if binary_df['True_Label'].isnull().any():
        print("Error: Found unknown conditions in binary dataset:")
        print(binary_df[binary_df['True_Label'].isnull()]['Condition'].unique())
        return

    y_true = binary_df['True_Label']
    y_pred = binary_df['predicted_class'] # 0 or 1 from model
    y_prob = binary_df['prob_stress']
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print("Performance Metrics (Stress vs Control)")
    print("(Excluding Het group)")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 40)
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("(TN FP)")
    print("(FN TP)")
    
    # --- Visualizations ---
    os.makedirs("chronic_stress_visualizations", exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Boxplot: Stress Probability by Condition (including Het)
    plt.figure(figsize=(10, 6))
    # Order: Control, Het, Stress
    order = ['Control', 'Het', 'Stress']
    # Normalize condition names in full df too
    valid_df['Condition'] = valid_df['Condition'].str.title()
    
    sns.boxplot(data=valid_df, x='Condition', y='prob_stress', order=order, palette="Set2")
    sns.swarmplot(data=valid_df, x='Condition', y='prob_stress', order=order, color=".25", size=4)
    plt.title("Predicted Stress Probability by Condition")
    plt.ylabel("Stress Probability (ReSup Model)")
    plt.axhline(0.5, color='r', linestyle='--', label='Threshold (0.5)')
    plt.legend()
    plt.savefig("chronic_stress_visualizations/stress_by_condition.png")
    plt.close()
    
    # 2. Boxplot: By Genotype and Condition
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=valid_df, x='Genotype', y='prob_stress', hue='Condition', palette="Set2")
    plt.title("Predicted Stress Probability by Genotype and Condition")
    plt.ylabel("Stress Probability")
    plt.axhline(0.5, color='r', linestyle='--', alpha=0.5)
    plt.savefig("chronic_stress_visualizations/stress_by_genotype.png")
    plt.close()
    
    # 3. Boxplot: By Sex and Condition
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=valid_df, x='Sex', y='prob_stress', hue='Condition', palette="Set2")
    plt.title("Predicted Stress Probability by Sex and Condition")
    plt.ylabel("Stress Probability")
    plt.axhline(0.5, color='r', linestyle='--', alpha=0.5)
    plt.savefig("chronic_stress_visualizations/stress_by_sex.png")
    plt.close()

    print("\nVisualizations saved to chronic_stress_visualizations/")

if __name__ == "__main__":
    analyze_chronic_stress()
