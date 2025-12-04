import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def compare_models():
    print("="*80)
    print("Comparative Analysis: Andre's Model vs. ReSup Model (Ensemble)")
    print("="*80)

    # File paths
    andre_path = "chronic_stress_advanced_video_summary.csv"
    resup_path = "chronic_stress_resup_ensemble_video_summary.csv"

    if not os.path.exists(andre_path) or not os.path.exists(resup_path):
        print(f"Error: One or both result files not found.\n{andre_path}\n{resup_path}")
        return

    # Load data
    df_andre = pd.read_csv(andre_path)
    df_resup = pd.read_csv(resup_path)

    # Rename columns for merging
    df_andre = df_andre.rename(columns={'prob_stress': 'prob_andre', 'predicted_class': 'pred_andre'})
    df_resup = df_resup.rename(columns={'prob_stress': 'prob_resup', 'predicted_class': 'pred_resup'})

    # Merge on video_name
    df_merged = pd.merge(df_andre[['video_name', 'prob_andre', 'pred_andre']], 
                         df_resup[['video_name', 'prob_resup', 'pred_resup']], 
                         on='video_name')

    print(f"Loaded and merged {len(df_merged)} videos.")

    # 1. Quantitative Summary
    print("\n1. Quantitative Summary")
    print("-" * 40)
    
    avg_andre = df_merged['prob_andre'].mean()
    avg_resup = df_merged['prob_resup'].mean()
    
    detected_andre = df_merged['pred_andre'].sum()
    detected_resup = df_merged['pred_resup'].sum()
    
    print(f"{'Metric':<25} | {'Andre Model':<15} | {'ReSup Model':<15}")
    print("-" * 65)
    print(f"{'Avg Stress Probability':<25} | {avg_andre:.4f}{'':<9} | {avg_resup:.4f}")
    print(f"{'Stress Cases Detected':<25} | {detected_andre}/{len(df_merged)} ({detected_andre/len(df_merged):.1%})   | {detected_resup}/{len(df_merged)} ({detected_resup/len(df_merged):.1%})")
    print("-" * 65)

    # 2. Correlation Analysis
    correlation = df_merged['prob_andre'].corr(df_merged['prob_resup'])
    print(f"\nCorrelation between models: {correlation:.4f}")

    # 3. Top Differences (Where did ReSup find stress that Andre missed?)
    print("\n2. Top Disagreements (ReSup detected Stress, Andre missed)")
    print("-" * 40)
    
    disagreements = df_merged[(df_merged['pred_resup'] == 1) & (df_merged['pred_andre'] == 0)].copy()
    disagreements['diff'] = disagreements['prob_resup'] - disagreements['prob_andre']
    disagreements = disagreements.sort_values('diff', ascending=False)
    
    if not disagreements.empty:
        print(f"{'Video Name':<25} | {'Andre Prob':<10} | {'ReSup Prob':<10} | {'Diff':<10}")
        print("-" * 65)
        for _, row in disagreements.iterrows():
            print(f"{row['video_name']:<25} | {row['prob_andre']:.4f}     | {row['prob_resup']:.4f}     | +{row['diff']:.4f}")
    else:
        print("No disagreements found where ReSup detected stress.")

    # 4. Visualization
    print("\n3. Generating Visualization...")
    
    # Identify Cohort 1
    df_merged['is_cohort1'] = df_merged['video_name'].apply(lambda x: 'Cohort1' in x)
    
    plt.figure(figsize=(12, 6))
    
    # Sort by ReSup probability for better visualization
    df_plot = df_merged.sort_values('prob_resup', ascending=False)
    
    x = np.arange(len(df_plot))
    width = 0.35
    
    plt.bar(x - width/2, df_plot['prob_andre'], width, label="Andre's Model", alpha=0.7)
    plt.bar(x + width/2, df_plot['prob_resup'], width, label="ReSup Model (5-Fold)", alpha=0.7)
    
    plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold (0.5)')
    
    plt.xlabel('Videos (Sorted by ReSup Confidence)')
    plt.ylabel('Predicted Stress Probability')
    plt.title('Comparison of Stress Probabilities: Andre vs. ReSup (5-Fold Ensemble)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Save plot
    output_img = "model_comparison_chronic_stress.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_img}")
    
    # Scatter plot with Cohort 1 Highlighting
    plt.figure(figsize=(8, 8))
    
    # Plot non-Cohort 1
    sns.scatterplot(data=df_merged[~df_merged['is_cohort1']], x='prob_andre', y='prob_resup', 
                    color='gray', label='Other Cohorts', s=50, alpha=0.6)
    
    # Plot Cohort 1
    sns.scatterplot(data=df_merged[df_merged['is_cohort1']], x='prob_andre', y='prob_resup', 
                    color='red', label='Cohort 1', s=80, marker='D')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Identity Line', alpha=0.5)
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Threshold')
    plt.axvline(x=0.5, color='g', linestyle='--', alpha=0.5)
    
    plt.xlabel("Andre's Model Probability")
    plt.ylabel("ReSup Model Probability (5-Fold)")
    plt.title("Probability Correlation (Highlighting Cohort 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_scatter = "model_comparison_scatter.png"
    plt.savefig(output_scatter, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {output_scatter}")

if __name__ == "__main__":
    compare_models()
