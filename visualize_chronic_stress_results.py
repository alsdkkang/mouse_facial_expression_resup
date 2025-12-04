"""
Chronic Stress Data Evaluation - Visualization Script
이 스크립트는 Chronic Stress 데이터셋 평가 결과를 시각화합니다.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 한글 폰트 설정 (Mac용)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")

# 출력 디렉토리 생성
output_dir = Path("chronic_stress_visualizations")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Chronic Stress Data Evaluation - 결과 시각화")
print("=" * 80)

# 1. 데이터 로드
print("\n[1] 데이터 로드 중...")
predictions_df = pd.read_csv("chronic_stress_predictions.csv")
video_summary_df = pd.read_csv("chronic_stress_video_summary.csv")

print(f"  - 총 프레임 수: {len(predictions_df):,}")
print(f"  - 총 비디오 수: {len(video_summary_df)}")

# 2. 전체 예측 분포
print("\n[2] 전체 예측 분포 시각화...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2-1. 예측 레이블 분포 (카운트)
label_counts = predictions_df['predicted_label'].value_counts().sort_index()
axes[0].bar(label_counts.index, label_counts.values, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[0].set_xlabel('예측 레이블', fontsize=12, fontweight='bold')
axes[0].set_ylabel('프레임 수', fontsize=12, fontweight='bold')
axes[0].set_title('전체 예측 레이블 분포 (프레임 단위)', fontsize=14, fontweight='bold')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Label 0\n(Non-stress)', 'Label 1\n(Stress)'])
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 500, f'{v:,}\n({v/len(predictions_df)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')

# 2-2. 예측 확률 분포
axes[1].hist(predictions_df['prob_0'], bins=50, alpha=0.6, label='Prob(Label 0)', color='#3498db', edgecolor='black')
axes[1].hist(predictions_df['prob_1'], bins=50, alpha=0.6, label='Prob(Label 1)', color='#e74c3c', edgecolor='black')
axes[1].set_xlabel('예측 확률', fontsize=12, fontweight='bold')
axes[1].set_ylabel('프레임 수', fontsize=12, fontweight='bold')
axes[1].set_title('예측 확률 분포', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "01_overall_prediction_distribution.png", dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: {output_dir / '01_overall_prediction_distribution.png'}")
plt.close()

# 3. 비디오별 예측 분포
print("\n[3] 비디오별 예측 분포 시각화...")

# 비디오 이름 정리 및 정렬
video_summary_df = video_summary_df.rename(columns={'Unnamed: 0': 'video_name'})
video_summary_df = video_summary_df.sort_values('video_name')

# 3-1. 비디오별 Label 1 비율 (가로 막대 그래프)
fig, ax = plt.subplots(figsize=(12, 16))

video_summary_df['total'] = video_summary_df['0'] + video_summary_df['1']
video_summary_df['label_1_pct'] = (video_summary_df['1'] / video_summary_df['total']) * 100
video_summary_df_sorted = video_summary_df.sort_values('label_1_pct', ascending=True)

colors = ['#2ecc71' if pct == 0 else '#e74c3c' if pct > 10 else '#f39c12' 
          for pct in video_summary_df_sorted['label_1_pct']]

bars = ax.barh(range(len(video_summary_df_sorted)), video_summary_df_sorted['label_1_pct'], 
               color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(video_summary_df_sorted)))
ax.set_yticklabels(video_summary_df_sorted['video_name'], fontsize=9)
ax.set_xlabel('Label 1 (Stress) 비율 (%)', fontsize=12, fontweight='bold')
ax.set_title('비디오별 Stress 예측 비율', fontsize=14, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)

# 값 표시
for i, (idx, row) in enumerate(video_summary_df_sorted.iterrows()):
    if row['label_1_pct'] > 0:
        ax.text(row['label_1_pct'] + 1, i, f"{row['label_1_pct']:.1f}%", 
               va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "02_video_level_stress_percentage.png", dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: {output_dir / '02_video_level_stress_percentage.png'}")
plt.close()

# 3-2. 비디오별 예측 분포 (스택 바 차트)
fig, ax = plt.subplots(figsize=(14, 8))

video_summary_df_sorted = video_summary_df.sort_values('label_1_pct', ascending=False)
x_pos = np.arange(len(video_summary_df_sorted))

ax.bar(x_pos, video_summary_df_sorted['0'], label='Label 0 (Non-stress)', 
       color='#3498db', alpha=0.7, edgecolor='black')
ax.bar(x_pos, video_summary_df_sorted['1'], bottom=video_summary_df_sorted['0'], 
       label='Label 1 (Stress)', color='#e74c3c', alpha=0.7, edgecolor='black')

ax.set_xticks(x_pos)
ax.set_xticklabels(video_summary_df_sorted['video_name'], rotation=90, ha='right', fontsize=8)
ax.set_ylabel('프레임 수', fontsize=12, fontweight='bold')
ax.set_title('비디오별 예측 레이블 분포 (스택 바 차트)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "03_video_level_stacked_bar.png", dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: {output_dir / '03_video_level_stacked_bar.png'}")
plt.close()

# 4. Cohort별 분석
print("\n[4] Cohort별 분석 시각화...")

# Cohort 정보 추출
video_summary_df['cohort'] = video_summary_df['video_name'].str.extract(r'(Cohort\s*\d+)')[0]
video_summary_df['cohort'] = video_summary_df['cohort'].str.replace(' ', '')

cohort_summary = video_summary_df.groupby('cohort').agg({
    '0': 'sum',
    '1': 'sum',
    'video_name': 'count'
}).rename(columns={'video_name': 'video_count'})

cohort_summary['total_frames'] = cohort_summary['0'] + cohort_summary['1']
cohort_summary['label_1_pct'] = (cohort_summary['1'] / cohort_summary['total_frames']) * 100
cohort_summary = cohort_summary.sort_index()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4-1. Cohort별 비디오 수
axes[0, 0].bar(range(len(cohort_summary)), cohort_summary['video_count'], 
               color='#9b59b6', alpha=0.7, edgecolor='black')
axes[0, 0].set_xticks(range(len(cohort_summary)))
axes[0, 0].set_xticklabels(cohort_summary.index, rotation=45)
axes[0, 0].set_ylabel('비디오 수', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Cohort별 비디오 수', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, axis='y', alpha=0.3)
for i, v in enumerate(cohort_summary['video_count']):
    axes[0, 0].text(i, v + 0.2, str(int(v)), ha='center', va='bottom', fontweight='bold')

# 4-2. Cohort별 총 프레임 수
axes[0, 1].bar(range(len(cohort_summary)), cohort_summary['total_frames'], 
               color='#1abc9c', alpha=0.7, edgecolor='black')
axes[0, 1].set_xticks(range(len(cohort_summary)))
axes[0, 1].set_xticklabels(cohort_summary.index, rotation=45)
axes[0, 1].set_ylabel('총 프레임 수', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Cohort별 총 프레임 수', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, axis='y', alpha=0.3)
for i, v in enumerate(cohort_summary['total_frames']):
    axes[0, 1].text(i, v + 200, f'{int(v):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4-3. Cohort별 Label 1 비율
axes[1, 0].bar(range(len(cohort_summary)), cohort_summary['label_1_pct'], 
               color='#e67e22', alpha=0.7, edgecolor='black')
axes[1, 0].set_xticks(range(len(cohort_summary)))
axes[1, 0].set_xticklabels(cohort_summary.index, rotation=45)
axes[1, 0].set_ylabel('Label 1 비율 (%)', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Cohort별 Stress 예측 비율', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, axis='y', alpha=0.3)
for i, v in enumerate(cohort_summary['label_1_pct']):
    axes[1, 0].text(i, v + 0.2, f'{v:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4-4. Cohort별 스택 바 차트
x_pos = np.arange(len(cohort_summary))
axes[1, 1].bar(x_pos, cohort_summary['0'], label='Label 0', color='#3498db', alpha=0.7, edgecolor='black')
axes[1, 1].bar(x_pos, cohort_summary['1'], bottom=cohort_summary['0'], 
               label='Label 1', color='#e74c3c', alpha=0.7, edgecolor='black')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(cohort_summary.index, rotation=45)
axes[1, 1].set_ylabel('프레임 수', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Cohort별 예측 레이블 분포', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "04_cohort_level_analysis.png", dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: {output_dir / '04_cohort_level_analysis.png'}")
plt.close()

# 5. 예측 확률 분석
print("\n[5] 예측 확률 상세 분석...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 5-1. Label 0 확률 분포 (레이블별)
for label in [0, 1]:
    subset = predictions_df[predictions_df['predicted_label'] == label]
    axes[0, 0].hist(subset['prob_0'], bins=50, alpha=0.6, 
                    label=f'Predicted as {label}', edgecolor='black')
axes[0, 0].set_xlabel('Prob(Label 0)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('프레임 수', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Label 0 확률 분포 (예측 레이블별)', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 5-2. Label 1 확률 분포 (레이블별)
for label in [0, 1]:
    subset = predictions_df[predictions_df['predicted_label'] == label]
    axes[0, 1].hist(subset['prob_1'], bins=50, alpha=0.6, 
                    label=f'Predicted as {label}', edgecolor='black')
axes[0, 1].set_xlabel('Prob(Label 1)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('프레임 수', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Label 1 확률 분포 (예측 레이블별)', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 5-3. 예측 신뢰도 분포
predictions_df['confidence'] = predictions_df[['prob_0', 'prob_1']].max(axis=1)
axes[1, 0].hist(predictions_df['confidence'], bins=50, color='#16a085', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('예측 신뢰도 (Max Probability)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('프레임 수', fontsize=11, fontweight='bold')
axes[1, 0].set_title('예측 신뢰도 분포', fontsize=12, fontweight='bold')
axes[1, 0].axvline(predictions_df['confidence'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'평균: {predictions_df["confidence"].mean():.3f}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5-4. 신뢰도 통계
confidence_stats = predictions_df['confidence'].describe()
axes[1, 1].axis('off')
stats_text = f"""
예측 신뢰도 통계

평균 (Mean):     {confidence_stats['mean']:.4f}
표준편차 (Std):   {confidence_stats['std']:.4f}
최소값 (Min):     {confidence_stats['min']:.4f}
25% 분위수:       {confidence_stats['25%']:.4f}
중앙값 (Median):  {confidence_stats['50%']:.4f}
75% 분위수:       {confidence_stats['75%']:.4f}
최대값 (Max):     {confidence_stats['max']:.4f}

신뢰도 > 0.9:    {(predictions_df['confidence'] > 0.9).sum():,} ({(predictions_df['confidence'] > 0.9).sum()/len(predictions_df)*100:.1f}%)
신뢰도 > 0.8:    {(predictions_df['confidence'] > 0.8).sum():,} ({(predictions_df['confidence'] > 0.8).sum()/len(predictions_df)*100:.1f}%)
신뢰도 > 0.7:    {(predictions_df['confidence'] > 0.7).sum():,} ({(predictions_df['confidence'] > 0.7).sum()/len(predictions_df)*100:.1f}%)
"""
axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / "05_prediction_confidence_analysis.png", dpi=300, bbox_inches='tight')
print(f"  ✓ 저장: {output_dir / '05_prediction_confidence_analysis.png'}")
plt.close()

# 6. 요약 통계 출력
print("\n" + "=" * 80)
print("요약 통계")
print("=" * 80)

print(f"\n[전체 통계]")
print(f"  - 총 프레임 수: {len(predictions_df):,}")
print(f"  - Label 0 (Non-stress): {(predictions_df['predicted_label'] == 0).sum():,} ({(predictions_df['predicted_label'] == 0).sum()/len(predictions_df)*100:.2f}%)")
print(f"  - Label 1 (Stress): {(predictions_df['predicted_label'] == 1).sum():,} ({(predictions_df['predicted_label'] == 1).sum()/len(predictions_df)*100:.2f}%)")
print(f"  - 평균 예측 신뢰도: {predictions_df['confidence'].mean():.4f}")

print(f"\n[비디오 통계]")
print(f"  - 총 비디오 수: {len(video_summary_df)}")
print(f"  - Stress 프레임이 없는 비디오: {(video_summary_df['1'] == 0).sum()}")
print(f"  - Stress 프레임이 있는 비디오: {(video_summary_df['1'] > 0).sum()}")
print(f"  - 최대 Stress 비율: {video_summary_df['label_1_pct'].max():.2f}% ({video_summary_df.loc[video_summary_df['label_1_pct'].idxmax(), 'video_name']})")

print(f"\n[Cohort 통계]")
for cohort, row in cohort_summary.iterrows():
    print(f"  - {cohort}: {int(row['video_count'])}개 비디오, {int(row['total_frames']):,}개 프레임, Stress 비율 {row['label_1_pct']:.2f}%")

print("\n" + "=" * 80)
print("시각화 완료! 모든 그래프가 저장되었습니다.")
print(f"저장 위치: {output_dir.absolute()}")
print("=" * 80)
