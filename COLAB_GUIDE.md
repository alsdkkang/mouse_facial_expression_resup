# 🚀 Google Colab 학습 가이드

배터리 소모 없이 더 빠른 속도로 학습하기 위해 Google Colab을 사용하는 방법입니다.

## 1. 준비물 (파일 압축)

맥북에서 다음 두 개의 zip 파일을 만들어주세요.

### 📦 1. `project_code.zip` (코드 파일들)
다음 폴더와 파일들을 포함해야 합니다:
- `mouse_facial_expressions/` (폴더 전체)
- `scripts/` (폴더 전체)
- `train_resup_model.py`
- `train_advanced_model.py`
- `setup.py`
- `requirements.txt`

### 📦 2. `data.zip` (데이터 파일들)
다음 폴더들을 포함해야 합니다 (용량 주의):
- `data/processed/` (메타데이터)
- `data/local_frames/` (학습용 이미지 - 약 3.3GB)
> ⚠️ 주의: `data/chronic_stress_frames` (126GB)는 너무 크므로 **포함하지 마세요**. 평가는 나중에 맥북에서 따로 진행합니다.

## 2. Google Drive 업로드

1. Google Drive에 `Mouse_Project`라는 새 폴더를 만듭니다.
2. 위에서 만든 `project_code.zip`과 `data.zip`을 그 안에 업로드합니다.
3. `Colab_Training.ipynb` 파일도 같은 곳에 업로드합니다.

## 3. Colab 실행

1. 업로드한 `Colab_Training.ipynb`를 더블 클릭하여 Colab에서 엽니다.
2. **런타임 > 런타임 유형 변경**에서 **T4 GPU** (또는 더 좋은 GPU)를 선택합니다.
3. 위에서부터 셀을 차례대로 실행합니다.

## 4. 학습 후

학습이 완료되면 `models/checkpoints_resup` 폴더에 `.ckpt` 파일이 생성됩니다.
이 파일을 다운로드하여 맥북의 같은 위치에 넣어주면, 평가 스크립트를 돌릴 수 있습니다.
