# GPU 서버에서 학습하기 (Google Colab)

이 가이드는 무료 GPU를 사용하여 빠르게 모델을 학습하는 방법을 설명합니다.

## 준비물

1. **데이터 압축 파일 만들기**

로컬에서 실행:
```bash
cd /Users/minakang/Desktop/mouse-facial-expressions-2023-main

# 필요한 데이터만 압축
zip -r data_for_colab.zip \
  data/processed/task-1.1/ \
  data/raw/treatments.csv \
  data/raw/raw_videos.csv \
  mouse_facial_expressions/ \
  setup.py \
  .env
```

2. **Google Drive에 업로드**
   - `data_for_colab.zip` 파일을 Google Drive에 업로드
   - 또는 Colab에서 직접 업로드

## 사용 방법

### Option 1: Google Colab (무료 GPU)

1. **Colab 열기**
   - https://colab.research.google.com 접속
   - File > Upload notebook
   - `GPU_Training.ipynb` 업로드

2. **GPU 활성화**
   - Runtime > Change runtime type
   - Hardware accelerator: GPU 선택
   - GPU type: T4 (무료) 또는 V100/A100 (Colab Pro)

3. **노트북 실행**
   - 모든 셀을 순서대로 실행
   - 데이터 업로드 시 `data_for_colab.zip` 선택

4. **학습 시간**
   - T4 GPU: ~1-2시간
   - V100 GPU: ~30분-1시간

### Option 2: Kaggle (무료 GPU)

1. **Kaggle 계정 생성**
   - https://www.kaggle.com

2. **새 노트북 생성**
   - Notebooks > New Notebook
   - Settings > Accelerator: GPU T4 x2

3. **데이터 업로드**
   - Add Data > Upload
   - `data_for_colab.zip` 업로드

4. **노트북 코드 복사**
   - `GPU_Training.ipynb` 내용 복사하여 실행

## 데이터 업로드 옵션

### 방법 1: Google Drive (추천)

```python
from google.colab import drive
drive.mount('/content/drive')

# 압축 해제
!unzip /content/drive/MyDrive/data_for_colab.zip -d /content/
```

### 방법 2: 직접 업로드

```python
from google.colab import files
uploaded = files.upload()  # data_for_colab.zip 선택

!unzip data_for_colab.zip
```

### 방법 3: wget (공개 URL이 있는 경우)

```python
!wget https://your-url.com/data_for_colab.zip
!unzip data_for_colab.zip
```

## 학습 후 체크포인트 다운로드

### Google Drive에 저장

```python
!mkdir -p /content/drive/MyDrive/mouse_model_checkpoints
!cp -r models/checkpoints/* /content/drive/MyDrive/mouse_model_checkpoints/
```

### 직접 다운로드

```python
from google.colab import files
import glob

checkpoints = glob.glob('models/checkpoints/*.ckpt')
for ckpt in checkpoints:
    files.download(ckpt)
```

## 로컬에서 사용하기

1. **체크포인트 다운로드**
   - Google Drive 또는 Colab에서 다운로드

2. **로컬에 복사**
```bash
cp downloaded_checkpoint.ckpt /Users/minakang/Desktop/mouse-facial-expressions-2023-main/models/checkpoints/
```

3. **Fine-tuning 시작**
```bash
cd /Users/minakang/Desktop/mouse-facial-expressions-2023-main

python mouse_facial_expressions/models/finetune_chronic.py \
  --checkpoint_path models/checkpoints/task1-fold0-epoch=9-val_acc=0.95.ckpt \
  --epochs 5 \
  --dataset_version "chronic_v1"
```

## 문제 해결

### GPU 메모리 부족

배치 사이즈 줄이기:
```python
!python mouse_facial_expressions/models/train_task1_baseline_model.py \
    --epochs 10 \
    --dataset_version "1.1" \
    --train_batch_size 16  # 32에서 16으로 줄임
```

### 세션 타임아웃

Colab은 12시간 제한이 있습니다. 중간 체크포인트를 주기적으로 저장하세요.

### 데이터 업로드 느림

- Google Drive 마운트 사용 (가장 빠름)
- 압축 파일 크기 줄이기 (필요한 데이터만)

## 비용

- **Google Colab (무료)**: T4 GPU, 12시간 제한
- **Colab Pro ($10/월)**: V100/A100 GPU, 24시간 제한
- **Kaggle (무료)**: T4 x2 GPU, 30시간/주

## 예상 학습 시간

| GPU | Epoch당 | 10 Epochs |
|-----|---------|-----------|
| T4 | 5-10분 | 1-2시간 |
| V100 | 2-3분 | 30분-1시간 |
| A100 | 1-2분 | 15-30분 |

무료 T4 GPU로도 충분히 빠르게 학습할 수 있습니다!
