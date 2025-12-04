
# GPU 학습 래퍼 스크립트
import sys
import os

# 원본 스크립트 임포트
sys.path.insert(0, os.path.join(os.getcwd(), 'mouse_facial_expressions', 'models'))

# train_task1_baseline_model.py의 내용을 수정하여 실행
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_module", 
    "mouse_facial_expressions/models/train_task1_baseline_model.py"
)
train_module = importlib.util.module_from_spec(spec)

# accelerator를 'mps'로 변경하기 위해 monkey patch
import lightning.pytorch as pl
original_trainer = pl.Trainer

def patched_trainer(*args, **kwargs):
    # accelerator를 'mps'로 강제 설정
    kwargs['accelerator'] = 'mps' if __import__('torch').backends.mps.is_available() else 'cpu'
    return original_trainer(*args, **kwargs)

pl.Trainer = patched_trainer

# 모듈 실행
spec.loader.exec_module(train_module)
