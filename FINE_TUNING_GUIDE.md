# Fine-Tuning Guide: Using Trained Weights for Chronic Mouse Data

## Overview

This guide explains how to use the trained model weights from the LPS acute pain study to fine-tune on your chronic mouse video data.

## Step 1: Locate the Trained Weights

After training completes, the best model checkpoint will be saved in:
```
models/checkpoints/task1-fold0-epoch=X-val_acc=Y.YY.ckpt
```

The checkpoint with the highest validation accuracy will be automatically saved.

## Step 2: Prepare Your Chronic Data

### 2.1 Create Metadata CSVs

**treatments_chronic.csv:**
```csv
mouse,date_of_birth,treatment,injection_time,notes
c1,2024-01-15,control,10:00,baseline
c2,2024-01-15,chronic_pain,10:15,CCI model
c3,2024-01-16,control,10:30,baseline
c4,2024-01-16,chronic_pain,10:45,CCI model
```

**raw_videos_chronic.csv:**
```csv
animal,recording,camera,year,month,day,hour,minutes,seconds,start,end,discard
c1,1,cam1,2024,2,1,10,0,0,0,-1,0
c1,7,cam1,2024,2,8,10,0,0,0,-1,0
c2,1,cam1,2024,2,1,10,15,0,0,-1,0
c2,7,cam1,2024,2,8,10,15,0,0,-1,0
```

### 2.2 Update Environment Variables

Create or update `.env`:
```bash
MFE_RAW_CSV_FOLDER=/path/to/your/chronic/csvs
MFE_EXTRACTED_FRAMES_FOLDER=/path/to/your/chronic/frames
MFE_VERSION=chronic_v1
MFE_TASKS=/Users/minakang/Desktop/mouse-facial-expressions-2023-main/data/processed
```

### 2.3 Modify Labeling Logic

Edit `mouse_facial_expressions/data/make_datasets.py` around line 91:

```python
# Original LPS labeling:
# combined_df.loc[(combined_df.recording == 1), 'label'] = 0  # preinjection
# combined_df.loc[(combined_df.recording == 4) & (combined_df.treatment == 'high'), 'label'] = 1  # 4h post

# New chronic labeling:
combined_df['label'] = np.nan
combined_df.loc[(combined_df.recording == 1) & (combined_df.treatment == 'control'), 'label'] = 0
combined_df.loc[(combined_df.recording == 7) & (combined_df.treatment == 'chronic_pain'), 'label'] = 1
combined_df = combined_df.dropna(subset='label')
```

### 2.4 Generate Chronic Dataset

```bash
python mouse_facial_expressions/data/make_datasets.py task1 \
  --version "chronic_v1" \
  --frameset_size 1 \
  --train_size 5000 \
  --test_size 500
```

## Step 3: Create Fine-Tuning Script

Create `mouse_facial_expressions/models/finetune_chronic.py`:

```python
import click
from pathlib import Path
from train_task1_baseline_model import *

@click.command()
@click.option("--checkpoint_path", type=click.Path(exists=True), required=True)
@click.option("--epochs", type=click.INT, default=5)
@click.option("--dataset_version", type=click.STRING, default='chronic_v1')
@click.option("--freeze_backbone", type=click.BOOL, default=False)
@click.option("--learning_rate", type=click.FLOAT, default=1e-3)
def main(checkpoint_path, epochs, dataset_version, freeze_backbone, learning_rate):
    """Fine-tune model on chronic data."""
    logger = logging.getLogger(__name__)
    logger.info(f"Fine-tuning from checkpoint: {checkpoint_path}")
    
    config = {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'dataset_version': dataset_version,
        'train_augmentation': 'TrivialAugmentWide'
    }
    
    with mlflow.start_run(nested=True) as active_run:
        mlflow.log_params(config)
        
        cv = Task1Folds(version=config['dataset_version'], 
                       train_augmentation=config['train_augmentation'])
        
        for fold, (train_dataset, test_dataset) in enumerate(cv):
            with mlflow.start_run(nested=True) as child_run:
                logger.info(f"Fine-tuning fold {fold}")
                
                train_dataloader = DataLoader(train_dataset, batch_size=20, 
                                            num_workers=8, shuffle=True)
                test_dataloader = DataLoader(test_dataset, batch_size=40, 
                                           num_workers=8)
                
                weights = train_dataset.get_class_weights()
                weights = torch.from_numpy(weights).float()
                model = DeepSet(config=config, class_weights=weights)
                
                # Load pre-trained weights
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                logger.info("Loaded pre-trained weights")
                
                # Optionally freeze backbone
                if freeze_backbone:
                    for param in model.model_.parameters():
                        param.requires_grad = False
                    logger.info("Froze ResNet backbone")
                
                checkpoint_callback = ModelCheckpoint(
                    dirpath='models/checkpoints/chronic',
                    filename=f'chronic-fold{fold}-{{epoch}}-{{val_acc:.2f}}',
                    save_top_k=1,
                    monitor='val_acc',
                    mode='max'
                )
                
                trainer = pl.Trainer(
                    max_epochs=config['epochs'],
                    callbacks=[checkpoint_callback],
                    enable_checkpointing=True,
                    accelerator='cpu',
                )
                
                trainer.fit(model, train_dataloader, test_dataloader)
                mlflow.pytorch.log_model(model, "model")
                
        logger.info("Fine-tuning complete")

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
```

## Step 4: Run Fine-Tuning

### Option A: Fine-tune entire model
```bash
python mouse_facial_expressions/models/finetune_chronic.py \
  --checkpoint_path models/checkpoints/task1-fold0-epoch=4-val_acc=0.95.ckpt \
  --epochs 5 \
  --dataset_version "chronic_v1" \
  --freeze_backbone False \
  --learning_rate 0.001
```

### Option B: Freeze backbone, train only final layer
```bash
python mouse_facial_expressions/models/finetune_chronic.py \
  --checkpoint_path models/checkpoints/task1-fold0-epoch=4-val_acc=0.95.ckpt \
  --epochs 10 \
  --dataset_version "chronic_v1" \
  --freeze_backbone True \
  --learning_rate 0.01
```

## Step 5: Evaluate on Chronic Data

After fine-tuning, evaluate the model:

```python
# Load fine-tuned model
model = DeepSet.load_from_checkpoint(
    'models/checkpoints/chronic/chronic-fold0-epoch=4-val_acc=0.XX.ckpt'
)

# Run predictions on your chronic test set
# ... evaluation code ...
```

## Tips for Success

1. **Start with frozen backbone** if you have limited chronic data (<1000 samples)
2. **Use lower learning rate** (1e-4 to 1e-3) for fine-tuning to avoid catastrophic forgetting
3. **Monitor validation accuracy** - if it plateaus quickly, try unfreezing the backbone
4. **Data augmentation** helps prevent overfitting on small chronic datasets
5. **Cross-validation** is crucial - use all 8 folds if possible

## Expected Results

- **With frozen backbone**: Faster training, good for limited data
- **Full fine-tuning**: Better performance if you have sufficient chronic data (>2000 samples)
- **Typical improvement**: 5-15% accuracy gain over training from scratch

## Troubleshooting

**Q: Model performs worse than random?**
- Check that your chronic data labels are correct
- Verify the labeling logic in `make_datasets.py`
- Ensure frames are properly aligned/cropped

**Q: Training is very slow?**
- Consider using a GPU-enabled machine
- Reduce batch size if running out of memory
- Use fewer workers if I/O is bottlenecked

**Q: Overfitting on chronic data?**
- Use more aggressive data augmentation
- Freeze more layers of the backbone
- Add dropout to the final classification layer
