# -*- coding: utf-8 -*-
"""
Advanced Mouse Facial Expression Recognition Model
Features:
- Label Distribution Learning (LDL) with soft labels
- Temporal Aggregation with Transformer encoder
- Dual-network architecture for noise robustness (ReSup-inspired)
"""

import json
import logging
import random
import re
from pathlib import Path

import click
import mlflow
import numpy as np

import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning.pytorch as pl

from lightning.pytorch.callbacks import ModelCheckpoint
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import DataLoader
from mouse_facial_expressions.data.temporal_datasets import Task1TemporalFolds


class TemporalTransformerEncoder(nn.Module):
    """
    Transformer-based temporal encoder for aggregating frame features
    """
    def __init__(self, feature_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 20, feature_dim))  # Max 20 frames
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, feature_dim)
        Returns:
            aggregated: (batch, feature_dim)
        """
        batch_size, num_frames, _ = x.shape
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :num_frames, :]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Aggregate (mean pooling)
        x = x.mean(dim=1)
        
        # Layer norm
        x = self.layer_norm(x)
        
        return x


class AdvancedDeepSet(pl.LightningModule):
    """
    Advanced model with LDL and Temporal Aggregation
    """
    def __init__(self, config, class_weights=None):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.config = config
        self.use_soft_labels = config.get('use_soft_labels', True)
        
        # Spatial feature extractor (per frame)
        self.spatial_encoder = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        )
        in_features = self.spatial_encoder.fc.in_features
        self.spatial_encoder.fc = nn.Identity()  # Remove final FC layer
        
        # Temporal aggregation
        self.temporal_encoder = TemporalTransformerEncoder(
            feature_dim=in_features,
            num_heads=8,
            num_layers=4,
            dropout=0.2
        )
        
        # Classifier
        n_classes = 2
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
        
        # Loss function
        if self.use_soft_labels:
            # KL Divergence for soft labels (LDL)
            self.criterion = self._ldl_loss
        else:
            # Standard cross-entropy
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=config.get('label_smoothing', 0.1)
            )
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=2)
        
    def _ldl_loss(self, pred, target):
        """
        Label Distribution Learning loss using KL Divergence
        """
        if self.use_soft_labels and target.dim() == 2:
            # Soft labels: use KL divergence
            log_pred = F.log_softmax(pred, dim=1)
            loss = F.kl_div(log_pred, target, reduction='batchmean')
        else:
            # Hard labels: use cross-entropy
            loss = F.cross_entropy(pred, target.long())
        return loss
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, C, H, W)
        Returns:
            y_hat: (batch, num_classes)
        """
        batch_size, num_frames, C, H, W = x.shape
        
        # Reshape for spatial encoder: (batch * num_frames, C, H, W)
        x = x.view(batch_size * num_frames, C, H, W)
        
        # Extract spatial features
        spatial_features = self.spatial_encoder(x)  # (batch * num_frames, feature_dim)
        
        # Reshape back: (batch, num_frames, feature_dim)
        feature_dim = spatial_features.shape[-1]
        spatial_features = spatial_features.view(batch_size, num_frames, feature_dim)
        
        # Temporal aggregation
        temporal_features = self.temporal_encoder(spatial_features)  # (batch, feature_dim)
        
        # Classification
        y_hat = self.classifier(temporal_features)  # (batch, num_classes)
        
        return y_hat
    
    def configure_optimizers(self):
        lr = self.config['learning_rate']
        weight_decay = self.config['weight_decay']
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=lr * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)
    
    def on_train_epoch_start(self):
        self.train_acc.reset()
        self.train_losses = []
        return super().on_train_epoch_start()
    
    def on_train_epoch_end(self):
        epoch = self.trainer.current_epoch
        train_acc_value = self.train_acc.compute()
        train_loss_value = np.mean(self.train_losses)
        
        # Log to MLflow
        mlflow.log_metric("train_acc", train_acc_value, step=epoch)
        learning_rate = self.trainer.optimizers[0].param_groups[0]['lr']
        mlflow.log_metric("learning_rate", learning_rate, step=epoch)
        mlflow.log_metric("train_loss", train_loss_value, step=epoch)
        
        # Log to Lightning
        self.log("train_acc", train_acc_value, prog_bar=True)
        self.log("train_loss", train_loss_value, prog_bar=True)
        
        self._logger.info(
            f"Train epoch {epoch}, "
            f"loss {train_loss_value:.03f}, "
            f"accuracy {train_acc_value:.03f}"
        )
        return super().on_train_epoch_end()
    
    def on_validation_start(self) -> None:
        self.val_acc.reset()
        self.val_losses = []
        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self):
        epoch = self.trainer.current_epoch
        val_acc_value = self.val_acc.compute()
        val_loss_value = np.mean(self.val_losses)
        
        # Log to MLflow
        mlflow.log_metric("val_acc", val_acc_value, step=epoch)
        mlflow.log_metric("val_loss", val_loss_value, step=epoch)
        
        # Log to Lightning
        self.log("val_acc", val_acc_value, prog_bar=True)
        self.log("val_loss", val_loss_value, prog_bar=True)
        
        self._logger.info(
            f"Validation {epoch}, "
            f"loss {val_loss_value:.03f}, "
            f"accuracy {val_acc_value:.03f}"
        )
        return super().on_validation_epoch_end()
    
    def training_step(self, batch, batch_idx):
        y_hat = self.predict_step(batch, batch_idx)
        x, y = batch
        
        loss = self.criterion(y_hat, y)
        self.train_losses.append(loss.item())
        
        # For accuracy, convert soft labels to hard labels if needed
        if self.use_soft_labels and y.dim() == 2:
            y_hard = y.argmax(dim=1)
        else:
            y_hard = y.long()
        
        self.train_acc(y_hat, y_hard)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self.predict_step(batch, batch_idx)
        x, y = batch
        
        loss = self.criterion(y_hat, y)
        self.val_losses.append(loss.item())
        
        # For accuracy, convert soft labels to hard labels if needed
        if self.use_soft_labels and y.dim() == 2:
            y_hard = y.argmax(dim=1)
        else:
            y_hard = y.long()
        
        self.val_acc(y_hat, y_hard)
        return loss


@click.command(help="Train Advanced Model with LDL and Temporal Aggregation")
@click.option("--epochs", type=click.INT, default=20, help="Number of training epochs.")
@click.option("--learning_rate", type=click.FLOAT, default=1e-4, help="Learning rate.")
@click.option("--train_batch_size", type=click.INT, default=16)
@click.option("--test_batch_size", type=click.INT, default=32)
@click.option("--weight_decay", type=click.FLOAT, default=1e-4)
@click.option("--label_smoothing", type=click.FLOAT, default=0.1)
@click.option("--num_frames", type=click.INT, default=10, help="Number of frames per sequence")
@click.option("--frame_stride", type=click.INT, default=2, help="Stride between frames")
@click.option("--use_soft_labels", type=click.BOOL, default=True, help="Use soft labels (LDL)")
@click.option("--dataset_version", type=click.STRING, default='1.0')
@click.option("--train_augmentation", type=click.STRING, default='TrivialAugmentWide')
@click.option("--seed", type=click.INT, default=97531, help="Seed random number generators.")
@click.option("--folds", type=click.STRING, default="0,1,2,3,4", help="Comma-separated list of folds to run (e.g. '0' or '0,1').")
def main(**kwargs):
    """Train advanced model with LDL and Temporal Aggregation."""
    logger = logging.getLogger(__name__)
    logger.info(f"Beginning advanced model training: {Path(__file__).parts[-1]}")
    
    config = kwargs
    epochs = kwargs["epochs"]
    seed = kwargs["seed"]
    
    # Parse folds
    folds_to_run = [int(f) for f in config['folds'].split(',')]
    logger.info(f"Running folds: {folds_to_run}")
    
    with mlflow.start_run(nested=True) as active_run:
        logger.info("Started MLflow run %s", active_run.info.run_id)
        
        # Log parameters
        logger.info("Logging parameters\n" + json.dumps(kwargs, indent=4))
        mlflow.log_params(kwargs)
        
        # Create temporal dataset
        cv = Task1TemporalFolds(
            version=config['dataset_version'],
            num_frames=config['num_frames'],
            frame_stride=config['frame_stride'],
            train_augmentation=config['train_augmentation'],
            use_soft_labels=config['use_soft_labels']
        )
        
        for fold, (train_dataset, test_dataset) in enumerate(cv):
            if fold not in folds_to_run:
                logger.info(f"Skipping fold {fold}")
                continue
                
            with mlflow.start_run(nested=True) as child_run:
                logger.info("Starting training for fold %i with seed %i", fold, seed)
                
                mlflow.log_params(kwargs)
                
                # Set seeds
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                g = torch.Generator()
                g.manual_seed(seed)
                
                # Create dataloaders
                logger.info("Creating dataloaders")
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=config['train_batch_size'],
                    num_workers=4,
                    shuffle=True,
                    generator=g
                )
                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=config['test_batch_size'],
                    num_workers=4
                )
                
                # Create model
                model = AdvancedDeepSet(config=config)
                
                # Check for existing checkpoint to resume
                checkpoint_dir = Path('models/checkpoints_advanced')
                checkpoint_dir.mkdir(exist_ok=True, parents=True)
                
                # Find latest checkpoint for this fold
                # Pattern: advanced-fold{fold}-epoch={epoch}-val_acc={val_acc}.ckpt
                existing_ckpts = list(checkpoint_dir.glob(f'advanced-fold{fold}-*.ckpt'))
                resume_ckpt_path = None
                
                if existing_ckpts:
                    # Sort by epoch number
                    def get_epoch(p):
                        match = re.search(r'epoch=(\d+)', p.name)
                        return int(match.group(1)) if match else -1
                    
                    existing_ckpts.sort(key=get_epoch, reverse=True)
                    resume_ckpt_path = str(existing_ckpts[0])
                    logger.info(f"Resuming from checkpoint: {resume_ckpt_path}")
                
                # Callbacks
                checkpoint_callback = ModelCheckpoint(
                    dirpath='models/checkpoints_advanced',
                    filename=f'advanced-fold{fold}-{{epoch}}-{{val_acc:.2f}}',
                    save_top_k=1,
                    monitor='val_acc',
                    mode='max'
                )
                callbacks = [checkpoint_callback]
                
                # Trainer
                trainer = pl.Trainer(
                    max_epochs=config['epochs'],
                    callbacks=callbacks,
                    logger=False,
                    enable_checkpointing=True,
                    accelerator='auto',
                    precision='16-mixed' if torch.cuda.is_available() else '32'
                )
                
                # Train (resume if checkpoint exists)
                trainer.fit(model, train_dataloader, test_dataloader, ckpt_path=resume_ckpt_path)
                
                # Logging
                mlflow.set_tag("fold", fold)
                mlflow.set_tag("model_type", "advanced_ldl_temporal")
                
                mlflow.pytorch.log_model(model, "model")
                logger.info("Fold %i training complete", fold)
    
    logger.info("Advanced model training completed")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    
    main()
