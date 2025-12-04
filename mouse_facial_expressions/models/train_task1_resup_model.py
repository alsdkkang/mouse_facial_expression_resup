# -*- coding: utf-8 -*-
"""
ReSup: Reliable Noise Suppression Model
Features:
- Dual-network architecture (two independent networks)
- Joint training with weight exchange
- Consistency loss for noise suppression
- LDL + Temporal Aggregation backbone
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
    (Same as Advanced Model)
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
        batch_size, num_frames, _ = x.shape
        x = x + self.pos_encoding[:, :num_frames, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.layer_norm(x)
        return x


class AdvancedNetwork(nn.Module):
    """
    Single branch network: ResNet34 + Transformer + Classifier
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        
        # Spatial feature extractor
        self.spatial_encoder = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        )
        in_features = self.spatial_encoder.fc.in_features
        self.spatial_encoder.fc = nn.Identity()
        
        # Temporal aggregation
        self.temporal_encoder = TemporalTransformerEncoder(
            feature_dim=in_features,
            num_heads=8,
            num_layers=4,
            dropout=0.2
        )
        
        # Classifier (Increased dropout as per plan)
        n_classes = 2
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),  # Increased dropout
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x):
        # x: (batch, num_frames, C, H, W)
        batch_size, num_frames, C, H, W = x.shape
        
        # Reshape for spatial encoder
        x = x.view(batch_size * num_frames, C, H, W)
        spatial_features = self.spatial_encoder(x)
        
        # Reshape back
        feature_dim = spatial_features.shape[-1]
        spatial_features = spatial_features.view(batch_size, num_frames, feature_dim)
        
        # Temporal aggregation
        temporal_features = self.temporal_encoder(spatial_features)
        
        # Classification
        y_hat = self.classifier(temporal_features)
        
        return y_hat


class ReSupModel(pl.LightningModule):
    """
    Dual-network ReSup model
    """
    def __init__(self, config):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.config = config
        self.use_soft_labels = config.get('use_soft_labels', True)
        self.label_smoothing = config.get('label_smoothing', 0.2) # Increased smoothing
        
        # Two independent networks
        self.net1 = AdvancedNetwork(dropout=config.get('dropout', 0.5))
        self.net2 = AdvancedNetwork(dropout=config.get('dropout', 0.5))
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=2)
        
    def forward(self, x):
        # Inference: Average of both networks
        out1 = self.net1(x)
        out2 = self.net2(x)
        return (out1 + out2) / 2
    
    def _ldl_loss(self, pred, target):
        if self.use_soft_labels and target.dim() == 2:
            log_pred = F.log_softmax(pred, dim=1)
            loss = F.kl_div(log_pred, target, reduction='none').sum(dim=1)
        else:
            loss = F.cross_entropy(pred, target.long(), reduction='none')
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Forward both networks
        out1 = self.net1(x)
        out2 = self.net2(x)
        
        # Calculate raw losses (element-wise)
        loss1_raw = self._ldl_loss(out1, y)
        loss2_raw = self._ldl_loss(out2, y)
        
        # ReSup Strategy: Weight Exchange
        # Use prediction confidence/disagreement to weight samples
        prob1 = F.softmax(out1, dim=1)
        prob2 = F.softmax(out2, dim=1)
        
        # Disagreement map (how different are the predictions?)
        # High disagreement -> likely noisy or hard sample -> lower weight
        disagreement = torch.abs(prob1 - prob2).sum(dim=1)
        
        # Simple weighting strategy based on disagreement
        # If networks agree, we trust them more.
        # weight = 1.0 - disagreement (normalized roughly)
        weight = torch.exp(-disagreement)
        
        # Weighted loss
        loss1 = (loss1_raw * weight).mean()
        loss2 = (loss2_raw * weight).mean()
        
        # Consistency loss (force networks to agree)
        consistency_loss = F.mse_loss(prob1, prob2)
        
        # Total loss
        total_loss = loss1 + loss2 + 0.1 * consistency_loss
        
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("consistency", consistency_loss, prog_bar=True)
        
        # Accuracy (using ensemble)
        pred_ensemble = (out1 + out2) / 2
        if self.use_soft_labels and y.dim() == 2:
            y_hard = y.argmax(dim=1)
        else:
            y_hard = y.long()
            
        self.train_acc(pred_ensemble, y_hard)
        self.log("train_acc", self.train_acc, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Ensemble prediction
        out1 = self.net1(x)
        out2 = self.net2(x)
        pred = (out1 + out2) / 2
        
        # Loss
        loss = self._ldl_loss(pred, y).mean()
        self.log("val_loss", loss, prog_bar=True)
        
        # Accuracy
        if self.use_soft_labels and y.dim() == 2:
            y_hard = y.argmax(dim=1)
        else:
            y_hard = y.long()
            
        self.val_acc(pred, y_hard)
        self.log("val_acc", self.val_acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        lr = self.config['learning_rate']
        weight_decay = self.config['weight_decay']
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
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


@click.command()
@click.option("--epochs", type=click.INT, default=20)
@click.option("--learning_rate", type=click.FLOAT, default=1e-4)
@click.option("--train_batch_size", type=click.INT, default=16)
@click.option("--test_batch_size", type=click.INT, default=32)
@click.option("--weight_decay", type=click.FLOAT, default=1e-4)
@click.option("--label_smoothing", type=click.FLOAT, default=0.2) # Increased
@click.option("--dropout", type=click.FLOAT, default=0.5) # Increased
@click.option("--num_frames", type=click.INT, default=10)
@click.option("--frame_stride", type=click.INT, default=2)
@click.option("--use_soft_labels", type=click.BOOL, default=True)
@click.option("--dataset_version", type=click.STRING, default='1.1')
@click.option("--train_augmentation", type=click.STRING, default='TrivialAugmentWide')
@click.option("--seed", type=click.INT, default=97531)
@click.option("--folds", type=click.STRING, default="0,1,2,3,4")
@click.option("--accelerator", type=click.STRING, default="auto")
def main(**kwargs):
    """Train ReSup model"""
    logger = logging.getLogger(__name__)
    config = kwargs
    seed = kwargs["seed"]
    
    folds_to_run = [int(f) for f in config['folds'].split(',')]
    
    with mlflow.start_run(nested=True) as active_run:
        mlflow.log_params(kwargs)
        
        cv = Task1TemporalFolds(
            version=config['dataset_version'],
            num_frames=config['num_frames'],
            frame_stride=config['frame_stride'],
            train_augmentation=config['train_augmentation'],
            use_soft_labels=config['use_soft_labels']
        )
        
        for fold, (train_dataset, test_dataset) in enumerate(cv):
            if fold not in folds_to_run:
                continue
                
            with mlflow.start_run(nested=True):
                logger.info(f"Starting ReSup training for fold {fold}")
                
                # Seeding
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                # DataLoaders
                train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True, num_workers=4)
                test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], num_workers=4)
                
                # Model
                model = ReSupModel(config)
                
                # Callbacks
                checkpoint_dir = Path('models/checkpoints_resup')
                checkpoint_dir.mkdir(exist_ok=True, parents=True)

                checkpoint_callback = ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename=f'resup-fold{fold}-{{epoch}}-{{val_acc:.2f}}',
                    save_top_k=1,
                    monitor='val_acc',
                    mode='max'
                )
                
                # Check for existing checkpoint to resume
                existing_ckpts = list(checkpoint_dir.glob(f'resup-fold{fold}-*.ckpt'))
                resume_ckpt_path = None
                
                if existing_ckpts:
                    # Sort by epoch number to find the latest
                    def get_epoch(p):
                        match = re.search(r'epoch=(\d+)', p.name)
                        return int(match.group(1)) if match else -1
                    
                    existing_ckpts.sort(key=get_epoch, reverse=True)
                    resume_ckpt_path = str(existing_ckpts[0])
                    logger.info(f"Resuming from checkpoint: {resume_ckpt_path}")

                trainer = pl.Trainer(
                    max_epochs=config['epochs'],
                    callbacks=[checkpoint_callback],
                    accelerator=config['accelerator'],
                    logger=False
                )
                
                trainer.fit(model, train_loader, test_loader, ckpt_path=resume_ckpt_path)
                
                mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv(find_dotenv())
    main()
