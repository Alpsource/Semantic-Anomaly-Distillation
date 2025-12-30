import torch
import torch.nn as nn
from src.models.student import StudentNetwork

class CrashClassifier(nn.Module):
    def __init__(self, pre_trained_ckpt=None):
        super().__init__()
        
        # Load MobileNet Student
        self.student = StudentNetwork(pretrained=True) # Start with ImageNet weights (Faster convergence)
        
        # If you have distilled weights later, load them here. 
        # For now, we go STRAIGHT to fine-tuning to save time.
        
        self.backbone = self.student.encoder
        
        # UNFREEZE immediately. We need speed and learning.
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MobileNetV3-Small outputs 576 channels
        self.head = nn.Sequential(
            nn.Linear(576, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)