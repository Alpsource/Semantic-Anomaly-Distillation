import torch
import torch.nn as nn
import torchvision.models as models

class StudentNetwork(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        print(f"Initializing Student: MobileNetV3-Small (Edge Mode)")
        
        # Load MobileNetV3 Small (The fastest modern CNN)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        
        # MobileNet structure is different. Features come from 'features' block.
        # We cut it before the final pooling/classifier.
        # Output channels of last conv layer in Small is 576.
        self.encoder = self.backbone.features
        
        student_dim = 576 
        teacher_dim = 768 
        
        # Projector: Map 576 -> 768 (Match DINO)
        self.projector = nn.Sequential(
            nn.Conv2d(student_dim, teacher_dim, 1, bias=False),
            nn.BatchNorm2d(teacher_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        # Extract features
        x = self.encoder(x) # (B, 576, 16, 16) for 518x518 input
        
        # Project to Teacher Dim
        projected = self.projector(x) # (B, 768, 16, 16)
        
        # Return only projected features for Distillation
        # Flatten: (B, 768, N) -> Transpose (B, N, 768)
        return projected.flatten(2).transpose(1, 2), None