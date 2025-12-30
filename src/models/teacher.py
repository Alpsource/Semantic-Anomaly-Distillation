import torch
import torch.nn as nn
import timm

class TeacherDINO(nn.Module):
    def __init__(self, model_name='vit_base_patch14_dinov2.lvd142m', pretrained=True):
        super().__init__()
        print(f"Loading Teacher: {model_name} (DINOv2)...")
        
        # Load DINOv2 with dynamic size support
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True  # <--- Prevents resolution errors
        )
        
        # Freeze strictly
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, x):
        """
        Returns: (patch_tokens, cls_token)
        """
        with torch.no_grad():
            features = self.model.forward_features(x)
            
            # DINOv2 Output is (Batch, Num_Tokens, 768)
            # Index 0 is CLS, 1: is Patches
            patch_tokens = features[:, 1:] 
            cls_token = features[:, 0]
            
            return patch_tokens, cls_token