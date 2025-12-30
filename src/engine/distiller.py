import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.teacher import TeacherDINO
from src.models.student import StudentNetwork

class AnomalyDistiller(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, max_epochs=20, student_backbone='mobilenetv3_small_100'): # Increased LR to 1e-3
        super().__init__()
        self.save_hyperparameters()
        
        self.teacher = TeacherDINO()
        # Pass the backbone string to the student
        self.student = StudentNetwork(model_name=student_backbone)
        
        # REMOVE MSELoss
        # We will use explicit Cosine Distance calculation manually
        
    def forward(self, x):
        _, score = self.student(x)
        return score

    def training_step(self, batch, batch_idx):
        x = batch
        
        # 1. Teacher Forward (Frozen)
        self.teacher.eval()
        with torch.no_grad():
            t_patch_tokens, _ = self.teacher(x) # (B, 256, 768)
        
        # 2. Student Forward
        s_patch_tokens, _ = self.student(x) # (B, N_s, 768)
        
        # 3. Dynamic Reshape & Upsample
        B, N_t, C = t_patch_tokens.shape
        # Calculate Teacher Grid Size (sqrt(256) = 16)
        H_t = W_t = int(N_t ** 0.5) 

        B, N_s, C = s_patch_tokens.shape
        # Calculate Student Grid Size (e.g., sqrt(196) = 14 or sqrt(49) = 7)
        H_s = W_s = int(N_s ** 0.5) 

        # View as spatial grids
        t_feats = t_patch_tokens.transpose(1, 2).view(B, C, H_t, W_t)
        s_feats = s_patch_tokens.transpose(1, 2).view(B, C, H_s, W_s)
        
        # Upsample Student to match Teacher (e.g., 14x14 -> 16x16)
        s_feats_up = F.interpolate(s_feats, size=(H_t, W_t), mode='bilinear', align_corners=False)
        
        # 4. Cosine Similarity Map
        t_norm = F.normalize(t_feats, dim=1)
        s_norm = F.normalize(s_feats_up, dim=1)
        
        # Calculate similarity per pixel
        similarity_map = (t_norm * s_norm).sum(dim=1)
        
        # 5. Loss Map (1 - similarity)
        loss_map = 1.0 - similarity_map
        
        # --- SKY MASKING (Top 30%) ---
        mask = torch.ones_like(loss_map)
        rows_to_mask = int(H_t * 0.3) # Dynamic 30% calculation
        mask[:, :rows_to_mask, :] = 0.0
        
        # Weighted Loss
        masked_loss = loss_map * mask
        loss = masked_loss.sum() / (mask.sum() + 1e-6) # Add epsilon for safety
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Increased LR to 1e-3 because Cosine loss gradients can be smaller
        optimizer = AdamW(self.student.parameters(), lr=self.hparams.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]