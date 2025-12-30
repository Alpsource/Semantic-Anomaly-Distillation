import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import os
from tqdm import tqdm

# CONFIG
DATA_PATH = "data/raw/finetune_train.json"
TEACHER_FEATS = "data/processed/teacher_features.pt"
BATCH_SIZE = 128 
MAX_EPOCHS = 15
LEARNING_RATE = 1e-4
ALPHA = 0.5             
DEVICE = "cuda"

class JointStudent(nn.Module):
    def __init__(self):
        super().__init__()
        # Student: MobileNetV3-Small
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.encoder = self.backbone.features # 576 channels
        
        # Projector: 576 -> 768
        self.projector = nn.Sequential(
            nn.Conv2d(576, 768, 1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)), # Flatten to vector
            nn.Flatten()
        )
        
        # Classifier Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(576, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x) # (B, 576, H, W)
        
        # Branch 1: Distillation (Project to 768 vector)
        proj_vec = self.projector(feat) # (B, 768)
        
        # Branch 2: Classification
        pooled = self.avgpool(feat).flatten(1)
        pred = self.head(pooled)
        
        return pred, proj_vec

class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, feat_path, transform=None):
        with open(json_path, 'r') as f: self.data = json.load(f)
        print("Loading cached teacher features into RAM...")
        self.teacher_feats = torch.load(feat_path) # Loads dict to RAM
        self.transform = transform
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        path = item["path"]
        
        img = Image.open(path).convert("RGB")
        label = torch.tensor(item["label"], dtype=torch.float32)
        
        # Retrieve Pre-computed Teacher Vector
        # Use zeros if missing (safety)
        t_feat = self.teacher_feats.get(path, torch.zeros(768))
        
        if self.transform: img = self.transform(img)
        return img, label, t_feat

def main():
    print("=== STARTING FAST JOINT TRAINING ===")
    
    # No Teacher Model Needed (It's Cached!)
    
    # Student
    student = JointStudent().to(DEVICE)
    
    # Data (Use MobileNet 224x224 for MAX SPEED)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = CachedDataset(DATA_PATH, TEACHER_FEATS, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    optimizer = optim.Adam(student.parameters(), lr=LEARNING_RATE)
    criterion_cls = nn.BCELoss()
    criterion_dist = nn.CosineEmbeddingLoss()
    
    for epoch in range(MAX_EPOCHS):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        total_loss = 0
        total_acc = 0
        
        student.train()
        
        for imgs, labels, t_vecs in loop:
            imgs, labels, t_vecs = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1), t_vecs.to(DEVICE)
            
            # Forward Student
            pred, s_vec = student(imgs)
            
            # Loss
            loss_c = criterion_cls(pred, labels)
            target = torch.ones(imgs.size(0)).to(DEVICE)
            loss_d = criterion_dist(s_vec, t_vecs, target)
            
            loss = loss_c + (ALPHA * loss_d)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = ((pred > 0.5) == labels).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            
            loop.set_postfix(acc=acc.item(), cls=loss_c.item(), dist=loss_d.item())
            
        print(f"Epoch {epoch+1}: Avg Loss {total_loss/len(loader):.4f} | Acc {total_acc/len(loader):.4f}")
        torch.save(student.state_dict(), f"checkpoints/joint_mobilenet_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()