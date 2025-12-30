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
BATCH_SIZE = 128        # MobileNet is tiny, so we can use a large batch size
MAX_EPOCHS = 15         # Give it enough time to converge
LEARNING_RATE = 1e-4    # Standard fine-tuning rate
DEVICE = "cuda"


class BaselineMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Load Standard ImageNet MobileNetV3-Small
        print("Loading Standard ImageNet MobileNetV3-Small (Unfrozen)...")
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        base_model = models.mobilenet_v3_small(weights=weights)
        
        # Extract Feature Extractor (The "Encoder")
        # MobileNetV3 separates the network into 'features' (CNN) and 'classifier'
        self.backbone = base_model.features
        
        # UNFREEZE
        # We allow the baseline to learn "Crash Features" from scratch
        # using the labeled data.
        for param in self.backbone.parameters():
            param.requires_grad = True  
            
        # Classifier Head (Same as Distilled Classifier)
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
        x = self.backbone(x)     # (B, 576, 16, 16)
        x = self.avgpool(x)      # (B, 576, 1, 1)
        x = torch.flatten(x, 1)  # (B, 576)
        return self.head(x)      # (B, 1)

# DATASET
class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f: self.data = json.load(f)
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["path"]).convert("RGB")
        label = torch.tensor(item["label"], dtype=torch.float32)
        if self.transform: img = self.transform(img)
        return img, label

def main():
    # Standard transforms (Match Distillation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    dataset = FinetuneDataset(DATA_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = BaselineMobileNet().to(DEVICE)
    
    # Optimizer updates EVERYTHING (Backbone + Head)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    print(f"Starting MobileNet Baseline Training on {len(dataset)} frames...")
    
    for epoch in range(MAX_EPOCHS):
        loop = tqdm(loader, desc=f"Baseline Epoch {epoch+1}")
        total_acc = 0
        total_loss = 0
        
        model.train() # BatchNorm active
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            acc = ((preds > 0.5) == labels).float().mean()
            total_acc += acc.item()
            total_loss += loss.item()
            
            loop.set_postfix(loss=loss.item(), acc=acc.item())
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f} | Acc {total_acc/len(loader):.4f}")
        torch.save(model.state_dict(), f"checkpoints/baseline_mobilenet_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()