import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import json
import os
import matplotlib.pyplot as plt
import random

# CONFIG
TEST_ROOT = "data/processed/test"
GT_PATH = "data/test_ground_truth.json"
BASELINE_CKPT = "checkpoints/baseline_mobilenet_epoch15.pth"
DISTILLED_CKPT = "checkpoints/joint_mobilenet_epoch15.pth"
DEVICE = "cuda"

# MODELS (Same as Training)
class BaselineMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)
        self.backbone = base.features
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
        return self.head(self.avgpool(self.backbone(x)).flatten(1))

class JointStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        self.encoder = self.backbone.features
        
        self.projector = nn.Sequential(
            nn.Conv2d(576, 768, 1, bias=False), 
            nn.BatchNorm2d(768), 
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)), 
            nn.Flatten()
        )
        
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
        # We only need the classification head for visualization
        feat = self.encoder(x)
        return self.head(self.avgpool(feat).flatten(1))

# DATASET
class VisualDataset(Dataset):
    def __init__(self, root, gt_path):
        self.samples = []
        with open(gt_path) as f: gt = json.load(f)
        
        print(f"Scanning {root} for qualitative samples...")
        for root_dir, dirs, files in os.walk(root):
            vid_id = os.path.basename(root_dir)
            if vid_id not in gt: continue
            
            for f in files:
                if f.endswith('.jpg'):
                    try:
                        fid = int(os.path.splitext(f)[0])
                        label = 0.0
                        # Check intervals
                        for s, e in gt[vid_id]['intervals']:
                            if s <= fid <= e: 
                                label = 1.0
                                break
                        self.samples.append((os.path.join(root_dir, f), label))
                    except: pass
        
        print(f"Found {len(self.samples)} samples.")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def main():
    # Load Models
    print("Loading Baseline...")
    base = BaselineMobileNet().to(DEVICE).eval()
    base.load_state_dict(torch.load(BASELINE_CKPT))
    
    print("Loading Distilled Model...")
    dist = JointStudent().to(DEVICE).eval()
    dist.load_state_dict(torch.load(DISTILLED_CKPT))
    
    ds = VisualDataset(TEST_ROOT, GT_PATH)
    
    if len(ds) == 0:
        print("No images found. Check path.")
        return

    print("Searching for 'Distillation Wins' (Baseline Missed, Ours Detected)...")
    
    wins = []
    
    # Shuffle indices to get random frames
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    # Check up to 5000 random frames
    count = 0
    for i in indices:
        count += 1
        if count > 5000: break
        
        path, label = ds.samples[i]
        
        # We ONLY want Accident frames (label=1.0)
        if label < 0.5: continue 
        
        try:
            img = Image.open(path).convert("RGB")
            img_t = ds.transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                p_base = base(img_t).item()
                p_dist = dist(img_t).item()
            
            # CRITERIA: 
            # 1. It IS a crash (Label=1)
            # 2. Baseline thinks it's Normal (p < 0.3) -> MISS
            # 3. Distilled thinks it's Crash (p > 0.6) -> HIT
            if p_base < 0.3 and p_dist > 0.6:
                wins.append((path, p_base, p_dist))
                print(f"Found Win! Base: {p_base:.2f}, Ours: {p_dist:.2f} -> {path}")
                if len(wins) >= 3: break
        except:
            continue
            
    if not wins:
        print("No clear wins found in random sample. Try running again.")
        return

    # Plotting
    print(f"Plotting {len(wins)} results...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Handle case where we found fewer than 3, but it never happens :)
    if len(wins) < 3:
        axes = axes[:len(wins)]
        
    if len(wins) == 1: axes = [axes]

    for ax, (path, pb, pd) in zip(axes, wins):
        img = Image.open(path)
        ax.imshow(img)
        ax.axis('off')
        
        # Title with background to ensure readability
        title = f"Ground Truth: CRASH\nBaseline: {pb:.2f} (Miss)\nOurs: {pd:.2f} (Detected)"
        ax.set_title(title, color='darkgreen', fontweight='bold', fontsize=11, 
                     backgroundcolor='white', pad=10)
        
    # Add specific margin at the top for the titles, otherwise, titles can't be read
    plt.tight_layout(rect=[0, 0, 1, 0.90]) 
    
    plt.savefig('qualitative_results.png', dpi=300)
    print("Saved qualitative_results.png")

if __name__ == "__main__":
    main()