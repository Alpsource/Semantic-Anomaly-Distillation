import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import os
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# CONFIG
TEST_ROOT = "data/processed/test"
GT_PATH = "data/test_ground_truth.json"
BASELINE_CKPT = "checkpoints/baseline_mobilenet_epoch15.pth"
DISTILLED_CKPT = "checkpoints/joint_mobilenet_epoch15.pth"
DEVICE = "cuda"


# MODEL DEFINITIONS 
class BaselineMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)
        self.backbone = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(576, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.head(self.avgpool(self.backbone(x)).flatten(1))
        return x

class JointStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        self.encoder = self.backbone.features
        self.projector = nn.Sequential(
            nn.Conv2d(576, 768, 1, bias=False), nn.BatchNorm2d(768), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(576, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        feat = self.encoder(x)
        return self.head(self.avgpool(feat).flatten(1)), None

# RECURSIVE TEST DATASET
class RealTestDataset(Dataset):
    def __init__(self, root_dir, gt_json, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        
        print(f"Loading Ground Truth from {gt_json}...")
        with open(gt_json, 'r') as f:
            self.gt_data = json.load(f)
            
        print(f"Recursively scanning images in {root_dir}...")
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    # Parent folder name is the Video ID (e.g., 0qfbmt4G8Rw_001201)
                    video_id = os.path.basename(root)
                    full_path = os.path.join(root, file)
                    
                    # Calculate label based on video_id and frame number
                    label = self.get_label(video_id, file)
                    self.data.append((full_path, label))
            
        print(f"Found {len(self.data)} test frames.")

    def get_label(self, video_id, filename):
        """
        video_id: '0qfbmt4G8Rw_001201' (From folder name)
        filename: '00003.jpg'
        """
        # Check if we have info for this video
        if video_id not in self.gt_data:
            return 0.0 # Default to normal if video not in GT
            
        # Extract Frame Number
        # Remove extension and leading zeros
        try:
            frame_str = os.path.splitext(filename)[0]
            frame_no = int(frame_str)
        except ValueError:
            return 0.0
            
        # Check Intervals
        intervals = self.gt_data[video_id].get('intervals', [])
        for start, end in intervals:
            if start <= frame_no <= end:
                return 1.0 # CRASH
        
        return 0.0 # NORMAL

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data[idx]
        
        try:
            img = Image.open(path).convert("RGB")
        except:
            print(f"Warning: Corrupt file {path}")
            img = Image.new('RGB', (224, 224))
            
        if self.transform: img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

def evaluate(model, loader, name):
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"Starting inference for {name}...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            
            if isinstance(outputs, tuple): outputs = outputs[0]
                
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Metrics
    preds_bin = [1 if p > 0.5 else 0 for p in all_preds]
    
    # Calculate Core Metrics
    acc = accuracy_score(all_labels, preds_bin)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5 
    f1 = f1_score(all_labels, preds_bin)
    cm = confusion_matrix(all_labels, preds_bin)
    
    # Calculate Recall (Sensitivity) = TP / (TP + FN)
    # confusion_matrix returns [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print(f"\n=== FINAL RESULTS: {name} ===")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Recall:   {recall*100:.2f}%")
    print(f"  AUC:      {auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    # Return a dictionary for easy plotting
    return {
        "name": name,
        "acc": acc * 100,
        "recall": recall * 100,
        "auc": auc * 100, # scaled to % for graph
        "f1": f1 * 100,   # scaled to % for graph
        "cm": cm
    }

def plot_comparison_results(res_base, res_dist):
    # Set style
    sns.set_style("whitegrid")
    
    # SAFETY METRICS BAR CHART
    metrics = ['Recall (Safety)', 'F1-Score', 'AUC']
    baseline_vals = [res_base['recall'], res_base['f1'], res_base['auc']]
    distilled_vals = [res_dist['recall'], res_dist['f1'], res_dist['auc']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (MobileNet)', color='#ff9999')
    rects2 = ax.bar(x + width/2, distilled_vals, width, label='Ours (Distilled)', color='#66b3ff')
    
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Critical Safety Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%', 
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('metric_comparison.png', dpi=300)
    print("\n[+] Saved metric_comparison.png")

    # CONFUSION MATRICES
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Helper to plot heatmap
    def plot_cm(ax, res, cmap):
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap=cmap, ax=ax, cbar=False, annot_kws={"size": 14})
        ax.set_title(f"{res['name']}\nRecall: {res['recall']:.1f}%", fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xticklabels(['Normal', 'Crash'], fontsize=11)
        ax.set_yticklabels(['Normal', 'Crash'], fontsize=11)

    plot_cm(axes[0], res_base, 'Reds')
    plot_cm(axes[1], res_dist, 'Blues')
    
    plt.tight_layout()
    plt.savefig('cm_comparison.png', dpi=300)
    print("[+] Saved cm_comparison.png")

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = RealTestDataset(TEST_ROOT, GT_PATH, transform=transform)
    
    if len(dataset) == 0:
        print("Dataset empty. Exiting.")
        return

    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Store results here
    res_base = None
    res_dist = None

    # Baseline
    if os.path.exists(BASELINE_CKPT):
        print("\nLoading Baseline...")
        baseline = BaselineMobileNet().to(DEVICE)
        baseline.load_state_dict(torch.load(BASELINE_CKPT))
        res_base = evaluate(baseline, loader, "Baseline MobileNet")
    else:
        print("Baseline Checkpoint not found! Cannot plot comparison.")
    
    # Distilled
    if os.path.exists(DISTILLED_CKPT):
        print("\nLoading Distilled Model...")
        distilled = JointStudent().to(DEVICE)
        distilled.load_state_dict(torch.load(DISTILLED_CKPT))
        res_dist = evaluate(distilled, loader, "Distilled MobileNet")
    else:
        print("Distilled Checkpoint not found! Cannot plot comparison.")

    # Plot if both exist
    if res_base and res_dist:
        print("\nGenerating Comparison Plots...")
        plot_comparison_results(res_base, res_dist)
    else:
        print("\nSkipping plots because one or both models failed to evaluate.")

if __name__ == "__main__":
    main()