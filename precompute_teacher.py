import torch
import timm
import json
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# CONFIG
DATA_PATH = "data/raw/finetune_train.json"
OUTPUT_PATH = "data/processed/teacher_features.pt"
DEVICE = "cuda"
BATCH_SIZE = 128

def main():
    print("Initializing DINOv2 for Pre-computation...")
    # Load Teacher
    model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
    model.to(DEVICE)
    model.eval()
    
    # DINO Standard Transform (518x518)
    transform = transforms.Compose([
        # transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load Data List
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
        
    feature_dict = {}
    
    print(f"Extracting features for {len(data)} images...")
    
    # Process Loop
    
    batch_paths = []
    batch_imgs = []
    
    with torch.no_grad():
        for item in tqdm(data):
            path = item['path']
            try:
                img = Image.open(path).convert("RGB")
                img_t = transform(img)
                
                batch_paths.append(path)
                batch_imgs.append(img_t)
                
                if len(batch_imgs) >= BATCH_SIZE:
                    # Run Batch
                    input_tensor = torch.stack(batch_imgs).to(DEVICE)
                    
                    # Get features (CLS + Patches)
                    # We usually want the Global Average or CLS for simple distillation
                    # DINOv2: forward_features returns (B, N, 768)
                    out = model.forward_features(input_tensor)
                    
                    # STRATEGY: Save the MEAN vector (B, 768). 
                    # It's small (RAM efficient) and captures the "Gist".
                    # Saving full patch tokens (1370x768) would kill disk space (100GB+).
                    feats = out.mean(dim=1).cpu() # (B, 768)
                    
                    for p, f in zip(batch_paths, feats):
                        feature_dict[p] = f
                    
                    batch_paths = []
                    batch_imgs = []
                    
            except Exception as e:
                print(f"Error on {path}: {e}")
                continue

        # Process remaining
        if batch_imgs:
            input_tensor = torch.stack(batch_imgs).to(DEVICE)
            out = model.forward_features(input_tensor)
            feats = out.mean(dim=1).cpu()
            for p, f in zip(batch_paths, feats):
                feature_dict[p] = f

    print(f"Saving {len(feature_dict)} features to {OUTPUT_PATH}...")
    torch.save(feature_dict, OUTPUT_PATH)
    print("Done! We can now delete the Teacher from memory.")

if __name__ == "__main__":
    main()