import os
import glob
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class DoTADataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, clean_intervals_path=None):
        """
        Args:
            root_dir: Path to 'data/processed'
            split: 'train' or 'test'
            transform: PyTorch transforms
            clean_intervals_path: Path to the generated JSON (only for train)
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        
        # 1. Load Clean Intervals (Only if provided)
        clean_map = None
        if clean_intervals_path and os.path.exists(clean_intervals_path):
            print(f"Loading clean intervals from {clean_intervals_path}...")
            with open(clean_intervals_path, 'r') as f:
                clean_map = json.load(f)
        
        # 2. Collect Images
        video_folders = glob.glob(os.path.join(self.root_dir, "*"))
        print(f"Scanning {len(video_folders)} folders for {split}...")
        
        for video_folder in video_folders:
            video_id = os.path.basename(video_folder)
            
            # If we have a clean map, get valid intervals for this video
            # If video_id is not in map, default to None (which we handle below)
            intervals = clean_map.get(video_id, None) if clean_map else None
            
            # Get all images in folder
            # Handles both direct .jpg or nested 'images/*.jpg' structure if present
            if os.path.exists(os.path.join(video_folder, "images")):
                images = sorted(glob.glob(os.path.join(video_folder, "images", "*.jpg")))
            else:
                images = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
            
            for img_path in images:
                # If this is TEST set, we take everything (we need anomalies to test!)
                if split == "test":
                    self.image_paths.append(img_path)
                    continue

                # If TRAIN set, apply filtering
                # If no clean_map provided, take everything (Legacy mode)
                if not clean_map:
                    self.image_paths.append(img_path)
                    continue

                # STRICT FILTERING logic
                if intervals is None:
                    # Video not in clean list? Skip it to be safe, 
                    # OR include it if you trust your generation script.
                    # Let's include it if intervals is empty list [] (means whole video bad)
                    # But get() returns None if missing. 
                    # If the key exists but is empty list [], it means "Drop Video"
                    pass 
                else:
                    # Parse frame number
                    try:
                        fname = os.path.splitext(os.path.basename(img_path))[0]
                        # Handle "frame_001" or "001"
                        if "frame_" in fname:
                            frame_num = int(fname.split("_")[-1])
                        else:
                            frame_num = int(fname)
                            
                        # Check if frame is inside ANY valid interval
                        is_safe = False
                        for start, end in intervals:
                            if start <= frame_num <= end:
                                is_safe = True
                                break
                        
                        if is_safe:
                            self.image_paths.append(img_path)
                    except:
                        pass # Skip files with weird names

        print(f"Final dataset size for {split}: {len(self.image_paths)} frames.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy tensor to prevent crash
            return torch.zeros(3, 224, 224)