import os
import glob
import json
import shutil
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_FRAMES_DIR = "data/raw/frames"
OUTPUT_DIR = "data/processed"

# Metadata files
SPLIT_FILES = {
    "train": "data/raw/train_split.txt",
    "test": "data/raw/val_split.txt"
}

# CHANGED: Native DINOv2 Resolution
TARGET_SIZE = (518, 518) 
SAMPLE_RATE = 1  # Keep 10 FPS
# ---------------------

def load_split_list(txt_path):
    if not os.path.exists(txt_path):
        print(f"Warning: Split file {txt_path} not found.")
        return set()
    with open(txt_path, 'r') as f:
        ids = {line.strip() for line in f if line.strip()}
    return ids

def process_and_save(src_path, dest_path):
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            # High-quality resize
            img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            img_resized.save(dest_path, quality=95)
    except Exception as e:
        print(f"Error processing {src_path}: {e}")

def parse_metadata_to_json(metadata_path, output_path):
    # (Same as before, just copying helper function)
    if not os.path.exists(metadata_path): return
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    clean_db = {}
    for vid_id, info in data.items():
        intervals = []
        # Handle "anomaly_start/end" format
        if "anomaly_start" in info and "anomaly_end" in info:
             intervals.append([info["anomaly_start"], info["anomaly_end"]])
        # Handle "intervals" format
        elif "intervals" in info:
            intervals = info["intervals"]
            
        clean_db[vid_id] = {
            "type": info.get("anomaly_class", "unknown"),
            "intervals": intervals
        }
        
    with open(output_path, 'w') as f:
        json.dump(clean_db, f, indent=4)
    print(f"Generated Ground Truth: {output_path}")

def run_processing():
    train_ids = load_split_list(SPLIT_FILES["train"])
    test_ids = load_split_list(SPLIT_FILES["test"])
    
    video_folders = sorted(glob.glob(os.path.join(SOURCE_FRAMES_DIR, "*")))
    print(f"Found {len(video_folders)} videos in {SOURCE_FRAMES_DIR}")
    
    processed_count = 0
    
    for video_folder in tqdm(video_folders):
        video_id = os.path.basename(video_folder)
        
        if video_id in train_ids:
            split = "train"
        elif video_id in test_ids:
            split = "test"
        else:
            continue
            
        save_dir = os.path.join(OUTPUT_DIR, split, video_id)
        
        # Optimization: Skip if already done
        if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
            continue
        
        os.makedirs(save_dir, exist_ok=True)
        
        src_images_dir = os.path.join(video_folder, "images")
        if not os.path.exists(src_images_dir):
            src_images_dir = video_folder
            
        images = sorted(glob.glob(os.path.join(src_images_dir, "*.jpg")))
        selected_images = images[::SAMPLE_RATE]
        
        for img_path in selected_images:
            frame_name = os.path.basename(img_path)
            dest_path = os.path.join(save_dir, frame_name)
            process_and_save(img_path, dest_path)
            
        processed_count += 1

    print(f"Processed {processed_count} videos to {TARGET_SIZE}.")
    
    # Generate GT files
    parse_metadata_to_json("data/raw/metadata_val.json", "data/test_ground_truth.json")

if __name__ == "__main__":
    run_processing()