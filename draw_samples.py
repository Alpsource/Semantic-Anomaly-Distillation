import matplotlib.pyplot as plt
import os
from PIL import Image
import random

# CONFIG
DATA_ROOT = "data/processed/test" # Or train folder

def draw_sample_grid():
    if not os.path.exists(DATA_ROOT):
        print(f"Path {DATA_ROOT} not found. Creating dummy placeholder.")
        # Create a dummy image if path is wrong
        fig, ax = plt.subplots(1, 1)
        ax.text(0.5, 0.5, "Dataset Samples Placeholder", ha='center')
        plt.savefig('sample_frames.png')
        return

    # Try to find images
    all_images = []
    for root, dirs, files in os.walk(DATA_ROOT):
        for f in files:
            if f.endswith(('.jpg', '.png')):
                all_images.append(os.path.join(root, f))
                if len(all_images) > 100: break # Stop searching
    
    if len(all_images) < 4:
        print("Not enough images found.")
        return

    # Pick 4 random images
    samples = random.sample(all_images, 4)
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    titles = ["Scenario A", "Scenario B", "Scenario C", "Scenario D"]
    
    for ax, img_path, title in zip(axes, samples, titles):
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        except:
            pass

    plt.tight_layout()
    plt.savefig('sample_frames.png', dpi=300)
    print("Saved sample_frames.png")

if __name__ == "__main__":
    draw_sample_grid()