import gradio as gr
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import glob
from torchvision import transforms, models
from PIL import Image

# CONFIG
CHECKPOINT_PATH = "checkpoints/joint_mobilenet_epoch15.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DATA_DIR = "data/processed/test"  # Where test frames are
EXAMPLES_DIR = "assets/examples"       # Where we will save example videos

# MODEL DEFINITION (Same as Training)
class JointStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        self.encoder = self.backbone.features
        # Projector
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
        feat = self.encoder(x)
        return self.head(self.avgpool(feat).flatten(1))

# LOAD MODEL
print(f"Loading model from {CHECKPOINT_PATH}...")
model = JointStudent().to(DEVICE)
try:
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure we have run 'train_distillation.py' first.")
model.eval()

# PREPROCESSING
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# GENERATE EXAMPLES
def create_example_videos():
    """Creates short video clips from test frames to use as Gradio examples."""
    if not os.path.exists(EXAMPLES_DIR):
        os.makedirs(EXAMPLES_DIR)
    
    # Check if we already have examples
    if len(os.listdir(EXAMPLES_DIR)) > 0:
        return [os.path.join(EXAMPLES_DIR, f) for f in os.listdir(EXAMPLES_DIR) if f.endswith('.mp4')]

    print("Generating example videos from test set...")
    example_videos = []
    
    # Find up to 10 folders in test set
    test_folders = [f for f in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, f))][:10]
    
    for i, folder in enumerate(test_folders):
        folder_path = os.path.join(TEST_DATA_DIR, folder)
        frames = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        if len(frames) < 10: continue
        
        # Take all frames, reduce if needed
        clip_frames = frames
        
        # Determine size from first frame
        first_img = cv2.imread(clip_frames[0])
        h, w, _ = first_img.shape
        
        save_path = os.path.join(EXAMPLES_DIR, f"example_{i}.mp4")
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        
        for f_path in clip_frames:
            out.write(cv2.imread(f_path))
        out.release()
        example_videos.append(save_path)
        
    return example_videos

# INFERENCE
def process_video(video_path):
    if not video_path:
        return None, "Please upload a video."
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 10  # Default fallback
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temp output file
    output_path = "output_demo.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_count = 0
    anomaly_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Preprocess
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        
        # 2. Inference
        with torch.no_grad():
            score = model(input_tensor).item()
            
        # 3. Visualize
        label = "NORMAL"
        color = (0, 255, 0) # Green
        
        if score > 0.5:
            label = f"ACCIDENT ({score:.2f})"
            color = (0, 0, 255) # Red
            anomaly_count += 1
        else:
            label = f"NORMAL ({score:.2f})"
        
        # Draw Label
        cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        out.write(frame)
        frame_count += 1
        
    cap.release()
    out.release()
    
    summary = f"Processed {frame_count} frames. Detected anomalies in {anomaly_count} frames."
    return output_path, summary

# UI
examples = create_example_videos()

with gr.Blocks(title="Accident Detection Demo") as demo:
    gr.Markdown("# 🚗 Real-Time Accident Detection")
    gr.Markdown("Upload a dashcam video (or use an example below). The Distilled MobileNetV3 model will process it frame-by-frame.")
    
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video", format="mp4")
            run_btn = gr.Button("Detect Accidents", variant="primary")
            
            if examples:
                gr.Examples(examples=examples, inputs=input_video, label="Try an Example from Test Set")
                
        with gr.Column():
            output_video = gr.Video(label="Processed Result")
            status = gr.Textbox(label="Status")
            
    run_btn.click(process_video, inputs=input_video, outputs=[output_video, status])

if __name__ == "__main__":
    demo.launch(share=True)