import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from RealESRGAN import RealESRGAN
from google.colab import files
import time

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
print("\nLoading RealESRGAN model (this may take a moment)...")
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)
print("Model ready!")

# Directories
os.makedirs('input_videos', exist_ok=True)
os.makedirs('output_videos', exist_ok=True)

# Upload
print("\nUpload video (recommend <10 seconds for testing)...")
uploaded = files.upload()

if not uploaded:
    raise ValueError("No file uploaded")

video_name = list(uploaded.keys())[0]
input_path = f"input_videos/{video_name}"
os.rename(video_name, input_path)

# Get video info
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"\nVideo: {video_name}")
print(f"  {width}x{height} @ {fps}fps")
print(f"  {total_frames} frames ({total_frames/fps:.1f}s)")
print(f"  Output: {width*4}x{height*4}")

# Estimate time
estimated_time = total_frames * (2 if device.type == 'cuda' else 5)
print(f"\nEstimated time: {estimated_time//60}m {estimated_time%60}s")
print("(This will take a while - AI upscaling is slow)\n")

# Process video
output_name = f"{os.path.splitext(video_name)[0]}_realesrgan_4x.mp4"
output_path = f"output_videos/{output_name}"

cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width*4, height*4))

start_time = time.time()
frame_num = 0

print("Processing frames...")
pbar = tqdm(total=total_frames, desc="Progress", ncols=100)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # Convert and upscale
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    sr_img = model.predict(pil_img)
    sr_frame = cv2.cvtColor(np.array(sr_img), cv2.COLOR_RGB2BGR)

    # Write
    out.write(sr_frame)

    # Update progress every 10 frames
    if frame_num % 10 == 0:
        elapsed = time.time() - start_time
        fps_proc = frame_num / elapsed if elapsed > 0 else 0
        remaining = (total_frames - frame_num) / fps_proc if fps_proc > 0 else 0
        pbar.set_postfix({
            'fps': f'{fps_proc:.1f}',
            'eta': f'{int(remaining//60)}m{int(remaining%60)}s'
        })

    pbar.update(1)

pbar.close()
cap.release()
out.release()

elapsed_total = time.time() - start_time
print(f"\nCompleted in {int(elapsed_total//60)}m {int(elapsed_total%60)}s")
print(f"Average: {total_frames/elapsed_total:.2f} fps")

# File sizes
input_size = os.path.getsize(input_path) / (1024**2)
output_size = os.path.getsize(output_path) / (1024**2)
print(f"\nFile sizes:")
print(f"  Input: {input_size:.1f} MB")
print(f"  Output: {output_size:.1f} MB")

# Download
print(f"\nDownloading {output_name}...")
files.download(output_path)
print("\nDone!")import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from RealESRGAN import RealESRGAN
from google.colab import files
import time

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
print("\nLoading RealESRGAN model (this may take a moment)...")
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)
print("Model ready!")

# Directories
os.makedirs('input_videos', exist_ok=True)
os.makedirs('output_videos', exist_ok=True)

# Upload
print("\nUpload video (recommend <10 seconds for testing)...")
uploaded = files.upload()

if not uploaded:
    raise ValueError("No file uploaded")

video_name = list(uploaded.keys())[0]
input_path = f"input_videos/{video_name}"
os.rename(video_name, input_path)

# Get video info
cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"\nVideo: {video_name}")
print(f"  {width}x{height} @ {fps}fps")
print(f"  {total_frames} frames ({total_frames/fps:.1f}s)")
print(f"  Output: {width*4}x{height*4}")

# Estimate time
estimated_time = total_frames * (2 if device.type == 'cuda' else 5)
print(f"\nEstimated time: {estimated_time//60}m {estimated_time%60}s")
print("(This will take a while - AI upscaling is slow)\n")

# Process video
output_name = f"{os.path.splitext(video_name)[0]}_realesrgan_4x.mp4"
output_path = f"output_videos/{output_name}"

cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width*4, height*4))

start_time = time.time()
frame_num = 0

print("Processing frames...")
pbar = tqdm(total=total_frames, desc="Progress", ncols=100)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # Convert and upscale
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    sr_img = model.predict(pil_img)
    sr_frame = cv2.cvtColor(np.array(sr_img), cv2.COLOR_RGB2BGR)

    # Write
    out.write(sr_frame)

    # Update progress every 10 frames
    if frame_num % 10 == 0:
        elapsed = time.time() - start_time
        fps_proc = frame_num / elapsed if elapsed > 0 else 0
        remaining = (total_frames - frame_num) / fps_proc if fps_proc > 0 else 0
        pbar.set_postfix({
            'fps': f'{fps_proc:.1f}',
            'eta': f'{int(remaining//60)}m{int(remaining%60)}s'
        })

    pbar.update(1)

pbar.close()
cap.release()
out.release()

elapsed_total = time.time() - start_time
print(f"\nCompleted in {int(elapsed_total//60)}m {int(elapsed_total%60)}s")
print(f"Average: {total_frames/elapsed_total:.2f} fps")

# File sizes
input_size = os.path.getsize(input_path) / (1024**2)
output_size = os.path.getsize(output_path) / (1024**2)
print(f"\nFile sizes:")
print(f"  Input: {input_size:.1f} MB")
print(f"  Output: {output_size:.1f} MB")

# Download
print(f"\nDownloading {output_name}...")
files.download(output_path)
print("\nDone!")
