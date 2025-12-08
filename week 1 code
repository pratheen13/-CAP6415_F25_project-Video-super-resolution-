# 1. Install dependencies
!pip install torch torchvision opencv-python ffmpeg-python tqdm scikit-image lpips Pillow matplotlib --quiet

# 2. Imports
import os, cv2
from tqdm import tqdm
from google.colab import files
import matplotlib.pyplot as plt

# 3. Folder structure (auto-created in /content)
base_dir = "/content/video-super-resolution"
os.makedirs(f"{base_dir}/demo_videos/input", exist_ok=True)
os.makedirs(f"{base_dir}/demo_videos/output", exist_ok=True)

print(f"Project folder created at: {base_dir}")

# 4. Upload a small sample video
print("Upload a small .mp4 file (5–10 seconds) when prompted...")
uploaded = files.upload()

# Move uploaded file to input folder
for name in uploaded.keys():
    os.rename(name, f"{base_dir}/demo_videos/input/{name}")
sample_video = f"{base_dir}/demo_videos/input/{name}"
print(f"Using: {sample_video}")

# 5. Bicubic upscaling pipeline (no frame saving)
def upscale_video(input_path, output_path, scale=2):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error: Cannot open input video.")
    
    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define output writer
    out_width, out_height = width * scale, height * scale
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    print(f"Upscaling video ×{scale} ({width}x{height} → {out_width}x{out_height})")

    for _ in tqdm(range(frame_count), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break
        upscaled = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
        out.write(upscaled)

    cap.release()
    out.release()
    print(f"[OK] Upscaled video saved to: {output_path}")

# 6. Run the upscaling
output_video = f"{base_dir}/demo_videos/output/sample_upscaled.mp4"
upscale_video(sample_video, output_video, scale=2)
print("\nBaseline upscale complete!")

# 7. Optional: show a comparison frame
def show_comparison(input_path, output_path, frame_id=10):
    cap1 = cv2.VideoCapture(input_path)
    cap2 = cv2.VideoCapture(output_path)

    cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    cap1.release()
    cap2.release()

    if not (ret1 and ret2):
        print("Frame not found.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original")
    ax[1].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Upscaled ×2")
    for a in ax:
        a.axis("off")
    plt.show()

# Uncomment to preview a comparison:
# show_comparison(sample_video, output_video, frame_id=10)

# 8. Download the upscaled video
files.download(output_video)
