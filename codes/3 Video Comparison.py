# Video Comparison & Analysis Tool
# Compare two videos frame-by-frame with detailed metrics

# Install dependencies
!pip install opencv-python numpy Pillow matplotlib scikit-image lpips torch --quiet

import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load LPIPS model for perceptual quality
print("Loading LPIPS model for perceptual quality assessment...")
lpips_model = lpips.LPIPS(net='alex').to(device)
print("Ready!")

# Create directories
os.makedirs('videos', exist_ok=True)
os.makedirs('analysis', exist_ok=True)

# Upload first video
print("\nUpload FIRST video (e.g., original or lower quality)...")
uploaded1 = files.upload()
video1_name = list(uploaded1.keys())[0]
video1_path = f"videos/{video1_name}"
os.rename(video1_name, video1_path)
print(f"Video 1: {video1_name}")

# Upload second video
print("\nUpload SECOND video (e.g., upscaled or higher quality)...")
uploaded2 = files.upload()
video2_name = list(uploaded2.keys())[0]
video2_path = f"videos/{video2_name}"
os.rename(video2_name, video2_path)
print(f"Video 2: {video2_name}")

# Upload third video
print("\nUpload THIRD video (e.g., another upscale method or reference)...")
uploaded3 = files.upload()
video3_name = list(uploaded3.keys())[0]
video3_path = f"videos/{video3_name}"
os.rename(video3_name, video3_path)
print(f"Video 3: {video3_name}")

# Get video info
def get_video_info(path):
    cap = cv2.VideoCapture(path)
    info = {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return info

info1 = get_video_info(video1_path)
info2 = get_video_info(video2_path)
info3 = get_video_info(video3_path)

print(f"\nVideo 1: {info1['width']}x{info1['height']}, {info1['frames']} frames @ {info1['fps']}fps")
print(f"Video 2: {info2['width']}x{info2['height']}, {info2['frames']} frames @ {info2['fps']}fps")
print(f"Video 3: {info3['width']}x{info3['height']}, {info3['frames']} frames @ {info3['fps']}fps")

# Quality metrics calculation
def calculate_metrics(img1, img2):
    # Resize images to same size if different
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    # PSNR (Peak Signal-to-Noise Ratio) - higher is better
    psnr_value = psnr(img1, img2, data_range=255)

    # SSIM (Structural Similarity Index) - higher is better (max 1.0)
    ssim_value = ssim(img1, img2, channel_axis=2, data_range=255)

    # LPIPS (Learned Perceptual Image Patch Similarity) - lower is better
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # Normalize to [-1, 1]
    img1_tensor = img1_tensor * 2 - 1
    img2_tensor = img2_tensor * 2 - 1

    with torch.no_grad():
        lpips_value = lpips_model(img1_tensor, img2_tensor).item()

    return {
        'psnr': psnr_value,
        'ssim': ssim_value,
        'lpips': lpips_value
    }

# Analyze frames
print("\nAnalyzing frames...")
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)
cap3 = cv2.VideoCapture(video3_path)

# Select frames to analyze
frame_count = min(info1['frames'], info2['frames'], info3['frames'])
analysis_frames = [
    frame_count // 10,
    frame_count // 4,
    frame_count // 2,
    3 * frame_count // 4,
    9 * frame_count // 10
]

results = []

for frame_num in analysis_frames:
    cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    cap3.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    if ret1 and ret2 and ret3:
        # Convert to RGB
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame3_rgb = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)

        # Calculate metrics (comparing video 2 and 3 against video 1)
        metrics2 = calculate_metrics(frame1_rgb, frame2_rgb)
        metrics3 = calculate_metrics(frame1_rgb, frame3_rgb)

        results.append({
            'frame': frame_num,
            'metrics2': metrics2,
            'metrics3': metrics3,
            'frame1': frame1_rgb,
            'frame2': frame2_rgb,
            'frame3': frame3_rgb
        })

        print(f"Frame {frame_num}:")
        print(f"  Video 2: PSNR={metrics2['psnr']:.2f}dB, SSIM={metrics2['ssim']:.4f}, LPIPS={metrics2['lpips']:.4f}")
        print(f"  Video 3: PSNR={metrics3['psnr']:.2f}dB, SSIM={metrics3['ssim']:.4f}, LPIPS={metrics3['lpips']:.4f}")

cap1.release()
cap2.release()
cap3.release()

# Visual comparison
print("\nGenerating visual comparisons...")

for idx, result in enumerate(results):
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    frame1 = result['frame1']
    frame2 = result['frame2']
    frame3 = result['frame3']

    # Resize for fair comparison (use largest resolution as reference)
    max_h = max(frame1.shape[0], frame2.shape[0], frame3.shape[0])
    max_w = max(frame1.shape[1], frame2.shape[1], frame3.shape[1])

    if frame1.shape != (max_h, max_w, 3):
        frame1_resized = cv2.resize(frame1, (max_w, max_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        frame1_resized = frame1

    if frame2.shape != (max_h, max_w, 3):
        frame2_resized = cv2.resize(frame2, (max_w, max_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        frame2_resized = frame2

    if frame3.shape != (max_h, max_w, 3):
        frame3_resized = cv2.resize(frame3, (max_w, max_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        frame3_resized = frame3

    # Calculate differences
    diff2 = np.abs(frame2_resized.astype(float) - frame1_resized.astype(float)).astype(np.uint8)
    diff3 = np.abs(frame3_resized.astype(float) - frame1_resized.astype(float)).astype(np.uint8)
    diff2_enhanced = np.clip(diff2 * 3, 0, 255).astype(np.uint8)
    diff3_enhanced = np.clip(diff3 * 3, 0, 255).astype(np.uint8)

    # Row 1: Full frames
    axes[0, 0].imshow(frame1_resized)
    axes[0, 0].set_title(f'Video 1 (Reference) - Frame {result["frame"]}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(frame2_resized)
    axes[0, 1].set_title(f'Video 2 - Frame {result["frame"]}', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(frame3_resized)
    axes[0, 2].set_title(f'Video 3 - Frame {result["frame"]}', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Zoomed center crops
    h, w = max_h, max_w
    crop_size = min(h, w) // 3
    cy, cx = h // 2, w // 2
    y1, y2 = cy - crop_size//2, cy + crop_size//2
    x1, x2 = cx - crop_size//2, cx + crop_size//2

    crop1 = frame1_resized[y1:y2, x1:x2]
    crop2 = frame2_resized[y1:y2, x1:x2]
    crop3 = frame3_resized[y1:y2, x1:x2]

    axes[1, 0].imshow(crop1)
    axes[1, 0].set_title('Video 1 (Zoomed)', fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(crop2)
    axes[1, 1].set_title('Video 2 (Zoomed)', fontsize=11)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(crop3)
    axes[1, 2].set_title('Video 3 (Zoomed)', fontsize=11)
    axes[1, 2].axis('off')

    # Row 3: Difference maps
    axes[2, 0].axis('off')  # Empty cell

    axes[2, 1].imshow(diff2_enhanced)
    axes[2, 1].set_title('Difference: Video 2 vs 1', fontsize=11)
    axes[2, 1].axis('off')

    axes[2, 2].imshow(diff3_enhanced)
    axes[2, 2].set_title('Difference: Video 3 vs 1', fontsize=11)
    axes[2, 2].axis('off')

    # Add metrics text
    metrics_text = f"Video 2 vs 1: PSNR={result['metrics2']['psnr']:.2f}dB, SSIM={result['metrics2']['ssim']:.4f}, LPIPS={result['metrics2']['lpips']:.4f}\n"
    metrics_text += f"Video 3 vs 1: PSNR={result['metrics3']['psnr']:.2f}dB, SSIM={result['metrics3']['ssim']:.4f}, LPIPS={result['metrics3']['lpips']:.4f}"

    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'analysis/comparison_frame_{result["frame"]}.png', dpi=150, bbox_inches='tight')
    plt.show()

# Summary statistics
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

avg_psnr2 = np.mean([r['metrics2']['psnr'] for r in results])
avg_ssim2 = np.mean([r['metrics2']['ssim'] for r in results])
avg_lpips2 = np.mean([r['metrics2']['lpips'] for r in results])

avg_psnr3 = np.mean([r['metrics3']['psnr'] for r in results])
avg_ssim3 = np.mean([r['metrics3']['ssim'] for r in results])
avg_lpips3 = np.mean([r['metrics3']['lpips'] for r in results])

print(f"\nVideo 2 vs Video 1 (Reference):")
print(f"  PSNR:  {avg_psnr2:.2f} dB  (higher is better, >30dB is good)")
print(f"  SSIM:  {avg_ssim2:.4f}    (higher is better, max 1.0)")
print(f"  LPIPS: {avg_lpips2:.4f}    (lower is better, <0.1 is good)")

print(f"\nVideo 3 vs Video 1 (Reference):")
print(f"  PSNR:  {avg_psnr3:.2f} dB  (higher is better, >30dB is good)")
print(f"  SSIM:  {avg_ssim3:.4f}    (higher is better, max 1.0)")
print(f"  LPIPS: {avg_lpips3:.4f}    (lower is better, <0.1 is good)")

print(f"\nComparison (Video 2 vs Video 3):")
if avg_psnr2 > avg_psnr3:
    print(f"  ✓ Video 2 has better PSNR (+{avg_psnr2-avg_psnr3:.2f}dB)")
else:
    print(f"  ✓ Video 3 has better PSNR (+{avg_psnr3-avg_psnr2:.2f}dB)")

if avg_ssim2 > avg_ssim3:
    print(f"  ✓ Video 2 has better SSIM (+{avg_ssim2-avg_ssim3:.4f})")
else:
    print(f"  ✓ Video 3 has better SSIM (+{avg_ssim3-avg_ssim2:.4f})")

if avg_lpips2 < avg_lpips3:
    print(f"  ✓ Video 2 has better LPIPS (-{avg_lpips3-avg_lpips2:.4f})")
else:
    print(f"  ✓ Video 3 has better LPIPS (-{avg_lpips2-avg_lpips3:.4f})")

print(f"\nOverall Winner:")
score2 = sum([avg_psnr2 > avg_psnr3, avg_ssim2 > avg_ssim3, avg_lpips2 < avg_lpips3])
score3 = sum([avg_psnr3 > avg_psnr2, avg_ssim3 > avg_ssim2, avg_lpips3 < avg_lpips2])

if score2 > score3:
    print(f"   Video 2 wins ({score2}/3 metrics)")
elif score3 > score2:
    print(f"   Video 3 wins ({score3}/3 metrics)")
else:
    print(f"   Tie ({score2}/3 each)")

print(f"\nFile Sizes:")
size1 = os.path.getsize(video1_path) / (1024**2)
size2 = os.path.getsize(video2_path) / (1024**2)
size3 = os.path.getsize(video3_path) / (1024**2)
print(f"  Video 1: {size1:.2f} MB")
print(f"  Video 2: {size2:.2f} MB (ratio: {size2/size1:.2f}x)")
print(f"  Video 3: {size3:.2f} MB (ratio: {size3/size1:.2f}x)")

print("\n" + "="*70)
print("Comparison images saved in 'analysis/' folder")
print("="*70)
