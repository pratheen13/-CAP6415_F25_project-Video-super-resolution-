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

print(f"\nVideo 1: {info1['width']}x{info1['height']}, {info1['frames']} frames @ {info1['fps']}fps")
print(f"Video 2: {info2['width']}x{info2['height']}, {info2['frames']} frames @ {info2['fps']}fps")

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

# Select frames to analyze
frame_count = min(info1['frames'], info2['frames'])
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

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2:
        # Convert to RGB
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Calculate metrics
        metrics = calculate_metrics(frame1_rgb, frame2_rgb)
        results.append({
            'frame': frame_num,
            'metrics': metrics,
            'frame1': frame1_rgb,
            'frame2': frame2_rgb
        })

        print(f"Frame {frame_num}: PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}, LPIPS={metrics['lpips']:.4f}")

cap1.release()
cap2.release()

# Visual comparison
print("\nGenerating visual comparisons...")

for idx, result in enumerate(results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    frame1 = result['frame1']
    frame2 = result['frame2']

    # Resize for fair comparison
    if frame1.shape != frame2.shape:
        frame1_resized = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    else:
        frame1_resized = frame1

    # Calculate absolute difference
    diff = np.abs(frame2.astype(float) - frame1_resized.astype(float)).astype(np.uint8)
    diff_enhanced = np.clip(diff * 3, 0, 255).astype(np.uint8)

    # Row 1: Full frames
    axes[0, 0].imshow(frame1_resized)
    axes[0, 0].set_title(f'Video 1 - Frame {result["frame"]}', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(frame2)
    axes[0, 1].set_title(f'Video 2 - Frame {result["frame"]}', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(diff_enhanced)
    axes[0, 2].set_title('Difference (Enhanced 3x)', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Zoomed center crops
    h, w = frame2.shape[:2]
    crop_size = min(h, w) // 3
    cy, cx = h // 2, w // 2
    y1, y2 = cy - crop_size//2, cy + crop_size//2
    x1, x2 = cx - crop_size//2, cx + crop_size//2

    crop1 = frame1_resized[y1:y2, x1:x2]
    crop2 = frame2[y1:y2, x1:x2]
    crop_diff = diff_enhanced[y1:y2, x1:x2]

    axes[1, 0].imshow(crop1)
    axes[1, 0].set_title('Video 1 (Zoomed)', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(crop2)
    axes[1, 1].set_title('Video 2 (Zoomed)', fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(crop_diff)
    axes[1, 2].set_title('Difference (Zoomed)', fontsize=12)
    axes[1, 2].axis('off')

    # Add metrics text
    metrics_text = f"PSNR: {result['metrics']['psnr']:.2f} dB\n"
    metrics_text += f"SSIM: {result['metrics']['ssim']:.4f}\n"
    metrics_text += f"LPIPS: {result['metrics']['lpips']:.4f}"

    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'analysis/comparison_frame_{result["frame"]}.png', dpi=150, bbox_inches='tight')
    plt.show()

# Summary statistics
print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)

avg_psnr = np.mean([r['metrics']['psnr'] for r in results])
avg_ssim = np.mean([r['metrics']['ssim'] for r in results])
avg_lpips = np.mean([r['metrics']['lpips'] for r in results])

print(f"\nAverage Metrics:")
print(f"  PSNR:  {avg_psnr:.2f} dB  (higher is better, >30dB is good)")
print(f"  SSIM:  {avg_ssim:.4f}    (higher is better, max 1.0)")
print(f"  LPIPS: {avg_lpips:.4f}    (lower is better, <0.1 is good)")

print(f"\nInterpretation:")
if avg_psnr > 30:
    print("  ✓ PSNR: Excellent quality preservation")
elif avg_psnr > 25:
    print("  ~ PSNR: Good quality with some differences")
else:
    print("  ✗ PSNR: Significant differences detected")

if avg_ssim > 0.95:
    print("  ✓ SSIM: Very similar structure")
elif avg_ssim > 0.85:
    print("  ~ SSIM: Moderately similar structure")
else:
    print("  ✗ SSIM: Different structural characteristics")

if avg_lpips < 0.1:
    print("  ✓ LPIPS: Perceptually very similar")
elif avg_lpips < 0.3:
    print("  ~ LPIPS: Noticeable perceptual differences")
else:
    print("  ✗ LPIPS: Significant perceptual differences")

print(f"\nFile Sizes:")
size1 = os.path.getsize(video1_path) / (1024**2)
size2 = os.path.getsize(video2_path) / (1024**2)
print(f"  Video 1: {size1:.2f} MB")
print(f"  Video 2: {size2:.2f} MB")
print(f"  Ratio:   {size2/size1:.2f}x")

print("\n" + "="*60)
print("Comparison images saved in 'analysis/' folder")
print("="*60)
