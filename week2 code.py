# ENHANCED VIDEO SUPER-RESOLUTION PIPELINE
# Supports multiple upscaling methods with quality metrics

!pip install torch torchvision opencv-python tqdm Pillow numpy --quiet

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from google.colab import files
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class Config:
    BASE_DIR = "/content/video-super-resolution"
    INPUT_DIR = f"{BASE_DIR}/input"
    OUTPUT_DIR = f"{BASE_DIR}/output"
    SCALE_FACTOR = 4
    UPSCALE_METHOD = "lanczos"

config = Config()

os.makedirs(config.INPUT_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
print(f"Project folder: {config.BASE_DIR}")

uploaded = files.upload()
if not uploaded:
    raise ValueError("No file uploaded!")

video_name = list(uploaded.keys())[0]
input_path = f"{config.INPUT_DIR}/{video_name}"
os.rename(video_name, input_path)
print(f"Video ready: {video_name}")

def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    cap.release()

    print("Video Information:")
    print(f"   Resolution: {info['width']}x{info['height']}")
    print(f"   FPS: {info['fps']}")
    print(f"   Frames: {info['frame_count']}")
    print(f"   Duration: {info['duration']}s")

    return info

video_info = get_video_info(input_path)

class VideoUpscaler:
    @staticmethod
    def upscale_frame(frame: np.ndarray, scale: int, method: str) -> np.ndarray:
        h, w = frame.shape[:2]
        new_size = (w * scale, h * scale)

        if method == 'lanczos':
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            upscaled = pil_img.resize(new_size, Image.LANCZOS)
            return cv2.cvtColor(np.array(upscaled), cv2.COLOR_RGB2BGR)

        elif method == 'bicubic':
            return cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)

        elif method == 'bilinear':
            return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def process_video(input_path: str,
                     output_path: str,
                     scale: int = 4,
                     method: str = 'lanczos',
                     quality: int = 95) -> None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Cannot open input video")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_width, out_height = width * scale, height * scale
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        print(f"Starting upscaling...")
        print(f"   Method: {method.upper()}")
        print(f"   Scale: ×{scale}")
        print(f"   Input: {width}x{height}")
        print(f"   Output: {out_width}x{out_height}")

        with tqdm(total=frame_count, desc="Processing frames", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                upscaled = VideoUpscaler.upscale_frame(frame, scale, method)
                out.write(upscaled)
                pbar.update(1)

        cap.release()
        out.release()
        print(f"Video saved: {output_path}")

output_name = f"{os.path.splitext(video_name)[0]}_upscaled_{config.SCALE_FACTOR}x.mp4"
output_path = f"{config.OUTPUT_DIR}/{output_name}"

upscaler = VideoUpscaler()
upscaler.process_video(
    input_path=input_path,
    output_path=output_path,
    scale=config.SCALE_FACTOR,
    method=config.UPSCALE_METHOD
)

def compare_frames(original_path: str,
                  upscaled_path: str,
                  frame_indices: list = [10, 50, 100]) -> None:
    cap_orig = cv2.VideoCapture(original_path)
    cap_upsc = cv2.VideoCapture(upscaled_path)

    fig, axes = plt.subplots(len(frame_indices), 2, figsize=(12, 4*len(frame_indices)))
    if len(frame_indices) == 1:
        axes = [axes]

    for idx, frame_id in enumerate(frame_indices):
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        cap_upsc.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret1, frame1 = cap_orig.read()
        ret2, frame2 = cap_upsc.read()

        if ret1 and ret2:
            h, w = frame2.shape[:2]
            frame1_resized = cv2.resize(frame1, (w, h), interpolation=cv2.INTER_LANCZOS4)

            axes[idx][0].imshow(cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB))
            axes[idx][0].set_title(f"Original (Frame {frame_id})")
            axes[idx][0].axis('off')

            axes[idx][1].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            axes[idx][1].set_title(f"Upscaled ×{config.SCALE_FACTOR} (Frame {frame_id})")
            axes[idx][1].axis('off')

    cap_orig.release()
    cap_upsc.release()
    plt.tight_layout()
    plt.show()

frame_samples = [
    video_info['frame_count'] // 4,
    video_info['frame_count'] // 2,
    3 * video_info['frame_count'] // 4
]
compare_frames(input_path, output_path, frame_samples[:min(3, video_info['frame_count'])])

orig_size = os.path.getsize(input_path) / (1024 * 1024)
upsc_size = os.path.getsize(output_path) / (1024 * 1024)

print(f"File Size Comparison:")
print(f"   Original: {orig_size:.2f} MB")
print(f"   Upscaled: {upsc_size:.2f} MB")
print(f"   Ratio: {upsc_size/orig_size:.2f}x")

files.download(output_path)
print("All done! Your video has been upscaled successfully.")
