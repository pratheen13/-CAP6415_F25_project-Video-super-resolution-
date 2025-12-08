!pip install torch torchvision opencv-python tqdm Pillow numpy --quiet

# DEPENDENCIES
import os
import cv2  # OpenCV: Used for video reading, writing, and frame extraction
import torch
import numpy as np
from tqdm import tqdm  # TQDM: Used to display a progress bar during long processing
from PIL import Image  # Pillow: Used for high-quality image resizing (Lanczos)
from google.colab import files  # Colab specific: Handles file upload/download
import matplotlib.pyplot as plt  # Matplotlib: Used for plotting visual comparisons
from typing import Tuple, Optional


# CONFIGURATION MODULE

class Config:
    """
    Central configuration class.
    Stores all file paths, constants, and hyperparameters for the pipeline.
    This makes it easy to change settings (like scale factor) in one place.
    """
    BASE_DIR = "/content/video-super-resolution"
    INPUT_DIR = f"{BASE_DIR}/input"
    OUTPUT_DIR = f"{BASE_DIR}/output"
    
    # The magnification factor (e.g., 4 means 4x resolution increase)
    SCALE_FACTOR = 4
    
    # The algorithm used for upscaling. Options: 'lanczos', 'bicubic', 'bilinear'
    UPSCALE_METHOD = "lanczos"

# Initialize configuration
config = Config()

# Setup the directory structure required for the pipeline
os.makedirs(config.INPUT_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
print(f"Project folder: {config.BASE_DIR}")


# INPUT MODULE

# Trigger the Google Colab file upload widget to let the user select a video.
uploaded = files.upload()
if not uploaded:
    raise ValueError("No file uploaded!")

# Get the filename and move it to the organized input directory
video_name = list(uploaded.keys())[0]
input_path = f"{config.INPUT_DIR}/{video_name}"
os.rename(video_name, input_path)
print(f"Video ready: {video_name}")


# METADATA EXTRACTION MODULE

def get_video_info(video_path: str) -> dict:
    """
    Analyzes the input video file to extract technical specifications.
    
    Args:
        video_path (str): Path to the source video.
        
    Returns:
        dict: A dictionary containing FPS, dimensions, frame count, and duration.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Extract properties using OpenCV constants
    info = {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    cap.release()

    # Log information to the console for user verification
    print("Video Information:")
    print(f"   Resolution: {info['width']}x{info['height']}")
    print(f"   FPS: {info['fps']}")
    print(f"   Frames: {info['frame_count']}")
    print(f"   Duration: {info['duration']}s")

    return info

video_info = get_video_info(input_path)


# CORE PROCESSING MODULE (UPSCALER)

class VideoUpscaler:
    """
    The main engine for the Super-Resolution task.
    Encapsulates frame processing logic and the full video processing loop.
    """
    
    @staticmethod
    def upscale_frame(frame: np.ndarray, scale: int, method: str) -> np.ndarray:
        """
        Upscales a single frame using the specified interpolation method.
        
        Args:
            frame: The source image/frame (NumPy array).
            scale: The multiplier for resolution (e.g., 4).
            method: 'lanczos' (highest quality), 'bicubic', or 'bilinear'.
            
        Returns:
            np.ndarray: The upscaled frame.
        """
        h, w = frame.shape[:2]
        new_size = (w * scale, h * scale)

        if method == 'lanczos':
            # Convert OpenCV BGR to Pillow RGB for high-quality Lanczos processing
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            upscaled = pil_img.resize(new_size, Image.LANCZOS)
            # Convert back to OpenCV BGR format
            return cv2.cvtColor(np.array(upscaled), cv2.COLOR_RGB2BGR)

        elif method == 'bicubic':
            # Use OpenCV's native bicubic interpolation
            return cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)

        elif method == 'bilinear':
            # Use OpenCV's native bilinear interpolation (fastest)
            return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def process_video(input_path: str,
                      output_path: str,
                      scale: int = 4,
                      method: str = 'lanczos',
                      quality: int = 95) -> None:
        """
        Reads the input video, processes every frame, and writes the output video.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Cannot open input video")

        # Get input video properties to replicate in output
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate target dimensions
        out_width, out_height = width * scale, height * scale
        
        # Setup VideoWriter. 'mp4v' is a widely compatible codec for mp4 containers.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        print(f"Starting upscaling...")
        print(f"   Method: {method.upper()}")
        print(f"   Scale: ×{scale}")
        print(f"   Input: {width}x{height}")
        print(f"   Output: {out_width}x{out_height}")

        # TQDM loop for visual progress tracking
        with tqdm(total=frame_count, desc="Processing frames", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # End of video stream
                
                # Perform upscaling on the current frame
                upscaled = VideoUpscaler.upscale_frame(frame, scale, method)
                
                # Write the new frame to the output video file
                out.write(upscaled)
                pbar.update(1)

        cap.release()
        out.release()
        print(f"Video saved: {output_path}")

# Define output filename based on input name and scale factor
output_name = f"{os.path.splitext(video_name)[0]}_upscaled_{config.SCALE_FACTOR}x.mp4"
output_path = f"{config.OUTPUT_DIR}/{output_name}"

# Execute the processing pipeline
upscaler = VideoUpscaler()
upscaler.process_video(
    input_path=input_path,
    output_path=output_path,
    scale=config.SCALE_FACTOR,
    method=config.UPSCALE_METHOD
)


# VALIDATION & ANALYSIS MODULE

def compare_frames(original_path: str,
                   upscaled_path: str,
                   frame_indices: list = [10, 50, 100]) -> None:
    """
    Visualizes the difference between the original and upscaled video frames side-by-side.
    """
    cap_orig = cv2.VideoCapture(original_path)
    cap_upsc = cv2.VideoCapture(upscaled_path)

    # Create a subplot grid for the images
    fig, axes = plt.subplots(len(frame_indices), 2, figsize=(12, 4*len(frame_indices)))
    if len(frame_indices) == 1:
        axes = [axes]

    for idx, frame_id in enumerate(frame_indices):
        # Seek both video players to the specific frame index
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        cap_upsc.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret1, frame1 = cap_orig.read()
        ret2, frame2 = cap_upsc.read()

        if ret1 and ret2:
            h, w = frame2.shape[:2]
            # Resize the original frame strictly for display purposes so it matches the upscaled size
            frame1_resized = cv2.resize(frame1, (w, h), interpolation=cv2.INTER_LANCZOS4)

            # Display Original Frame
            axes[idx][0].imshow(cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB))
            axes[idx][0].set_title(f"Original (Frame {frame_id})")
            axes[idx][0].axis('off')

            # Display Upscaled Frame
            axes[idx][1].imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            axes[idx][1].set_title(f"Upscaled ×{config.SCALE_FACTOR} (Frame {frame_id})")
            axes[idx][1].axis('off')

    cap_orig.release()
    cap_upsc.release()
    plt.tight_layout()
    plt.show()

# Select 3 sample frames distributed evenly across the video for comparison
frame_samples = [
    video_info['frame_count'] // 4,
    video_info['frame_count'] // 2,
    3 * video_info['frame_count'] // 4
]
compare_frames(input_path, output_path, frame_samples[:min(3, video_info['frame_count'])])


# METRICS & DOWNLOAD MODULE

# Calculate file sizes in MB
orig_size = os.path.getsize(input_path) / (1024 * 1024)
upsc_size = os.path.getsize(output_path) / (1024 * 1024)

print(f"File Size Comparison:")
print(f"   Original: {orig_size:.2f} MB")
print(f"   Upscaled: {upsc_size:.2f} MB")
print(f"   Ratio: {upsc_size/orig_size:.2f}x")

# Trigger browser download of the final result
files.download(output_path)
print("All done! Your video has been upscaled successfully.")
