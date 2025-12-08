
# Video Super-Resolution Project

CAP6415 – Fall 2025

This repository contains a complete implementation of a video super-resolution system using both classical interpolation and deep-learning–based methods. The project includes video upscaling, frame-level comparison tools, Real-ESRGAN-based enhancement, and weekly progress reports documenting work from Week 1 to Week 5.

The pipeline is designed to run on Google Colab or any Python environment with GPU support.



## Repository Structure

```
project-root/
│
├── codes/
│   ├── RealESRGAN.py                 # Real-ESRGAN x4 upscaling module
│   ├── Comparison.py                 # Frame-by-frame quality comparison tool
│   ├── 3 Video Comparison.py         # Multi-video comparison and analysis
│   ├── week 1 code.py                # Week 1 baseline implementation
│   ├── week2(lanczos) code.py        # Week 2 Lanczos interpolation module
│
├── ipynb files/
│   ├── Week 1.ipynb                  # Notebook version of Week 1 pipeline
│   ├── cvv (1).ipynb                 # Additional testing notebook
│
├── outputs/
│   └── upscaled_videos/              # Processed high-resolution outputs
│
├── weekly report/
│   ├── week 1.txt
│   ├── week 2.txt
│   ├── week 3.txt
│   ├── week 4.txt
│   ├── week 5.txt
│
├── requirements.txt                  # Project dependencies
├── README.md                         # Project documentation
└── video-super-resolution.zip        # Exported project archive
```



## Requirements

Listed in `requirements.txt`:

```
torch
torchvision
opencv-python
ffmpeg-python
tqdm
scikit-image
lpips
Pillow
matplotlib
numpy
huggingface-hub==0.13.4
torch>=1.7
torchvision>=0.8.0
```

Install using:

```
pip install -r requirements.txt
```



## Project Overview

The goal of this project is to enhance low-resolution videos using advanced super-resolution techniques. This includes:

* Traditional interpolation → Bicubic, Lanczos
* Deep learning enhancement → Real-ESRGAN
* Frame extraction and reconstruction
* Objective quality metrics
* Visual comparison across multiple upscaling methods

Each module in the `codes/` folder corresponds to a specific part of the development timeline.



## Usage Instructions

### 1. Place your input video

Copy your video file into:

```
/input/    (for notebooks)
or provide direct path in python scripts
```

### 2. Run an upscaling module

Example: Real-ESRGAN enhancement

```bash
python codes/RealESRGAN.py
```

Example: Lanczos baseline upscaling

```bash
python codes/week2(lanczos) code.py
```

### 3. Compare results

Use:

```bash
python codes/Comparison.py
```

or multi-video comparison:

```bash
python "codes/3 Video Comparison.py"
```

### 4. Check generated output

Upscaled videos are saved in:

```
outputs/
```



## Notebooks

The project includes Jupyter notebooks for step-by-step demonstration:

* **Week 1.ipynb** — Initial pipeline: frame extraction, resizing, reconstruction
* **cvv (1).ipynb** — Testing different interpolation and comparison features



## Weekly Reports

The folder `weekly report/` contains written summaries of progress from Week 1 to Week 5, documenting:

* Implemented features
* Challenges encountered
* Experiments performed
* Results and evaluation
* Plans for next phase

These reports support the final project submission requirement.



## Shortcomings

* Real-ESRGAN inference is computationally intensive without GPU.
* Extremely low-quality input videos may introduce artifacts even after enhancement.
* ffmpeg processing speed depends heavily on system performance.



## Future Improvements

* Integration of NCNN-based Real-ESRGAN for faster CPU performance
* Adding a GUI front-end for local users
* Support batch video processing
* Add audio-preserving upscale pipeline
* Explore additional models (SwinIR, EGVSR, BasicSR)



## Author

Pratheen Reddy, Shiva Prasad, Sai Kiran

Florida Atlantic University

CAP6415 – Artificial Intelligence

Fall 2025

