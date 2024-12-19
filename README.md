# Video Processing Application

This project is a video processing application that uses face detection and various image stretching techniques to stretch video from 4:3 to 16:9 aspect ratio. Currently, it supports three methods: Gaussian Stretch based on faces, Low Center Stretch based on faces, and Unified Stretch. User can check the result of each scene.

## Features

- Detects scenes in a video and processes each scene using different stretching methods.
- Provides a GUI to preview processed video scenes.
- Supports face detection using MediaPipe.

## Installation

1. Clone the repository:

   ```bash
   conda create -n face_centered_stretch python=3.10
   conda activate face_centered_stretch
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:

   ```bash
   python face_centered_stretch/main.py
   ```

2. Use the GUI to open a video file and process it.

## Dependencies

- PyQt5: For creating the GUI.
- OpenCV: For video processing.
- NumPy: For numerical operations.
- MediaPipe: For face detection.

