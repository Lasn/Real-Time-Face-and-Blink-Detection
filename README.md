# Real-Time Blink Detection with OpenCV and MediaPipe

This project implements real-time blink detection using OpenCV, MediaPipe FaceMesh, and an EAR (Eye Aspect Ratio) calculation method. It captures video from a webcam, detects faces and eyes, and determines whether the eyes are open or closed.

## Features

- **Face Detection**: Uses OpenCV's DNN module with a pre-trained Caffe model for face detection.
- **Facial Landmark Detection**: Leverages MediaPipe FaceMesh to extract facial landmarks, including eye regions.
- **Blink Detection**: Implements the Eye Aspect Ratio (EAR) algorithm to detect whether the eyes are closed.
- **Real-Time Processing**: Captures and processes video frames from the webcam in real time.

## Dependencies

Ensure you have the following Python libraries installed:

- `opencv-python`
- `mediapipe`
- `numpy`

To install the required libraries, run:

```bash
pip install opencv-python mediapipe numpy
