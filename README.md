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
```

## Usage

1. Clone this repository or download the script.

2. Ensure the necessary pre-trained models are downloaded:

   - `res10_300x300_ssd_iter_140000.caffemodel`
   - `deploy.prototxt`

   Place these files in a `model` folder in the same directory as the script.

3. Run the script:

```bash
python blink_detection.py
```

4. The webcam will open, and the program will display a window showing the video feed with real-time blink detection.

   - Green text indicates that the eyes are open.
   - Red text indicates that the eyes are closed.

5. Press the `Esc` key to exit the program.

## How It Works

1. **Face Detection**: The OpenCV DNN model detects faces in each frame.
2. **Facial Landmarks**: MediaPipe FaceMesh identifies 468 facial landmarks, including specific points around the eyes.
3. **EAR Calculation**: For each eye, the EAR is calculated using predefined landmark indices. If the EAR drops below a threshold (default: 0.2), the eyes are considered closed.
4. **Blink Status Display**: The blink status is displayed on the video feed.

## File Structure

```
├── blink_detection.py   # Main script
├── model/
│   ├── deploy.prototxt  # Face detection model configuration
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Pre-trained face detection model
```

## Customization

- **EAR Threshold**: Adjust the `EAR_THRESHOLD` value to fine-tune the sensitivity of blink detection.
- **Model Files**: Replace the face detection model with any other DNN-based model supported by OpenCV.

## References

- [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh)
- [OpenCV DNN Module](https://docs.opencv.org/master/d6/d0f/group__dnn.html)
- [Eye Aspect Ratio (EAR)]([https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf](https://pmc.ncbi.nlm.nih.gov/articles/PMC9044337/))



