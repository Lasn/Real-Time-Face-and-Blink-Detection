import cv2
import mediapipe as mp
import numpy as np

# MediaPipe FaceMesh setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


# EAR calculation function
def calculate_ear(eye_landmarks, facial_landmarks, image_width, image_height):
    def get_coords(landmark):
        return np.array([landmark.x * image_width, landmark.y * image_height])

    A = np.linalg.norm(
        get_coords(facial_landmarks[eye_landmarks[1]])
        - get_coords(facial_landmarks[eye_landmarks[5]])
    )
    B = np.linalg.norm(
        get_coords(facial_landmarks[eye_landmarks[2]])
        - get_coords(facial_landmarks[eye_landmarks[4]])
    )
    C = np.linalg.norm(
        get_coords(facial_landmarks[eye_landmarks[0]])
        - get_coords(facial_landmarks[eye_landmarks[3]])
    )
    ear = (A + B) / (2.0 * C)
    return ear


# Indices for left and right eyes landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR threshold to detect closed eyes, 0.2 is standard
EAR_THRESHOLD = 0.2

# OpenCV DNN setup for face detection
model_path = "model/res10_300x300_ssd_iter_140000.caffemodel"
config_path = "model/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)

# For real-time video
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        h, w = frame.shape[:2]

        # Prepare the image for input to the neural network
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Set the blob as input to the network
        net.setInput(blob)

        # Perform forward pass to get the detections
        detections = net.forward()

        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.5:
                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Extract the face region
                face_region = frame[startY:endY, startX:endX]

                # Convert the BGR image to RGB before processing with MediaPipe FaceMesh
                face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(face_rgb)

                if not results.multi_face_landmarks:
                    continue

                for face_landmarks in results.multi_face_landmarks:
                    # Convert the landmarks to a list of tuples for easier access
                    landmarks = face_landmarks.landmark

                    # Calculate EAR for left and right eyes
                    left_ear = calculate_ear(
                        LEFT_EYE, landmarks, endX - startX, endY - startY
                    )
                    right_ear = calculate_ear(
                        RIGHT_EYE, landmarks, endX - startX, endY - startY
                    )

                    # Check if the eyes are closed
                    if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                        blink_status = "Eyes are closed"
                    else:
                        blink_status = "Eyes are open"

                    # Draw landmarks on the face region
                    mp_drawing.draw_landmarks(
                        image=face_region,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=face_region,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=face_region,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )

                # Place the annotated face region back into the original image
                frame[startY:endY, startX:endX] = face_region

                # Display the blink status
                cv2.putText(
                    frame,
                    blink_status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if "open" in blink_status else (0, 0, 255),
                    2,
                )

        # Display the frame
        cv2.imshow("Blink Detection", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
