import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize MediaPipe Pose class
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# Squat detection logic
def detect_squats():
    cap = cv2.VideoCapture(0)
    squat_count = 0
    stage = None

    # Start webcam feed and detect squats
    stframe = st.empty()  # Placeholder for Streamlit frame to update the webcam feed
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB for MediaPipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Convert back to BGR for OpenCV rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks and calculate the angle
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates of key points (left side)
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate the angles
                hip_knee_angle = calculate_angle(hip, knee, ankle)

                # Squat counting logic
                if hip_knee_angle > 160:
                    stage = "up"
                if hip_knee_angle < 90 and stage == 'up':
                    stage = "down"
                    squat_count += 1
                    st.session_state["squat_count"] = squat_count

                # Add squat count to the image
                cv2.putText(image, f'Squats: {squat_count}', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                pass

            # Draw landmarks and pose connections on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Show the image in Streamlit
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Streamlit
            img = Image.fromarray(frame)
            stframe.image(img, channels="RGB")

        cap.release()

# Streamlit Layout
st.title("Squat Detection App")
st.write("Perform squats in front of your webcam to start detection.")

# Initialize squat count in session state
if "squat_count" not in st.session_state:
    st.session_state["squat_count"] = 0

# Start the squat detection
if st.button("Start Squat Detection"):
    detect_squats()

# Display the squat count
st.write(f"Total Squats: {st.session_state['squat_count']}")
