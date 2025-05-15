import cv2
import mediapipe as mp
import math
import time
import av
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

import base64
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:images/nen_main.png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("images/6.gif")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh

# EAR threshold
EYE_CLOSE_THRESHOLD = 0.2
HEAD_TILT_THRESHOLD = 10

left_eye_idx = [33, 160, 158, 133, 153, 144]
right_eye_idx = [362, 385, 387, 263, 373, 380]

def compute_EAR(landmarks, indices):
    p = [landmarks[i] for i in indices]
    vertical1 = math.dist((p[1].x, p[1].y), (p[5].x, p[5].y))
    vertical2 = math.dist((p[2].x, p[2].y), (p[4].x, p[4].y))
    horizontal = math.dist((p[0].x, p[0].y), (p[3].x, p[3].y))
    return (vertical1 + vertical2) / (2.0 * horizontal)


class DrowsinessDetector(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.eye_closed_start_time = None
        self.drowsy_count = 0
        self.tilt_count = 0
        self.eye_warning_triggered = False
        self.head_warning_triggered = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        show_eye_warning = False
        show_head_warning = False

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                if len(landmarks.landmark) < 468:
                    continue

                left_EAR = compute_EAR(landmarks.landmark, left_eye_idx)
                right_EAR = compute_EAR(landmarks.landmark, right_eye_idx)

                eye_closed = left_EAR < EYE_CLOSE_THRESHOLD and right_EAR < EYE_CLOSE_THRESHOLD

                if eye_closed:
                    if self.eye_closed_start_time is None:
                        self.eye_closed_start_time = time.time()
                    else:
                        duration = time.time() - self.eye_closed_start_time
                        cv2.putText(img, f"Eye Closed: {duration:.1f}s", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        if duration > 3:
                            show_eye_warning = True
                else:
                    self.eye_closed_start_time = None

                # Head tilt
                left_cheek = landmarks.landmark[234]
                right_cheek = landmarks.landmark[454]
                dx = right_cheek.x - left_cheek.x
                dy = right_cheek.y - left_cheek.y
                angle = math.degrees(math.atan2(dy, dx))
                if abs(angle) > HEAD_TILT_THRESHOLD:
                    show_head_warning = True

        if show_eye_warning:
            cv2.putText(img, "DROWSINESS DETECTED!", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if not self.eye_warning_triggered:
                self.drowsy_count += 1
                self.eye_warning_triggered = True
        else:
            self.eye_warning_triggered = False

        if show_head_warning:
            cv2.putText(img, "HEAD TILT DETECTED!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if not self.head_warning_triggered:
                self.tilt_count += 1
                self.head_warning_triggered = True
        else:
            self.head_warning_triggered = False

        # Display counters
        cv2.putText(img, f"Drowsy Count: {self.drowsy_count}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, f"Tilt Count: {self.tilt_count}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return img

# --- Streamlit interface ---
st.title("🛑 Driver Drowsiness & Head Tilt Detection")
st.markdown("Sử dụng webcam để phát hiện buồn ngủ và nghiêng đầu.")

webrtc_streamer(key="drowsiness",
                video_transformer_factory=DrowsinessDetector,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
