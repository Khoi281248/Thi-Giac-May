import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import math

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

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
tip_ids = [4, 8, 12, 16, 20]

class HandGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=1)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Lật ảnh theo chiều ngang
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gesture = "Unknown"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                fingers = []

                # Ngón cái
                if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0] and lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
                    fingers.append(1)  # Ngón cái mở
                else:
                    fingers.append(0)  # Ngón cái đóng

                # Điều chỉnh: Kiểm tra hướng của ngón cái cho cả tay trái và tay phải
                if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:  # Nếu ngón cái tay phải
                    if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
                        fingers[0] = 0  # Đóng ngón cái khi có sự thay đổi chiều y
                else:  # Nếu ngón cái tay trái
                    if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
                        fingers[0] = 0  # Đóng ngón cái khi có sự thay đổi chiều y

                # 4 ngón còn lại
                for i in range(1, 5):
                    if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1]:
                        fingers.append(1)  # Ngón tay mở
                    else:
                        fingers.append(0)  # Ngón tay đóng

                total_fingers = sum(fingers)

                # Nhận diện cử chỉ
                if total_fingers == 0:
                    gesture = "Fist "
                elif total_fingers == 5:
                    gesture = "Open Hand "
                elif fingers == [0, 1, 1, 0, 0]:
                    gesture = "Helo "
                elif fingers == [1, 0, 0, 0, 0]:
                    gesture = "Like "
                elif distance(lm_list[4], lm_list[8]) < 40:
                    gesture = "OK "
                else:
                    gesture = f"{total_fingers} fingers"

                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(img, gesture, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        else:
            cv2.putText(img, "No Hand Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return img

# Giao diện Streamlit
st.title("Nhận diện cử chỉ tay với Streamlit")
st.markdown("Real-time detection using MediaPipe + OpenCV + streamlit-webrtc")

webrtc_streamer(key="hand-gesture", video_transformer_factory=HandGestureTransformer)
