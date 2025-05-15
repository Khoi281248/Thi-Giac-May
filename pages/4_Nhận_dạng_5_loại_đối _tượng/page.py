import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
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
            # background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("images/4.jpg")

# Load YOLO model
model = YOLO('yolo11n_trai_cay.pt')  # Đảm bảo file .pt không bị lỗi

st.title("YOLO11n Object Detection with Streamlit")
st.write("Upload an image to detect objects using YOLO11n.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    imgin = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run prediction
    names = model.names
    imgout = imgin.copy()
    annotator = Annotator(imgout)
    results = model.predict(imgin, conf=0.6, verbose=False)

    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()
    confs = results[0].boxes.conf.tolist()

    for box, cls, conf in zip(boxes, clss, confs):
        label = f"{names[int(cls)]} {conf:.2f}"
        annotator.box_label(box, label=label, txt_color=(255, 0, 0), color=(255, 255, 255))

    imgout = annotator.result()
    imgout_rgb = cv2.cvtColor(imgout, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Ảnh gốc", use_container_width=True)

    with col2:
        st.image(imgout_rgb, caption="Detection Result", use_container_width=True)
