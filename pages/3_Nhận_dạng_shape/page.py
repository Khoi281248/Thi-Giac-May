import cv2
import numpy as np
from PIL import Image
import streamlit as st
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

set_background("images/3.jpg")
# Hàm phân ngưỡng
def phan_nguong(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    for x in range(M):
        for y in range(N):
            r = imgin[x, y]
            if r == 63:
                s = 255
            else:
                s = 0
            imgout[x, y] = np.uint8(s)
    # Xóa nhiễu
    imgout = cv2.medianBlur(imgout, 7)
    return imgout

# Hàm nhận dạng shape
def mnu_shape_predict_click(imgin):
    img_gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh về xám trước
    img_thresh = phan_nguong(img_gray)
    m = cv2.moments(img_thresh)
    hu = cv2.HuMoments(m)

    M, N = imgin.shape[:2]
    imgout = imgin.copy()

    if 0.000622 <= hu[0, 0] <= 0.000628:
        s = 'Hinh Tron'
    elif 0.000646 <= hu[0, 0] <= 0.000666:
        s = 'Hinh Vuong'
    elif 0.000727 <= hu[0, 0] <= 0.000749:
        s = 'Hinh Tam Giac'
    else:
        s = 'Unknown'

    imgout = cv2.putText(imgout, s, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return imgout

# ==============================
# Giao diện Streamlit
st.markdown("<h1 style='color: black;'>NHẬN DẠNG SHAPE</h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
imgin_frame = col1.empty()
imgout_frame = col2.empty()

img_file_buffer = st.file_uploader("Upload an image", type=["bmp", "png", "jpg", "jpeg", "tif"])

if img_file_buffer is not None:
    imgin_pil = Image.open(img_file_buffer)
    imgin_frame.image(imgin_pil, caption="Ảnh gốc")

    imgin = np.array(imgin_pil)

    if st.button('Nhận dạng shape'):
        imgout = mnu_shape_predict_click(imgin)
        imgout_frame.image(imgout, channels="BGR", caption="Kết quả nhận dạng")
