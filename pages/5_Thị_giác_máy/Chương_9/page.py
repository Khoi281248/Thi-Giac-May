import streamlit as st
import numpy as np
import cv2
from PIL import Image
L = 256

def ConnectedComponnents(imgin):
    nguong = 200
    _, temp = cv2.threshold(imgin, nguong,L-1,cv2.THRESH_BINARY)
    imgout = cv2.medianBlur(temp, 7)
    n, label = cv2.connectedComponents(imgout, None)
    a = np.zeros(n, np.int32)
    M, N = label.shape
    for x in range(0, M):
        for y in range(0, N):
            r = label[x,y]
            if r > 0:
                a[r] = a[r] + 1
    s = 'Co %d thanh phan len thong' %(n-1)
    cv2.putText(imgout, s, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    for r in range(1, n):
        s = '%3d %5d' %(r, a[r])
        cv2.putText(imgout, s, (10,(r+1)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    return imgout

def RemoveSmallRice(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(81,81))
    temp = cv2. morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    nguong = 100
    _, temp = cv2.threshold(temp, nguong, L-1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    n, label = cv2.connectedComponents(temp, None)
    a = np.zeros(n, np.int32)
    M, N = label.shape
    for x in range(0, M):
        for y in range(0, N):
            r = label[x,y]
            if r > 0:
                a[r] = a[r] + 1
    max_value = np.max(a)
    imgout = np.zeros((M,N), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = label[x,y]
            if r >0:
                if a[r] > 0.7*max_value:
                    imgout[x,y]= L-1
    return imgout

# Danh sách các phương pháp xử lý
option = st.selectbox("Chọn phương pháp xử lý", [
    "Connected Components",
    "Remove Small Rice",
])

uploaded_file = st.file_uploader("Tải lên ảnh", type=["bmp", "png", "jpg", "jpeg", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Ảnh gốc", use_container_width=True)

    with col2:
        # Xử lý theo lựa chọn
        if option == "Connected Components":
            output = ConnectedComponnents(gray)
            st.image(output, caption="Connected Components", use_container_width=True, clamp=True)
        if option == "Remove Small Rice":
            output = RemoveSmallRice(gray)
            st.image(output, caption="Remove Small Rice", use_container_width=True, clamp=True)
