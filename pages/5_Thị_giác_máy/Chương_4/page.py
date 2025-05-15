import streamlit as st
import numpy as np
import cv2
from PIL import Image
L = 256

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

set_background("images/5_2.jpg")

def Spectrum(imgin):
    M,N = imgin.shape
    # Bước 1 và 2: Mở rộng và thêm số 0 vòa phần mở rộngrộng
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = 1.0*imgin/(L-1)

    # Bước 3: nhan fp với (-1)^(x+y)
    for x in range (0,M):
        for y in range (0,N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]

    # Bước 4: DFT
    F = cv2.dft(fp,flags= cv2.DFT_COMPLEX_OUTPUT)

    # Bước 5: Spectrum
    S = np.sqrt(F[:,:,0]**2 + F[:,:,1]**2)
    S = np.clip(S,0,L-1)
    S = S.astype(np.uint8) 
    return S       

def CreateNotchFilter(P, Q):
    # Tạo bộ loc H là số phức, có phân faor bằng 0
    H = np.ones((P,Q,2),np.float32)
    H[:,:,1] = 0.0
    
    u1,v1 = 45, 59
    u2,v2 = 86, 59
    u3,v3 = 39, 119
    u4,v4 = 83, 119
    D0 = 10
    for u in range(0,P):
        for v in range(0,Q):
            # u1, v1
            Duv = np.sqrt((u-u1)**2 + (v-v1)**2)
            if Duv <= D0:
                H[u,v,0] = 0.0
            # u2, v2
            Duv = np.sqrt((u-u2)**2 + (v-v2)**2)
            if Duv <= D0:
                H[u,v,0] = 0.0
            # u3, v3
            Duv = np.sqrt((u-u3)**2 + (v-v3)**2)
            if Duv <= D0:
                H[u,v,0] = 0.0
            # u4, v4
            Duv = np.sqrt((u-u4)**2 + (v-v4)**2)
            if Duv <= D0:
                H[u,v,0] = 0.0

            # Đối xứng của u1, v1
            Duv = np.sqrt((u-(P-u1))**2 + (v-(Q-v1))**2)
            if Duv <= D0:
                H[u,v,0] = 0.0
            # Đối xứng của u2, v2
            Duv = np.sqrt((u-(P-u2))**2 + (v-(Q-v2))**2)
            if Duv <= D0:
                H[u,v,0] = 0.0
            # Đối xứng của u3, v3
            Duv = np.sqrt((u-(P-u3))**2 + (v-(Q-v3))**2)
            if Duv <= D0:
                H[u,v,0] = 0.0
            # Đối xứng của u4, v4
            Duv = np.sqrt((u-(P-u4))**2 + (v-(Q-v4))**2)
            if Duv <= D0:
                H[u,v,0] = 0.0
    return H


def CreateButterworthNotchRejectFilter(P, Q):
    # Tạo bộ loc H là số phức, có phân faor bằng 0
    H = np.ones((P,Q,2),np.float32)
    H[:,:,1] = 0.0
    
    u1,v1 = 45, 59
    u2,v2 = 86, 59
    u3,v3 = 39, 119
    u4,v4 = 83, 119
    D0 = 10
    n = 2
    for u in range(0,P):
        for v in range(0,Q):
            r=1.0
            # u1, v1
            Duv = np.sqrt((u-u1)**2 + (v-v1)**2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r= r * 1.0/(1.0 + np.power(D0/Duv,n))
                else:
                    r = 0.0

            # u2, v2
            Duv = np.sqrt((u-u2)**2 + (v-v2)**2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r= r * 1.0/(1.0 + np.power(D0/Duv,n))
                else:
                    r = 0.0

            # u3, v3
            Duv = np.sqrt((u-u3)**2 + (v-v3)**2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r= r * 1.0/(1.0 + np.power(D0/Duv,n))
                else:
                    r = 0.0

            # u4, v4
            Duv = np.sqrt((u-u4)**2 + (v-v4)**2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r= r * 1.0/(1.0 + np.power(D0/Duv,n))
                else:
                    r = 0.0

            # Đối xứng của u1, v1
            Duv = np.sqrt((u-(P-u1))**2 + (v-(Q-v1))**2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r= r * 1.0/(1.0 + np.power(D0/Duv,n))
                else:
                    r = 0.0

            # Đối xứng của u2, v2
            Duv = np.sqrt((u-(P-u2))**2 + (v-(Q-v2))**2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r= r * 1.0/(1.0 + np.power(D0/Duv,n))
                else:
                    r = 0.0

            # Đối xứng của u3, v3
            Duv = np.sqrt((u-(P-u3))**2 + (v-(Q-v3))**2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r= r * 1.0/(1.0 + np.power(D0/Duv,n))
                else:
                    r = 0.0

            # Đối xứng của u4, v4
            Duv = np.sqrt((u-(P-u4))**2 + (v-(Q-v4))**2)
            if Duv <= D0:
                if abs(Duv) >= 1e-10:
                    r= r * 1.0/(1.0 + np.power(D0/Duv,n))
                else:
                    r = 0.0

            H[u,v,0] = r
    return H

def CreateVerticalNotchRejectFilter(P, Q):
    # Tạo bộ loc H là số phức, có phân faor bằng 0
    H = np.ones((P,Q,2),np.float32)
    H[:,:,1] = 0.0
    D0 = 7
    D1 = 7
    for u in range(0, P):
        for v in range(0, Q):
            if not u in range(Q//2-D1, Q//2+D1):
                D = v-Q//2
                if abs(D) <= D0:
                    H[u,v,0] = 0
    return H

def FrequencyFiltering(imgin,H):
    M,N = imgin.shape
    # Bước 1 và 2: Mở rộng và thêm số 0 vòa phần mở rộng
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = 1.0*imgin

    # Bước 3: nhan fp với (-1)^(x+y)
    for x in range (0,M):
        for y in range (0,N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]

    # Bước 4: DFT
    F = cv2.dft(fp,flags= cv2.DFT_COMPLEX_OUTPUT)

    # Bước 5: Nhân F với H
    G = cv2.mulSpectrums(F,H,flags=cv2.DFT_ROWS)

    # Bước 6: IDFT
    g = cv2.idft(G, flags= cv2.DFT_SCALE)

    # Bước 7: Bỏ phần thêm vào, nhân với (-1)^(x+y)
    gR = g[:M,:N,0]
    for x in range(0,M):
        for y in range(0,N):
            if (x+y)%2 == 1:
                gR[x,y] = -gR[x,y]
    gR = np.clip(gR, 0, L-1)
    imgout = gR.astype(np.uint8)
    return imgout

def RemoveMoireSimple(imgin):
    M,N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H = CreateNotchFilter(P,Q)
    imgout = FrequencyFiltering(imgin,H)
    return imgout

def RemoveMoire(imgin):
    M,N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H = CreateButterworthNotchRejectFilter(P,Q)
    imgout = FrequencyFiltering(imgin,H)
    return imgout

def RemoveInterference(imgin):
    M,N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    H = CreateVerticalNotchRejectFilter(P,Q)
    imgout = FrequencyFiltering(imgin,H)
    return imgout

# Danh sách các phương pháp xử lý
option = st.selectbox("Chọn phương pháp xử lý", [
    "Spectrum",
    "Remove Moire Simple",
    "Remove Moire",
    "Remove Interference"
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
        if option == "Spectrum":
            output = Spectrum(gray)
            st.image(output, caption="Spectrum", use_container_width=True, clamp=True)
        if option == "Remove Moire Simple":
            output = RemoveMoireSimple(gray)
            st.image(output, caption="Remove Moire Simple", use_container_width=True, clamp=True)
        if option == "Remove Moire":
            output = RemoveMoire(gray)
            st.image(output, caption="Remove Moire", use_container_width=True, clamp=True)
        if option == "Remove Interference":
            output = RemoveInterference(gray)
            st.image(output, caption="Remove Interference", use_container_width=True, clamp=True)
