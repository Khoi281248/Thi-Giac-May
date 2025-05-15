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

set_background("images/5_1.jpg")

def Negative(imgin):
    # M la do cao cua anh
    # N la do rong cua anh
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    # Quet anh
    for x in range (0, M):
        for y in range (0, N):
            r = imgin[x,y]
            s = L-1-r
            imgout[x,y] = np.uint8(s)
    return imgout

def NegativeColor(imgin):
    # C: Channel la so 3 cho anh mau
    M, N, C= imgin.shape
    imgout = np.zeros((M,N,C), np.uint8)
    for x in range (0,M):
        for y in range (0, N):
            b = imgin [x,y,0]
            g = imgin [x,y,1]
            r = imgin [x,y,2]

            b = L-1-b
            g = L-1-g
            r = L-1-r

            imgout [x,y,0] = np.uint8(b)
            imgout [x,y,1] = np.uint8(g)
            imgout [x,y,2] = np.uint8(r)

    return imgout

def Logarit(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    c = (L-1.0)/np.log(1.0*L)
    for x in range (0, M):
        for y in range (0, N):
            r = imgin[x,y]
            if r ==0:
                r=1
            s = c*np.log(1.0 + r)
            imgout[x,y]= np.uint8(s)
    return imgout

def Power(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    gamma = 5.0
    c = np.power(L-1.0,1-gamma)
    for x in range (0, M):
        for y in range (0, N):
            r = imgin[x,y]
            if r ==0:
                r=1
            s = c*np.power(1.0*r, gamma)
            imgout[x,y]= np.uint8(s)
    return imgout

def PiecewiseLine (imgin):
    M,N =imgin.shape
    imgout= np.zeros((M,N),np.uint8)
    rmin,rmax, _, _= cv2.minMaxLoc(imgin)
    r1=rmin
    s1=0
    r2= rmax
    s2=L-1
    for x in range (0,M):
        for y in range(0,N):
            r = imgin[x,y]
            #doan 1
            if r<r1:
                s=s1/r1*r
            #doan 2
            elif r<r2:
                s=1.0*(s2-s1)/(r2-r1)*(r-r1)+s1
            else:
                s=1.0*(L-1-s2)/(L-1-r2)*(r-r2)+s2
            imgout [x,y]=np.uint8(s)
    return imgout

def Histogram(imgin):
    M,N= imgin.shape
    imgout= np.zeros((M,L,3),np.uint8) 
    imgout[:,:,] = np.array([255,255,255],np.uint8)
    h = np.zeros(L,np.int32)
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            h[r]=h[r]+1
    p=1.0*h/(M*N)
    scale=3000
    for r in range(0,L):
        cv2.line(imgout,(r,M-1),(r,M-1-np.int32(scale*p[r])),(255,0,0))
    return imgout

def HistEqual(imgin):
    M,N= imgin.shape 
    imgout= np.zeros((M,N),np.uint8) 
    h = np.zeros(L,np.int32)
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            h[r]=h[r]+1
    p=1.0*h/(M*N)
    s = np.zeros(L,np.float64)

    for k in range(0,L):
        for j in range(0, k+1):
            s[k] = s[k] + p[j]
    
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            imgout[x,y] = np.uint8((L-1)*s[r])
    return imgout

def HistEqualColor(imgin):
    # Ảnh màu của opencv là BGR
    # Ảnh màu của pillow là RGB
    # Pillow là thư viện ảnh của python
    b = imgin[:,:,0]
    g = imgin[:,:,1]
    r = imgin[:,:,2]

    b= cv2.equalizeHist(b)
    g= cv2.equalizeHist(g)
    r= cv2.equalizeHist(r)

    imgout= imgin.copy()
    imgout[:,:,0]= b
    imgout[:,:,1]= g
    imgout[:,:,2]= r
    return imgout

def LocalHist(imgin):
    M,N= imgin.shape
    imgout= np.zeros((M,N),np.uint8)
    m=3
    n=3
    a= m//2
    b=n//2 
    for x in range(a,M-a):
        for y in range(b,N-b):
            w = imgin[x-a:x+a+1,y-b:y+b+1]
            w = cv2.equalizeHist(w)
            imgout[x,y]=w[a,b]
    return imgout

def HistStat(imgin):
    M,N= imgin.shape
    imgout= np.zeros((M,N),np.uint8)
    mean, stddev = cv2.meanStdDev(imgin)
    mG= mean[0,0]
    sigmaG= stddev[0,0]
    m=3
    n=3
    a= m//2
    b=n//2 

    C=22.8
    k0=0.0
    k1=0.1
    k2=0.0
    k3= 0.1

    for x in range(a,M-a):
        for y in range(b,N-b):
            w = imgin[x-a:x+a+1,y-b:y+b+1]
            mean, stddev = cv2.meanStdDev(w)
            msxy=mean[0,0]
            sigmasxy= stddev[0,0]
            if (k0*mG <=msxy <= k1*mG) and (k2*mG <= sigmasxy <=k3*sigmaG):
                imgout[x,y]=np.uint8(C*imgin[x,y])
            else:
                imgout[x,y]=imgin[x,y]
    return imgout

def Sharp(imgin):
    w = np.array([[1,1,1],[1,-8,1],[1,1,1]],np.float32)
    Laplace= cv2.filter2D(imgin,cv2.CV_32FC1,w)
    imgout = imgin -Laplace
    imgout = np.clip(imgout,0,L-1)
    imgout = imgout.astype(np.uint8)
    return imgout

def Gradient(imgin):
    Sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],np.float32)
    Sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32)
    
    gx = cv2.filter2D(imgin,cv2.CV_32FC1,Sobel_x)
    gy = cv2.filter2D(imgin,cv2.CV_32FC1,Sobel_y)

    imgout = abs(gx) + abs(gy)

    imgout = np.clip(imgout,0,L-1)
    imgout = imgout.astype(np.uint8)
    return imgout

# Danh sách các phương pháp xử lý
option = st.selectbox("Chọn phương pháp xử lý", [
    "Negative (Grayscale)", 
    "NegativeColor (RGB)",
    "Logarit", 
    "Power", 
    "Piecewise Line", 
    "Histogram", 
    "Histogram Equalization", 
    "Histogram Equalization (Color)", 
    "Local Histogram Equalization", 
    "Histogram Statistics", 
    "Smooth Box",
    "Smooth Gauss",
    "Median filter",
    "Sharpen", 
    "Gradient"
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
        if option == "Negative (Grayscale)":
            output = Negative(gray)
            st.image(output, caption="Ảnh âm bản", use_container_width=True, clamp=True)
        elif option == "NegativeColor (RGB)":
            output = NegativeColor(img)
            st.image(output, caption="Ảnh âm bản màu", use_container_width=True, clamp=True)
        elif option == "Logarit":
            output = Logarit(gray)
            st.image(output, caption="Logarit", use_container_width=True, clamp=True)
        elif option == "Power":
            output = Power(gray)
            st.image(output, caption="Power Law", use_container_width=True, clamp=True)
        elif option == "Piecewise Line":
            output = PiecewiseLine(gray)
            st.image(output, caption="Piecewise Linear", use_container_width=True, clamp=True)
        elif option == "Histogram":
            output = Histogram(gray)
            st.image(output, caption="Histogram", use_container_width=True, clamp=True)
        elif option == "Histogram Equalization":
            output = HistEqual(gray)
            st.image(output, caption="Hist Equalization", use_container_width=True, clamp=True)
        elif option == "Histogram Equalization (Color)":
            output = HistEqualColor(img)
            st.image(output, caption="Hist Equalization (Color)", use_container_width=True, clamp=True)
        elif option == "Local Histogram Equalization":
            output = LocalHist(gray)
            st.image(output, caption="Local Histogram", use_container_width=True, clamp=True)
        elif option == "Histogram Statistics":
            output = HistStat(gray)
            st.image(output, caption="Histogram Statistics", use_container_width=True, clamp=True)
        elif option == "Smooth Box":
            output = cv2.boxFilter(gray, cv2.CV_8UC1,(21,21))
            st.image(output, caption="Smooth Box", use_container_width=True, clamp=True)
        elif option == "Smooth Gauss":
            output = cv2.GaussianBlur(gray, (43,43),7.0)
            st.image(output, caption="Smooth Gauss", use_container_width=True, clamp=True)
        elif option == "Median filter":
            output = cv2.medianBlur(gray, 3)
            st.image(output, caption="Median filter", use_container_width=True, clamp=True)
        elif option == "Sharpen":
            output = Sharp(gray)
            st.image(output, caption="Sharpen", use_container_width=True, clamp=True)
        elif option == "Gradient":
            output = Gradient(gray)
            st.image(output, caption="Gradient Edge Detection", use_container_width=True, clamp=True)
