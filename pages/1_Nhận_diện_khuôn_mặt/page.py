import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import os
import base64

def set_background1(image_file):
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

set_background1("images/2.jpg")

# Khởi động streamlit
st.title("NHẬN DIỆN KHUÔN MẶT")

# Khởi tạo trạng thái
if "run" not in st.session_state:
    st.session_state.run = False
if "cap" not in st.session_state:
    st.session_state.cap = None

# Load model
@st.cache_resource
def load_models():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    detector = cv.FaceDetectorYN.create(
        os.path.join(CURRENT_DIR, 'face_detection_yunet_2023mar.onnx'),
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )
    recognizer = cv.FaceRecognizerSF.create(
        os.path.join(CURRENT_DIR, 'face_recognition_sface_2021dec.onnx'),
        ""
    )
    svc = joblib.load(os.path.join(CURRENT_DIR, 'svc.pkl'))
    return detector, recognizer, svc

detector, recognizer, svc = load_models()
mydict = ['Doan', 'Khanh', 'Khoi', 'Tai','Thong']

# Hàm vẽ kết quả
def visualize(input, faces, names, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1][:5]):  # Tăng số lượng khuôn mặt nhận diện lên 5
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.putText(input, names[idx], (coords[0], coords[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return input

# Streamlit hiển thị video
stframe = st.empty()
tm = cv.TickMeter()

# Chọn đầu vào (webcam hoặc file)
input_type = st.radio("Chọn nguồn video", ("Webcam", "Tải lên video", "Tải lên ảnh"))

if input_type == "Webcam":
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Open Webcam") and not st.session_state.run:
            st.session_state.run = True
            st.session_state.cap = cv.VideoCapture(0)  # Chỉ mở video khi nhấn Start
            st.session_state.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
            st.session_state.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    with col2:
        if st.button("Close Webcam") and st.session_state.run:
            st.session_state.run = False
            st.session_state.cap.release()  # Đóng video khi nhấn Stop
            cv.destroyAllWindows()
        pass

# Xử lý video tải lên
elif input_type == "Tải lên video":
    video_file = st.file_uploader("Chọn video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(video_file.read())
        st.session_state.cap = cv.VideoCapture("uploaded_video.mp4")
        st.session_state.run = True


# Xử lý ảnh tải lên
elif input_type == "Tải lên ảnh":
    image_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "bmp"])
    if image_file is not None:
        img = cv.imdecode(np.frombuffer(image_file.read(), np.uint8), cv.IMREAD_COLOR)
        
        # Đặt kích thước đầu vào phù hợp với ảnh
        h, w = img.shape[:2]
        detector.setInputSize([w, h])

        faces = detector.detect(img)
        names = []
        if faces[1] is not None:
            for face in faces[1][:5]:  # Tăng số lượng khuôn mặt nhận diện lên 5
                face_align = recognizer.alignCrop(img, face)
                face_feature = recognizer.feature(face_align)

                similarity_score = svc.decision_function(face_feature.reshape(1, -1))
                max_similarity = np.max(similarity_score)

                threshold = 0.5
                if max_similarity > threshold:
                    test_predict = svc.predict(face_feature)
                    names.append(mydict[test_predict[0]])
                else:
                    names.append("Unknown")

        img = visualize(img, faces, names, 0)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        st.image(img, channels="RGB", use_container_width=True)

# Điều khiển video
if st.session_state.run and st.session_state.cap:
    frameWidth = int(st.session_state.cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(st.session_state.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    # Tạo nút "Stop Video" chỉ khi xử lý video
    if input_type != "Webcam":  # Hiển thị nút Stop chỉ khi xử lý video (không phải webcam)
        stop_button = st.button("Stop Video")
        if stop_button:
            st.session_state.run = False  # Dừng video khi nhấn Stop
            st.session_state.cap.release()  # Giải phóng tài nguyên video
            cv.destroyAllWindows()

    # Vòng lặp xử lý video
    while st.session_state.run:
        hasFrame, frame = st.session_state.cap.read()
        if not hasFrame:
            st.error("Không thể lấy frame từ video.")
            break  # Nếu không có frame, dừng vòng lặp

        tm.start()
        faces = detector.detect(frame)
        tm.stop()

        names = []
        if faces[1] is not None:
            for face in faces[1][:5]:  # Tăng số lượng khuôn mặt nhận diện lên 5
                face_align = recognizer.alignCrop(frame, face)
                face_feature = recognizer.feature(face_align)

                similarity_score = svc.decision_function(face_feature.reshape(1, -1))
                max_similarity = np.max(similarity_score)

                threshold = 0.5
                if max_similarity > threshold:
                    test_predict = svc.predict(face_feature)
                    names.append(mydict[test_predict[0]])
                else:
                    names.append("Unknown")

        frame = visualize(frame, faces, names, tm.getFPS())

        # Chuyển BGR sang RGB cho Streamlit
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    # Giải phóng tài nguyên video khi kết thúc
    st.session_state.cap.release()
    cv.destroyAllWindows()
