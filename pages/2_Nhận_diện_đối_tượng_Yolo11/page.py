import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import streamlit_webrtc as webrtc
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Load YOLOv11 model once with caching
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolo11n.pt')  # Ensure this file is in your working directory
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

def draw_boxes(image, results, class_names):
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        label = f"{class_names[class_ids[i]]} {confidences[i]:.2f}"

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for the bounding box

        # Calculate the size of the text
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw a green rectangle for the background of the text
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)  # Green background

        # Add the label with black text on top of the green background
        cv2.putText(image, label, (x1, y1 - 10 if y1 > 20 else y1 + text_height + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)  # Black for the text
    return image

class VideoTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def transform(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Run YOLOv11 detection
        results = self.model.predict(image_rgb)

        # Draw boxes on detected objects
        if results and results[0].boxes:
            processed_img = draw_boxes(image_bgr.copy(), results, self.model.names)
            return processed_img
        else:
            return image_bgr

# Initialize Streamlit interface
st.title("🔍 Object Detection with YOLOv11 and Webcam")

model = load_model()

if model is not None:
    st.write("📷 Choose to use your webcam or upload an image")

    option = st.radio("Select Input Mode", ["Upload Image", "Use Webcam"])

    if option == "Upload Image":
        uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png", "bmp"])

        if uploaded_file:
            image_bytes = uploaded_file.read()
            image_np = np.frombuffer(image_bytes, np.uint8)
            image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            if image_bgr is None:
                st.error("❌ Failed to read image.")
            else:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                # Show original image and detected image side by side using st.columns
                col1, col2 = st.columns(2)
                with col1:
                    st.image(Image.fromarray(image_rgb), caption="Original Image", use_container_width=True)

                if st.button("💡 Run YOLOv11 Detection"):
                    with st.spinner("Detecting..."):
                        results = model.predict(image_rgb)
                        if results and results[0].boxes:
                            processed_img = draw_boxes(image_bgr.copy(), results, model.names)
                            with col2:
                                st.image(Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)),
                                         caption="Detected Image", use_container_width=True)
                        else:
                            st.info("✅ No objects detected.")

    elif option == "Use Webcam":
        webrtc_streamer(
            key="yolo-webcam",
            video_transformer_factory=lambda: VideoTransformer(model),
            rtc_configuration={"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]},
        )

else:
    st.warning("⚠️ YOLOv11 model not loaded. Please check your model file and dependencies.")
