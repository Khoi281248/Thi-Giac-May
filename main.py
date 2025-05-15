import streamlit as st
import os
import base64

st.set_page_config(page_title=" Thị Giác Máy", layout="wide")

st.sidebar.image("images/logo.png", use_container_width=True)

st.markdown("""
    <style>
    .stApp {
    }
    .stSidebar {
        background: linear-gradient(to bottom, #006400, #2e8b57, #228b22) !important;  /* Green gradient */
        border-right: 1px solid #4e54c8;
    }
    .css-1d391kg, .st-b7, .st-b8, .st-b9 {
        background-color: rgba(0, 100, 0, 0.8) !important;  /* Dark green background */
    }
    # .stTextInput>div>div>input, .stSelectbox>div>div>select {
    #     background-color: rgba(255, 255, 255, 0.1) !important;
    #     color: white !important;
    # }
    # h1, h2, h3, h4, h5, h6 {
    #     color: #32cd32 !important;  /* Green text */
    }
    .stButton>button {
        background: linear-gradient(to right, #000000, #444444);  /* Black to dark gray gradient */
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #666666, #000000);  /* Lighter gray to black on hover */
    }
    </style>
""", unsafe_allow_html=True)


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

set_background("images/1.jpg")

PAGES_DIR = "pages"

# Khởi tạo session state
if "selected_page_path" not in st.session_state:
    st.session_state.selected_page_path = None
if "selected_section" not in st.session_state:
    st.session_state.selected_section = None  # sẽ chứa tên phần như "Thị giác máy"

def run_page(page_path):
    if os.path.exists(page_path):
        with open(page_path, "r", encoding="utf-8") as f:
            code = f.read()
            try:
                exec(code, globals())
            except Exception as e:
                st.error(f"❌ Đã xảy ra lỗi khi chạy mã: {str(e)}")
    else:
        st.error(f"❌ Không tìm thấy file: {page_path}")

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: black; font-size: 20px;'>Điều hướng chức năng</h2>",
            unsafe_allow_html=True)

    if st.button(" Giao Diện Chính"):
        st.session_state.selected_page_path = None
        st.session_state.selected_section = None
        st.rerun()

    # Các phần 1 đến 4
    for i in range(1, 5):
        folder_name = f"{i}_"
        for folder in sorted(os.listdir(PAGES_DIR)):
            if folder.startswith(folder_name):
                folder_path = os.path.join(PAGES_DIR, folder)
                page_path = os.path.join(folder_path, "page.py")
                if os.path.exists(page_path):
                    if st.button(f" {folder.replace('_', ' ')}"):
                        st.session_state.selected_page_path = page_path
                        st.session_state.selected_section = None
                        st.rerun()

    # Phần 5 - Thị Giác Máy
    if st.button(" 5 Thị giác máy"):
        st.session_state.selected_page_path = None
        st.session_state.selected_section = "Thị giác máy"
        st.rerun()

    # Các chương con trong Thị Giác Máy
    if st.session_state.selected_section == "Thị giác máy":
        vision_base = os.path.join(PAGES_DIR, "5_Thị_giác_máy")
        for chapter in ["Chương_3", "Chương_4", "Chương_9"]:
            chapter_path = os.path.join(vision_base, chapter, "page.py")
            label = chapter.replace("_", " ")
            if st.button(f" {label}"):
                st.session_state.selected_page_path = chapter_path
                st.rerun()

    # Phần 6 - Tùy chọn
    if st.button(" 6 Tùy chọn"):
        st.session_state.selected_page_path = None
        st.session_state.selected_section = "Tùy chọn"
        st.rerun()

     # Các chương con trong Tùy chọn
    if st.session_state.selected_section == "Tùy chọn":
        vision_base = os.path.join(PAGES_DIR, "6_Tùy_chọn")
        for chapter in ["Cảnh_báo_ngủ_gật", "Cử_chỉ_tay"]:
            chapter_path = os.path.join(vision_base, chapter, "page.py")
            label = chapter.replace("_", " ")
            if st.button(f" {label}"):
                st.session_state.selected_page_path = chapter_path
                st.rerun()

# Giao diện chính
if st.session_state.selected_page_path:
    # st.info(f" Đang tải trang: {st.session_state.selected_page_path}")
    run_page(st.session_state.selected_page_path)

elif st.session_state.selected_section == "Thị giác máy" and st.session_state.selected_page_path is None:
    # Khi chọn phần "Thị giác máy" nhưng chưa chọn chương nào
    st.markdown("<h1 style='text-align: center;font-size: 50px; color: blue;'> THỊ GIÁC MÁY</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:black;font-size: 50px;'> Hãy chọn một chương ở menu bên trái để bắt đầu.</h3>", unsafe_allow_html=True)

elif st.session_state.selected_section == "Tùy chọn" and st.session_state.selected_page_path is None:
    # Khi chọn phần "Thị giác máy" nhưng chưa chọn chương nào
    st.markdown("<h1 style='text-align: center;font-size: 50px; color: blue;'> TÙY CHỌN</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:black;font-size: 50px;'> Hãy chọn một chương ở menu bên trái để bắt đầu.</h3>", unsafe_allow_html=True)


else:

    # --- Tiêu đề ---
    st.markdown("<h1 style='text-align: center;font-size: 50px; color: blue;'>DỰ ÁN THỊ GIÁC MÁY</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Thông tin thành viên ---
    st.markdown("<h3 style='color: black;font-size: 30px;'>Thành Viên Thực Hiện</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='color:black;font-size: 30px;'> Thành viên 1</h3>", unsafe_allow_html=True)
        if os.path.exists("images/anh1.jpg"):
            st.image("images/anh1.jpg", width=200)
        else:
            st.warning("Không tìm thấy ảnh: anh1.jpg")
        st.markdown(
            "<h2 style='color: black; font-size: 30px;'>Họ tên: Trần Nguyên Khôi</h2>",
            unsafe_allow_html=True
)
        st.markdown(
            "<h2 style='color: black; font-size: 30px;'>MSSV: 22146158</h2>",
            unsafe_allow_html=True
)
    with col2:
        st.markdown("<h3 style='color:black;'> Thành viên 2</h3>", unsafe_allow_html=True)
        if os.path.exists("images/anh2.jpg"):
            st.image("images/anh2.jpg", width=200)
        else:
            st.warning("⚠️ Không tìm thấy 'anh2.jpg'.")
        st.markdown(
            "<h2 style='color: black; font-size: 30px;'>Họ tên: Lê Thanh Thông</h2>",
            unsafe_allow_html=True
)
        st.markdown(
            "<h2 style='color: black; font-size: 30px;'>MSSV: 22146235</h2>",
            unsafe_allow_html=True
)

    # --- Footer ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: black;'>© 2025 - Nhóm Đồ Án Thị Giác Máy</h5>", unsafe_allow_html=True)
