import streamlit as st
import yaml
import torch
from model.vqa_model import ViBlipVQAModel
from PIL import Image
import os

# Load config
with open('KLTN_demo/config/viblip_text_config.yaml') as conf_file:
    config = yaml.safe_load(conf_file)

cuda_device = config['train']['cuda_device']
device = torch.device(f'{cuda_device}' if torch.cuda.is_available() else 'cpu')

# Initialize model once
if 'model' not in st.session_state:
    st.session_state.model = ViBlipVQAModel(config)
    st.session_state.model.to(device)
    st.session_state.model.eval()  # Set model to evaluation mode

st.set_page_config(layout="wide")  # Adjust layout for wide screen

st.title("Simple Vietnamese Visual Question Answering App")

# Initialize session variables for image
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Layout for image and question side by side
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Only save the image if it's not already saved in session state
        image_name = f"temp_{uploaded_file.name}"
        if image_name not in st.session_state:
            image = Image.open(uploaded_file)
            st.session_state[image_name] = image
            if not os.path.exists(image_name):
                image.save(image_name)
        st.session_state.current_image = image_name  # Store current image key in session


    # Render the current image
    if st.session_state.current_image:
        image_path = st.session_state.get(st.session_state.current_image)
        st.image(image_path, caption="Ảnh đang sử dụng", width=500)

with col2:
    question = st.text_input("Nhập câu hỏi về ảnh:")

    # Display the prediction result
    if st.button("Dự đoán"):
        if question:
            try:
                # Make prediction using the current image and the input question
                pred_result = st.session_state.model(question, st.session_state.current_image)
                st.success(f"Kết quả: {pred_result}")
            except Exception as e:
                st.error(f"Lỗi: {e}")
        else:
            st.warning("Vui lòng nhập câu hỏi trước khi dự đoán.")
