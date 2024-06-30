import os
import base64
import streamlit as st

from utils import save_upload_file, delete_file, download_model
from models.yolov10.detector import inference
from config.model_config import DetectorConfig


@st.cache_resource(max_entries=1000)
def process_inference(image_path):
    result_img = inference(
        image_path, weight_path=DetectorConfig().weight_path)
    return result_img


def main():
    st.set_page_config(
        page_title='Safety Helmet Detection',
        page_icon='👷',
        layout='wide'
    )

    st.title(':construction_worker: Safety Helmet Detection')
    st.text('Model: Helmet Safety Detector YOLOv10')

    uploaded_img = st.file_uploader(
        'Upload an image...', type=['jpg', 'jpeg', 'png'])
    predict_btn = st.button('Predict')

    st.divider()
    col1, col2 = st.columns(2)
    result_img = None
    with col1:
        if uploaded_img:
            st.markdown('**Uploaded Image**')
            st.image(uploaded_img, caption='Uploaded Image', use_column_width=True)
        if predict_btn and uploaded_img:
            result_img = process_inference(uploaded_img)
    with col2:
        if result_img is not None:
            st.markdown('**Detected Image**')
            st.image(result_img, caption='Detected Image', use_column_width=True)

    
    

if __name__ == '__main__':
    if not os.path.exists(DetectorConfig().weight_path):
        download_model()
    main()

# MODEL_PATH = 'helmet_safety_best.pt'
# model = YOLOv10(MODEL_PATH)

# st.title('Safety Helmet Detection')
# st.write('Upload an image and the model will predict the bounding boxes of the helmets')

# uploaded_file = st.file_uploader(
#     "Choose an image...", type=['jpg', 'jpeg', 'png'])

# if uploaded_file is not None:
#     col1, col2 = st.columns(2)

#     with col1:
#         st.write('Uploaded Image')
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image.', use_column_width=True)

#     if st.button('Detect'):
#         with st.spinner('Detecting...'):
#             results = model(image)

#     with col2:
#         st.write('Detected Image')
#         if 'results' in locals():
#             st.image(results[0].plot(), caption='Detected Image.',
#                      channels='BGR', use_column_width=True)
