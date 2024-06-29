import streamlit as st
from ultralytics import YOLOv10
from PIL import Image

MODEL_PATH = 'helmet_safety_best.pt'
model = YOLOv10(MODEL_PATH)

st.title('Safety Helmet Detection')
st.write('Upload an image and the model will predict the bounding boxes of the helmets')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('Uploaded Image')
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    if st.button('Predict'):
        with st.spinner('Predicting...'):
            results = model(image)
        
    with col2:
        st.write('Predicted Image')
        if 'results' in locals():
            st.image(results[0].plot(), caption='Predicted Image.', channels='BGR', use_column_width=True)