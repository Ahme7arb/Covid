import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
import cv2
from PIL import Image
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="COVID-19 Detector",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"}
)
# Load the model
model_path = 'D:\M\Covid19\covid_19_model.h5'
model = load_model(model_path)

class_mapping = {0: 'COVID-19',
                 1: 'Normal',
                 2: 'Pneumonia'}

x_resize = 224
y_resize = 224
ndim = 3

# Preprocess the image for prediction
def preprocess_image(image, x_resize, y_resize, ndim):
    """
    Preprocess the image for prediction.
    - Convert the image to a numpy array.
    - Resize the image to (224,224).
    - If the image is grayscale, convert it to RGB.
    - Normalize the image to [0,1].
    - Reshape the image to (1,224,224,3).
    - Return the preprocessed image.
    """
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (x_resize, y_resize))

    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)  # convert grayscale to RGB

    img_array = img_array.astype(np.float32) / 255.0  # normalize the image to [0,1]
    new_img_array = img_array.reshape(1, x_resize, y_resize, ndim)  # reshape the image to (1,224,224,3)

    return new_img_array
# Predict the image
def predict_image(image):
    prediction = model.predict(image)  # output --> softmax probabilities for each class
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    return class_mapping[predicted_class], confidence, prediction[0]
# Sidebar
st.sidebar.title("🩺 About")
st.sidebar.info("""
This AI-powered application analyzes chest X-ray images to detect:
- **COVID-19**: Signs of coronavirus infection
- **Normal**: Healthy lung appearance
- **Pneumonia**: Bacterial or viral pneumonia

**Note**: This tool is for educational purposes only and should not replace professional medical diagnosis.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with:**")
st.sidebar.markdown("- TensorFlow/Keras")
st.sidebar.markdown("- Streamlit")
st.sidebar.markdown("- OpenCV & PIL")

# Main content
st.title("🩺 COVID-19 Detection from Chest X-ray Images")
st.markdown("### Upload a chest X-ray image to get AI-powered analysis")

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    img_file = st.file_uploader(
        "Choose a chest X-ray image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear chest X-ray image for analysis"
    )

    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

with col2:
    st.subheader("🔍 Analysis Results")

    if img_file is not None:
        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image... This may take a few seconds."):
                preprocessed_image = preprocess_image(image, x_resize, y_resize, ndim)
                prediction, confidence, probs = predict_image(preprocessed_image)

            # Display results
            if prediction == 'COVID-19':
                st.error(f"**Predicted Class:** {prediction}")
            elif prediction == 'Normal':
                st.success(f"**Predicted Class:** {prediction}")
            else:
                st.warning(f"**Predicted Class:** {prediction}")

            st.info(f"**Confidence Score:** {confidence:.2%}")

            # Prediction probabilities chart
            st.markdown("### Prediction Probabilities")
            prob_data = pd.DataFrame({
                'Condition': list(class_mapping.values()),
                'Probability': [f"{p:.1%}" for p in probs]
            })
            st.dataframe(prob_data, use_container_width=True)

            # Bar chart
            chart_data = pd.DataFrame({
                'Condition': list(class_mapping.values()),
                'Probability': probs
            })
            st.bar_chart(chart_data.set_index('Condition'), use_container_width=True)
    else:
        st.info("👆 Please upload an image in the left column to begin analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>
        This application uses deep learning to assist in medical image analysis.<br>
        Always consult with healthcare professionals for medical decisions.
    </small>
</div>
""", unsafe_allow_html=True)