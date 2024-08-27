import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your trained model
model = load_model(r'C:\Users\hazem\OneDrive\Desktop\Computer vision\Week-2\Teeth_Disease_Classifier\VGG16\vgg16_model.h5')

# Define image size
img_height, img_width = 224, 224  # Adjust these to your model's expected input size

# Class labels
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT'] 

# Streamlit app interface
st.title("🦷 Teeth Disease Classifier")
st.markdown("Upload an image of teeth to classify the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Add a button to initiate classification
    if st.button("Classify"):
        with st.spinner('Classifying...'):
            # Preprocess the image
            img = img.resize((img_width, img_height))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            

            # Make prediction
            predictions = model.predict(img_array)
            score = np.max(predictions[0])
            predicted_class = class_names[np.argmax(predictions[0])]

        # Display prediction
        st.success(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{score:.2f}**")

        # Display probability distribution
        st.subheader("Class Probabilities")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {predictions[0][i]:.2f}")

else:
    st.write("Please upload an image file to classify.")

# Footer
st.markdown("---")
st.markdown("🔍 **Note**: This tool is intended for educational purposes and should not be used for actual medical diagnosis.")
