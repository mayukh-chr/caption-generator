import streamlit as st
import numpy as np
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set page config
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load model
processor, model = load_model()

# Streamlit app
st.title("Image Caption Generator")
st.markdown(
    "Upload an image, and this app will generate a caption for it using a trained BLIP model."
)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.subheader("Generated Caption")
    
    # Display loading spinner while processing
    with st.spinner("Generating caption..."):
        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_image)
        image = image.convert('RGB')
        
        # Process the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Generate caption
        outputs = model.generate(**inputs)
        generated_caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Display the generated caption with custom styling
    st.markdown(
        f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
        f'<p style="font-style: italic;">"{generated_caption}"</p>'
        f'</div>',
        unsafe_allow_html=True
    )