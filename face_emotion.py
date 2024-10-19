import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Load the trained model
model = torch.load('cnn_fer2013_64percent.pth')
model.eval()

# Define the labels
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Resize to the input size expected by your model
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Adjust normalization as per your training
])

def predict(image):
    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return labels[predicted.item()]

# Streamlit application
st.title("Facial Emotion Recognition")
st.write("Upload an image to predict the emotion.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale if necessary
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Predict the emotion
    if st.button("Predict"):
        emotion = predict(image)
        st.write(f"Predicted Emotion: {emotion}")