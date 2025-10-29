import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import json
import os

#Load class names 
@st.cache_data
def load_classes():
    with open('class_names.json', 'r') as f:
        return json.load(f)

#Load model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_classes = len(load_classes())
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load('models/plant_disease_resnet50.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Predict function
def predict(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = load_model()(img)
        pred_idx = torch.argmax(output, dim=1).item()
    return load_classes()[pred_idx]

#Streamlit UI
st.set_page_config(page_title="Crop Disease Detector", page_icon="🌱")

st.title("🌱 Crop Disease Detector")
st.write("Upload a leaf image to detect 38 types of crop diseases instantly!")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Leaf", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        disease = predict(image)
    
    st.success(f"**Predicted Disease:** `{disease}`")
    st.balloons()