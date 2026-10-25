import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image

# Load the model with pre-trained weights (cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    weights = EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=weights)
    model.eval()
    return model, weights.transforms()

model, preprocess = load_model()

st.title("EfficientNet-B3 Image Classifier")
st.write("Upload an image to get a real prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner('Running prediction...') :
        # Preprocess the image and add batch dimension
        img_tensor = preprocess(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            prediction = model(img_tensor).squeeze(0).softmax(0)
            
        # Get the top 3 categories
        weights = EfficientNet_B3_Weights.DEFAULT
        categories = weights.meta["categories"]
        top3_prob, top3_catid = torch.topk(prediction, 3)
        
        st.subheader("Top 3 Predictions:")
        for i in range(top3_prob.size(0)):
            st.write(f"**{categories[top3_catid[i]]}**: {top3_prob[i].item() * 100:.2f}%")
