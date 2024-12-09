import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import io

# Load your pre-trained model
@st.cache_resource
def load_model():
    model = torch.load("best_model.pth", map_location=torch.device('cpu'))  # Replace with your model's path
    model.eval()
    return model

model = load_model()

# Preprocessing function
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size to your model's input dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Replace with your model's normalization if needed
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Postprocessing function
def postprocess_output(output_tensor):
    output_image = output_tensor.squeeze(0).permute(1, 2, 0)  # Remove batch dimension and reorder channels
    output_image = output_image.detach().numpy()
    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())  # Normalize to [0, 1]
    output_image = (output_image * 255).astype("uint8")  # Convert to uint8
    return Image.fromarray(output_image)

# Streamlit app
st.title("Model-Based Image Processing App")
st.write("Upload an image to process it through a trained model!")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Process Image"):
        # Preprocess the image
        input_tensor = preprocess_image(input_image)
        
        # Pass the image through the model
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Postprocess the output
        processed_image = postprocess_output(output_tensor)
        
        # Display the processed image
        st.image(processed_image, caption="Processed Image", use_column_width=True)
        
        # Download option
        buf = io.BytesIO()
        processed_image.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="Download Processed Image",
            data=buf,
            file_name="processed_image.png",
            mime="image/png"
        )
