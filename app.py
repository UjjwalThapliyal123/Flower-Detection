import streamlit as st
import pickle
import json
import os
from PIL import Image
app_title = "My Streamlit App"

model_path = "flower_species_model (1).keras"
images_data_path = "flower_info.json"
class_names_path = "class_names.pkl"
image_size = "img_size.pkl"

def load_model(model_path):
    with open(model_path,'rb') as file:
        model = pickle.load(file)
    return model

with open(class_names_path,'rb') as file:
    file_p = pickle.load(file)

with open(image_size,'rb') as f:
    img_size = pickle.load(f)
    
@st.cache_data
def flower_data():
    if not os.path.exists(images_data_path):
        st.error("❌ flower_info.json not found")
        return {}
    with open(images_data_path,'r',encoding='UTF-8') as f:
        return json.load(f) 

# LOAD
model = load_model(model_path)
flower_info = flower_data() 

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1,2])
    with col1:
        st.image(image,caption="Uploaded Image",use_column_width=True)
 
        
        