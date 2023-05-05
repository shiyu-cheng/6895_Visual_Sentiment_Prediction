import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from torchvision import models
import torch.nn as nn
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import transforms, models, datasets
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from collections import Counter
import pandas as pd
import base64

def background(bg):
   bg_ext = 'png'
   st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{bg_ext};base64,{base64.b64encode(open(bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
      unsafe_allow_html=True,
      )
      
bg = '/Users/uuu/Desktop/CU/S23/6895/bg.png'
background(bg)

st.image('/Users/uuu/Desktop/CU/S23/6895/icon.png')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_labels =['Aggressiveness',
 'Anger',
 'Awe',
 'Anticipation',
 'Boredom',
 'Contempt',
 'Disapproval',
 'Disgust',
 'Envy',
 'Fear',
 'Guilt',
 'Indifferent',
 'Irritation',
 'Joy',
 'Love',
 'Optimism',
 'Pessimism',
 'Remorse',
 'Sadness',
 'Serenity',
 'Shame',
 'Submission',
 'Surprise',
 'Trust']

@st.cache_resource()
def load_model():
    model = models.resnet50(pretrained=False)  #Use the resnet18 model
    num_ftrs =model .fc.in_features #Modify the number of model categories
    model.fc = nn.Sequential(nn.Dropout(0.9),nn.Linear(num_ftrs, 24))
    model= model.to(device)
    checkpoint = torch.load('/Users/uuu/Desktop/CU/S23/6895/model.pkl')
    model.load_state_dict(checkpoint['net'])
    return model
  
with st.spinner('Model is being loaded..'):
  model=load_model()

title = '<p style="font-family:sans-serif;font-size: 50px;font-weight:800">Emotion Image Classification</p>'
st.markdown(title, unsafe_allow_html=True)

# upload = '<p style="font-family:sans-serif;font-size: 18px;font-weight:800">Please Upload the Image to be Classified</p>'
# st.markdown(upload, unsafe_allow_html=True)
 
file = st.file_uploader(" ", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)
 
def upload_predict(upload_image, model):

    transform = transforms.Compose([
    transforms.Resize(256),        # Resize the image to 256x256 pixels
    transforms.CenterCrop(224),   # Crop the center 224x224 pixels
    transforms.ToTensor(),        # Convert the image to a PyTorch tensor
    ])

    image = transform(upload_image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image.to(device))

    _, predicted = torch.max(output.data, 1)
    predicted_label = class_labels[predicted.item()]

    return predicted_label

if file is None:
    pass
else:
    image = Image.open(file)
    image = image.convert("RGB")
    st.image(image, use_column_width=True)
    predictions = upload_predict(image, model)
    st.write("The image is classified as",predictions)