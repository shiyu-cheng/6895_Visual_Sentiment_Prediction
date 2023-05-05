# +
from __future__ import print_function, division

import streamlit as st
import numpy as np
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms, models, datasets
import torch.optim as optim
import time
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from collections import Counter
from sklearn.metrics import confusion_matrix
import pandas as pd
import itertools
from sklearn.preprocessing import LabelEncoder

import base64
import io
import requests
import random
import time
import cv2
import pdb
import shap
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import itertools

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from skimage.segmentation import mark_boundaries
from torch.autograd import Variable
from lime import lime_image

import os
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import urllib.parse

# Blip
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base") #.to("cuda")


# -

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

bg = 'bg.png'
background(bg)

st.image('icon.jpg')

features_blobs = []

# +
# @st.cache_resource
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
    
# @st.cache_resource
def is_file_path(path):
    """Checks if a string is a path to a file."""
    return os.path.isfile(path)

# @st.cache_resource
def is_url(url):
    """Checks if a string is a URL."""
    return urllib.parse.urlparse(url).scheme != ''

@st.cache_resource
def return_cam(feature_conv, weight_softmax, class_idx, model_input_size):
    # generate the class activation maps upsample to 256x256
    size_upsample = (model_input_size, model_input_size)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

# @st.cache_resource
# def crop_center(img, size):
#     """
#     Crop the center square of an image and resize it to the specified size
#     """
#     h, w = img.shape[:2]
#     crop_size = min(h, w)
#     start_h = (h - crop_size) // 2
#     start_w = (w - crop_size) // 2
#     img_cropped = img[start_h:start_h+crop_size, start_w:start_w+crop_size]
#     img_resized = cv2.resize(img_cropped, (size, size))
#     return img_resized

# @st.cache_resource
def calculate_cam(model, features_blobs, raw_image, classes,  model_input_size):
    # get the softmax weight
    params = list(model.cpu().parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    
    # preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(model_input_size),        # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),   # Crop the center 224x224 pixels
        transforms.ToTensor()        # Convert the image to a PyTorch tensor
    ])
    img = raw_image
    img_tensor = preprocess(img)
    img_variable = Variable(img_tensor.unsqueeze(0))
    
    # get model prediction
    logit = model(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    
    # generate class activation mapping for the top1 prediction
    CAMs = return_cam(features_blobs[0], weight_softmax, [idx[0]], model_input_size)
    
    # render the CAM and output
    img = cv2.resize(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (model_input_size, model_input_size))
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(CAMs[0], cv2.COLORMAP_JET) #cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite("CAM.jpg", result)
    del features_blobs[:]
    return Image.open("CAM.jpg"), classes[idx[0]]


# -

with st.spinner('Model is being loaded..'):
    PATH = os.path.join("models", 'm-resnext50' + ".pt")
#     model = torch.load(PATH).to('cpu')
    model = torch.load(PATH, map_location=torch.device('cpu'))

title = '<p style="font-family:sans-serif;font-size: 50px;font-weight:800">Emotion Image Classification</p>'
st.markdown(title, unsafe_allow_html=True)

upload = '<p style="font-family:sans-serif;font-size: 18px;font-weight:800">Please Upload the Image to be Classified</p>'
st.markdown(upload, unsafe_allow_html=True)

img_path = st.file_uploader(" ", type=["jpg", "png", "jpeg", "webp"])
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

# +
if img_path is None:
    pass
else:
    model._modules.get("layer4").register_forward_hook(hook_feature)

#     if is_url(img_path):
#         urllib.request.urlretrieve(img_path, "image.jpg")
#         img_path = "image.jpg"
        
    raw_image = Image.open(img_path).convert('RGB')
    st.image(raw_image, use_column_width=True)
#     st.write(type(img_path))

    label_map = ['Aggressiveness', 'Anger', 'Awe', 'Boredom', 'Disgust', 'Envy', 'Fear', 'Guilt', 
                         'Irritation', 'Joy', 'Love', 'Sadness', 'Serenity', 'Shame', 'Surprise', 'Trust']
    img, pred = calculate_cam(model, features_blobs, raw_image, label_map, 256)
    
    # Blub: unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt") #.to("cuda")
    out = blip_model.generate(**inputs)
    
    col1, col2 = st.columns(2)
    with col1:
#        st.header("A cat")
       st.image(img, use_column_width=True)

    with col2:
#        st.header("A dog")
#        st.image("https://static.streamlit.io/examples/dog.jpg")
        st.write("## Image sentiment: {}".format(pred))
        st.write(processor.decode(out[0], skip_special_tokens=True))

