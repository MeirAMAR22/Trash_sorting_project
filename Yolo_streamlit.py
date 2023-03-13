import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
import cv2
import tensorflow as tf
assert tf.__version__.startswith('2')

model_path = "./runs/detect/train/weights/best.pt"


header = st.container()
dataset = st.container()
modelVal = st.container()
model = st.container()

@st.cache
def loadmodel(path):
    model = YOLO(path) # "./runs/detect/train/weights/best.pt
    return model



with header:
    st.title('AcoVision - YOLOv8n model')
    st.text('AcoVision is a garbage object detection app, adapted to the israeli recycling system.\n'
            'The app is based on the pretrained YOLOv8n model, trained an a custom dataset.')

with dataset:
    st.header('Dataset analysis')

with modelVal:
    st.header('Validation metric')

with model:
    st.header('YOLOv8n')
    model = YOLO(model_path)
    # results = model.predict(source=img, save=True)  # save plotted images
    # display(Image.open('runs/detect/predict3/image0.jpg'))
