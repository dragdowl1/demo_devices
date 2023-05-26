
import os
import cv2
import ssl
import torch
import base64
import numpy as np
import pandas as pd
from PIL import Image 
import streamlit as st
from io import BytesIO
import matplotlib.pyplot as plt
from streamlit_image_select import image_select
from warnings import filterwarnings, simplefilter

filterwarnings("ignore")
simplefilter(action='ignore', category=FutureWarning)


def load_model():
  model = torch.hub.load(repo_or_dir= 'yolov5', model = 'custom', path = 'best.pt', source = 'local', force_reload = True)
  model.conf = 0.5
  return model

def detect_image(image_file, model):

  image = Image.open(image_file)
  im = image.convert('RGB') 
  im = np.array(image)
  im_size = im.shape 
  img = im[:, :, ::-1].copy()
  out = model(img, size=640)
  results = [
    [
        {
            "class": int(pred[5]),
            "class_name": model.model.names[int(pred[5])],
            "bbox": [int(x) for x in pred[:4].tolist()],  
            "confidence": np.round(float(pred[4]),2),
        }
        for pred in result
    ]
    for result in out.xyxy
    ]
  
  df = pd.DataFrame.from_dict(results[0])
  coord = df['bbox'].to_numpy()
  img_out = np.array(image.copy())
  colors = {0: (255,0,0), 1: (0,255,0), 2: (0,0,255), 
      3: (128,128,128), 4: (128,0,0), 5: (128,128,0),
      6: (0,128,0), 7: (128,0,128), 8: (0,128,128)}
  
  for i in range(len(df)):
    class_ind = df['class'][i]
    class_name = df['class_name'][i]
    conf = df['confidence'][i]
    x1, y1, x2, y2 = coord[i]
    bgr = colors[class_ind]
    img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), bgr, 2)
    img_out  = cv2.putText(img_out, class_name+": "+str(round(conf,2)), (x1, y1-3), cv2.FONT_HERSHEY_COMPLEX, 3, bgr, 3)

  return  image, img_out, df['class_name'].tolist(), df['confidence'].tolist()


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;

    }}
    </style>
    """,
    unsafe_allow_html=True
    )

curr_dir = os.getcwd()
add_bg_from_local(curr_dir+"/images/back1.png") 

st.title("Malicious device detection")
st.info("Object detection for nine potentially malicious devices with hidden Wi-fi cameras: alarms, speakers, wall clocks, power outlets, chargers, doorbells, keyfobs, pens and fire detectors.")
model = load_model()

# st.write("Select an image from examples below -> then press the Start button for detection")

img = image_select(
    label="Select an example image",
    images=[
         
         curr_dir+"/images/image1.jpg",
         curr_dir+"/images/image2.JPG",
         curr_dir+"/images/image3.jpeg"
        # Image.open("images/cat3.jpeg"),
        # np.array(Image.open("images/cat4.jpeg")),
    ],
    captions=["Example 1", "Example 2", "Example 3"],
)



st.write("Selected image:", img.split('/')[-1])
image_file = img

press_but = st.button('Start')

if not image_file is None and press_but:                                          
    st.write('Detecting...')                            
    # image = Image.open(image_file) 
    im, im_out, names, confs = detect_image(image_file, model)                                  
    col1, col2 = st.columns(2)                                     
    col1.text('Original image:')
    col1.image(im)                                           
    col2.text('Result:')
    col2.image(im_out)
    for i in range(len(names)):
      st.write(names[i], confs[i])
                                       

st.write("Try it on your own image:")
im_file = st.file_uploader('Load your image', type=['png', 'jpg', 'jpeg', 'JPG']) 
press_but1 = st.button('Try it now!')
image_file = im_file

if not image_file is None and press_but1:                                          
    st.write('Detecting...')                            
    # image = Image.open(image_file) 
    im, im_out, names, confs = detect_image(image_file, model)                                  
    col1, col2 = st.columns(2)                                     
    col1.text('Original image:')
    col1.image(im)                                           
    col2.text('Result:')
    col2.image(im_out)
    for i in range(len(names)):
      st.write(names[i], confs[i])
elif image_file is None and press_but1:
    st.write("Please, load your image first!")                                           

