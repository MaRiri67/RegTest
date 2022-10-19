import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

image_library = {'Obama':'obama','Daw Su':'daw_su','Jackie':'jackie_chan','Messi':'messi'}

img_dirs = pd.DataFrame()
#img_dirs['dir'] = ['obama','daw_su','jackie_chan','messi']
img_dirs['dir'] = list(image_library.values())


template_dirs = pd.DataFrame()
template_dirs['dir'] = None

st.title('Face Recognition Application')
"""As entitled, we developed a facial recognition software by Machine Model(CNN) which was trianed by multiple images gives a predicted result of 91.25% Accuracy.
Various libraries are imported included pandas,numpy, keras,os,mobilenetv2, and etc.. and by utilizing them created such a good model.
The whole Porject's source code can be found [here](https://github.com/Rajkap/Streamlit_app)."""


with st.sidebar:
    #option = st.sidebar.selectbox(
    #'Select One Person',img_dirs['dir'])
    
    new = st.sidebar.selectbox('Select One Person',
    list(image_library.keys()))
    
    option = image_library[new]
    
    def load_img(name):
        for f in os.scandir("templates"):
            if (f.is_dir() and f.name == name):
                list_dir = []
                for img in os.scandir(f):
                    list_dir.append(img.name)
        template_dirs['dir'] = list_dir
    
    load_img(option)
                        
    img_file = st.selectbox("Choose any one image", template_dirs['dir'])
    
    col1,col2,col3 = st.sidebar.columns([1,1,1])
    with col1:
        st.write('')
    with col2:
        starting = st.button('predict')
    with col3:
        st.write('')
    
    st.sidebar.write(' ')
    st.sidebar.title("Note")
    
    
    st.sidebar.write(
        """Playing with the options in Selectbox, you will find _images of Four Famous-Person_ exist in this
        model.The second Selectbox includes twenty test images of those people. You can test each person's images by choosing the avaliable options
        in those two selectboxes. After that, the Model predicted answer will apper. Keep in mind that this model's prediction accuracy is **_91.25%_** _which is not bad._
        
        """
    )
    st.info("Copyright@Clover")
    
    
    
from zipfile import ZipFile
work_dir = os.getcwd()       #Saves the current working directory.
# print(work_dir)
# st.write(work_dir)
with ZipFile(os.path.join(work_dir ,'model_file.zip'),'r') as zipobject:
  zipobject.extractall() 
path = os.scandir(work_dir)
# st.write(path)
model_path = None
for f in path:
     if f.name == 'model_face_recog_eg1 - Copy.h5':
         model_path = f


model_file = tf.keras.models.load_model(model_path)

template ='templates'
real_path = os.path.join(work_dir ,template,option,img_file)
img = tf.keras.preprocessing.image.load_img(real_path, target_size=(160,160))

col1, col2, col3 = st.columns([1,2,1])
with col1:
    st.write("")

with col2:
    st.image(tf.keras.preprocessing.image.load_img(real_path),width=250)


    # st.image(tf.keras.preprocessing.image.load_img(real_path),width=250)
dictionary = {0:'Daw Aung San SuuKyi',1:'Jackie Chan',2:'Messi',3:'Barack Obama'}
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x /= 255.0
images = np.vstack([x])
classes = model_file.predict(x)
y_classes=classes.argmax(axis=-1)
label = y_classes[0]#9


with col3:
    st.write("")
    st.write("")
    st.write("")
    
if starting == True:
    col4,col5,col6 = st.columns([1,2,1])
    with col4:
        st.write('')
    with col5:
        st.write('')
        st.write("Predicted as","**_",dictionary[label],"_**")
    with col6:
        st.write('')
        
    st.write("")
    st.write("")
st.write(
            """_This project is created as a AI-course pritical project by team `Clover`_."""
        )




