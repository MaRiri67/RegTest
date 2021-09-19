import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os


img_dirs = pd.DataFrame()
img_dirs['dir'] = ['obama','daw_su','jackie_chan','messi']

template_dirs = pd.DataFrame()
template_dirs['dir'] = None

st.title('Face Recognition Application')


with st.sidebar:
    option = st.sidebar.selectbox(
    'Select One Person',img_dirs['dir'])
    
    def load_img(name):
        for f in os.scandir("templates"):
            if (f.is_dir() and f.name == name):
                list_dir = []
                for img in os.scandir(f):
                    list_dir.append(img.name)
        template_dirs['dir'] = list_dir
    
    load_img(option)
                        
    img_file = st.selectbox("Choose any one image", template_dirs['dir'])
    st.info("Copyright@Anonymous")



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
st.image(tf.keras.preprocessing.image.load_img(real_path),width=250)
dictionary = {0:'Daw Aung San SuuKyi',1:'Jackie Chan',2:'Messi',3:'Barack Obama'}
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x /= 255.0
images = np.vstack([x])
classes = model_file.predict(x)
y_classes=classes.argmax(axis=-1)
label = y_classes[0]#9
st.write("Model မှခန့်မှန်း လိုက်သော ပုဂ္ဂိုလ် မှာ ",dictionary[label], "ဖြစ်ပါသည်။")
