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

st.title('顔認証アプリケーション')
"""表題のように複数の画像で試行錯誤された Machine Model(CNN) による顔認証ソフトウェアを開発し、91.25% の精度の予測結果が得られました。
pandas、numpy、keras、os、mobilenetv2 などのさまざまなライブラリがインポートされ、それらを利用してこのような優れたモデルが作成されました。"""


with st.sidebar:
    #option = st.sidebar.selectbox(
    #'Select One Person',img_dirs['dir'])
    
    new = st.sidebar.selectbox('対象者を選択',
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
                        
    img_file = st.selectbox("１枚の画像を選択", template_dirs['dir'])
    
    col1,col2,col3 = st.sidebar.columns([1,1,1])
    with col1:
        st.write('')
    with col2:
        starting = st.button('予測')
    with col3:
        st.write('')
    
    st.sidebar.write(' ')
    st.sidebar.title("手順")
    
    
    st.sidebar.write(
        """①「対象者を選択」に有名な方４人の名前があり、その中から予測したい人を選択。       
        """
    )
    st.sidebar.write(
        """②「１枚の画像を選択」に①に選択した方の写真２０枚があり、好きな写真を選択。
        """
    )
    st.sidebar.write(
        """③「予測」ボタンをクリックし、①に選択した人の名前をアプリが正しく予測できるかみてみましょう！！
補足情報として、当モデルは91.25%正しく予測できる。精度により性能は良いと言えるー
        
        """
    )
    st.info("Copyright@🌟🌟🌟🌟")
    
    
    
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
            """_Enjoy Life We Only Live Once_."""
        )




