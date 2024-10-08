import streamlit as st
import numpy as np
import tensorflow as tf
from time import time
import os
import librosa
import requests
import streamlit as st
from streamlit_lottie import st_lottie

try:
 from PIL import Image
except ImportError:
 import Image

st.title("------ Music Gerne Classification 🎸")

st.write("The GTZAN genre collection dataset which is a very popular audio collection dataset. It contains approximately 1000 audio files that belong to 10 different classes. Each audio file is in .wav format (extension). The classes to which audio files belong are Blues, Hip-hop, classical, pop, Disco, Country, Metal, Jazz, Reggae, and Rock. ")

image = Image.open('rock.jpg')
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
anime1="https://assets10.lottiefiles.com/private_files/lf30_fjln45y5.json"
anime1_json=load_lottieurl(anime1)
st_lottie(anime1_json,key='music')
#st.image(image)
st.subheader('Feature Extraction')
code='''
  genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 
        'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

for genre, genre_number in genres.items():
    for filename in os.listdir(f'path to {genre}'):
        songname = f('path to {genre}\\{filename}'
        for index in range(2):
            audio, sr = librosa.load(songname,res_type='kaiser_test')
            mfcc_fea = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T,axis=0)
            fea_class=genre
            dataset.append(mfcc_fea)
            cla.append(fea_class) '''
st.code(code, language='python')
st.subheader('                   Upload your .wav file                 ')


audio_file_path = st.file_uploader("Choose a .wav file",type=['wav'])

if st.button('Predict'):
    mydict = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 
            'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
    librosa_audio_data,librosa_sample_rate=librosa.load(audio_file_path)
    mfccs = np.mean(librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40).T,axis=0)
    x=[]
    x.append(mfccs)
    x=np.array(x)
    x=np.reshape(x,(x.shape[0],10,4,1))
    model = tf.keras.models.load_model("gerne_model.h5")
    y_pre=model.predict(x)
    y_pre=np.round_(y_pre)
    a,b=np.where(y_pre==1)
    for gerne, classs in mydict.items(): 
        if classs == b[0]:
            co1,co2=st.columns(2)
            with co1:
               st.info('The audio file belongs to ------->')
           
  
            with co2:
               st.subheader(gerne)
else:
   pass
