import streamlit as st
import numpy as np
import tensorflow as tf
import os
import librosa
import requests
from streamlit_lottie import st_lottie
from PIL import Image

# Title and description
st.title("------ Music Genre Classification 🎸")
st.write("The GTZAN genre collection dataset which is a very popular audio collection dataset. It contains approximately 1000 audio files that belong to 10 different classes. Each audio file is in .wav format (extension). The classes to which audio files belong are Blues, Hip-hop, Classical, Pop, Disco, Country, Metal, Jazz, Reggae, and Rock.")

# Display image
'''image = Image.open('F:/DBDA/Final Project/Music Genre Classification Using Machine Learning-20240516T173721Z-001 (1)/Music Genre Classification Using Machine Learning/rock.jpg')
st.image(image)'''

# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load and display Lottie animation
anime1 = "https://assets10.lottiefiles.com/private_files/lf30_fjln45y5.json"
anime1_json = load_lottieurl(anime1)
st_lottie(anime1_json, key='music')

# Subheader and code snippet for feature extraction
st.subheader('Feature Extraction')
code = '''
genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 
          'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

for genre, genre_number in genres.items():
    for filename in os.listdir(f'path to {genre}'):
        songname = f'path to {genre}\\{filename}'
        for index in range(2):
            audio, sr = librosa.load(songname, res_type='kaiser_best')
            mfcc_fea = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
            fea_class = genre
            dataset.append(mfcc_fea)
            cla.append(fea_class) 
'''
st.code(code, language='python')

# File uploader for .wav files
uploaded_file = st.file_uploader("Choose a .wav file", type=['wav'])

# If a file is uploaded
if uploaded_file is not None:
    # Load the audio file using librosa
    audio, sr = librosa.load(uploaded_file, sr=None)
    
    # Display basic information about the audio file
    st.write(f"File name: {uploaded_file.name}")
    st.write(f"Sample rate: {sr} Hz")
    st.write(f"Audio duration: {librosa.get_duration(y=audio, sr=sr):.2f} seconds")

    # Optionally, display the waveform using matplotlib (optional)
    import matplotlib.pyplot as plt
    import librosa.display

    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

    # Perform any additional processing here (e.g., feature extraction, prediction, etc.)
else:
    st.write("Please upload a .wav file.")

# Prediction button
if st.button('Predict'):
    mydict = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 
              'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
    
    # Load and preprocess the audio file
    if uploaded_file is not None:
        audio_file_path = uploaded_file
        librosa_audio_data, librosa_sample_rate = librosa.load(audio_file_path)
        mfccs = np.mean(librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40).T, axis=0)
        x = np.array([mfccs])
        x = np.reshape(x, (x.shape[0], 10, 4, 1))
        
        # Load the pre-trained model
        model = tf.keras.models.load_model("genre_model.h5")
        
        # Predict the genre
        y_pre = model.predict(x)
        y_pre = np.round_(y_pre)
        a, b = np.where(y_pre == 1)
        
        # Display the predicted genre
        for genre, class_id in mydict.items():
            if class_id == b[0]:
                col1, col2 = st.columns(2)
                with col1:
                    st.info('The audio file belongs to ------->')
                with col2:
                    st.subheader(genre)
else:
    pass
