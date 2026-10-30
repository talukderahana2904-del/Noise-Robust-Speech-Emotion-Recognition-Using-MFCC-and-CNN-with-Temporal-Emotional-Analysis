import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('Agg')   # FIX for tkinter crash

from flask import Flask, render_template, request, send_from_directory
import numpy as np
import librosa
import tensorflow as tf
import pickle
from pydub import AudioSegment
import matplotlib.pyplot as plt

# Optional FFmpeg path
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

# Load model
model = tf.keras.models.load_model("emotion_model_final.keras")
le = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Convert audio
def convert_to_wav(input_path, output_path="uploads/temp.wav"):
    if input_path.endswith(".wav"):
        return input_path
    if input_path.endswith(".m4a"):
        try:
            audio = AudioSegment.from_file(input_path, format="m4a")
            audio.export(output_path, format="wav")
            return output_path
        except:
            return input_path
    return input_path

# Feature extraction
def extract_features(file_path):
    file_path = convert_to_wav(file_path)
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)

    return np.hstack([mfcc, chroma, mel, contrast])

# Waveform
def create_waveform(file_path):
    y, sr = librosa.load(file_path)

    plt.figure(figsize=(10,3))
    plt.plot(y)
    plt.title("Waveform")

    path = "static/waveform.png"
    plt.savefig(path)
    plt.close()

    return path

# Per-second analysis
def analyze_per_second(file_path):
    y, sr = librosa.load(file_path)
    duration = len(y) // sr

    emotions = []
    times = []

    for sec in range(duration):
        segment = y[sec*sr:(sec+1)*sr]
        if len(segment) < sr:
            continue

        mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=segment, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=segment, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr).T, axis=0)

        feature = np.hstack([mfcc, chroma, mel, contrast])
        feature = np.expand_dims(feature, axis=0)
        feature = np.expand_dims(feature, axis=2)

        pred = model.predict(feature, verbose=0)
        emo = le.inverse_transform([np.argmax(pred)])[0]

        emotions.append(emo)
        times.append(sec)

    return times, emotions

# Emotion graph
def create_emotion_graph(times, emotions):
    unique = list(set(emotions))
    mapping = {e:i for i,e in enumerate(unique)}
    y = [mapping[e] for e in emotions]

    plt.figure(figsize=(10,4))
    plt.plot(times, y, marker='o')

    plt.yticks(list(mapping.values()), list(mapping.keys()))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Emotion")
    plt.title("Emotion Timeline")

    path = "static/emotion_graph.png"
    plt.savefig(path)
    plt.close()

    return path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    features = extract_features(filepath)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)

    prediction = model.predict(features)
    emotion = le.inverse_transform([np.argmax(prediction)])[0]
    confidence = round(np.max(prediction)*100, 2)

    waveform = create_waveform(filepath)
    times, emotions = analyze_per_second(filepath)
    graph = create_emotion_graph(times, emotions)

    return render_template(
        "result.html",
        emotion=emotion,
        confidence=confidence,
        audio=file.filename,
        waveform=waveform,
        graph=graph,
        duration=len(times)
    )

if __name__ == "__main__":
    app.run(debug=True)