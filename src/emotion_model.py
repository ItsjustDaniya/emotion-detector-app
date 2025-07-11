import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = load_model(os.path.join("model", "emotion_model.h5"))

def preprocess_face(face_img):
    face = cv2.resize(face_img, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face

def predict_emotion(face_img):
    processed = preprocess_face(face_img)
    preds = model.predict(processed, verbose=0)[0]
    return emotion_labels[np.argmax(preds)]

def get_prediction_probs(face_img):
    processed = preprocess_face(face_img)
    preds = model.predict(processed, verbose=0)[0]
    label = emotion_labels[np.argmax(preds)]
    return label, preds.round(3)
