import cv2
import os

model_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(model_path)

def get_faces(gray_img):
    return face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48))
