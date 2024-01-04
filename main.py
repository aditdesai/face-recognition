'''
For face detection (cropped image with just face), Haar cascades and MTCNN can be used
For feature extraction, FaceNet can be used
For feature classification, euclidean distance, cosine similarity, SVM, KNN can be used

This code uses MTCNN, FaceNet and Euclidean Distance
'''


import cv2
from face_rec import recognize, img_to_encoding
import tensorflow as tf
from tensorflow.keras.models import load_model
import PIL
from face_cropping import extract_image
import keyboard
import pandas as pd
from database import get_database

model = load_model("facenet_keras.h5")

database = get_database()

attendance = {"Student Name" : [], "P/A" : []}

cap = cv2.VideoCapture(0)

while True:
    if keyboard.is_pressed("q"):
        break
    
    try:
        success, img = cap.read()
        if success:
            pil_image = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            extracted_image = extract_image(pil_image)
            succ, identity = recognize(extracted_image, database, model)

            if succ and identity not in attendance["Student Name"]:
                attendance["Student Name"].append(identity)
                attendance["P/A"].append('P')
    except:
        pass

pd.DataFrame(attendance).to_excel("attendance.xlsx")
    