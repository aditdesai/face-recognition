import cv2
from face_rec import recognize, img_to_encoding
import tensorflow as tf
from tensorflow.keras.models import load_model
import PIL

model = load_model("facenet_keras.h5")

database = {}
database["danielle"] = img_to_encoding(tf.keras.utils.load_img("images/danielle.png"), model)
database["younes"] = img_to_encoding(tf.keras.utils.load_img("images/younes.jpg"), model)
database["tian"] = img_to_encoding(tf.keras.utils.load_img("images/tian.jpg"), model)
database["andrew"] = img_to_encoding(tf.keras.utils.load_img("images/andrew.jpg"), model)
database["kian"] = img_to_encoding(tf.keras.utils.load_img("images/kian.jpg"), model)
database["dan"] = img_to_encoding(tf.keras.utils.load_img("images/dan.jpg"), model)
database["sebastiano"] = img_to_encoding(tf.keras.utils.load_img("images/sebastiano.jpg"), model)
database["bertrand"] = img_to_encoding(tf.keras.utils.load_img("images/bertrand.jpg"), model)
database["kevin"] = img_to_encoding(tf.keras.utils.load_img("images/kevin.jpg"), model)
database["felix"] = img_to_encoding(tf.keras.utils.load_img("images/felix.jpg"), model)
database["benoit"] = img_to_encoding(tf.keras.utils.load_img("images/benoit.jpg"), model)
database["arnaud"] = img_to_encoding(tf.keras.utils.load_img("images/arnaud.jpg"), model)
database["adit"] = img_to_encoding(tf.keras.utils.load_img("images/adit.jpg"), model)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if success:
        pil_image = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        succ, identity = recognize(pil_image, database, model)