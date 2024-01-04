from face_rec import img_to_encoding
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("facenet_keras.h5")

def get_database():
    database = {}
    database["danielle"] = img_to_encoding(tf.keras.utils.load_img("images/danielle.png"), model)
    database["younes"] = img_to_encoding(tf.keras.utils.load_img("images/younes.jpg"), model)
    database["tian"] = img_to_encoding(tf.keras.utils.load_img("images/tian.jpg"), model)
    database["andrew"] = img_to_encoding(tf.keras.utils.load_img("images/andrew.jpg"), model)
    database["kian"] = img_to_encoding(tf.keras.utils.load_img("images/kian.jpg"), model)
    database["sebastiano"] = img_to_encoding(tf.keras.utils.load_img("images/sebastiano.jpg"), model)
    database["bertrand"] = img_to_encoding(tf.keras.utils.load_img("images/bertrand.jpg"), model)
    database["kevin"] = img_to_encoding(tf.keras.utils.load_img("images/kevin.jpg"), model)
    database["felix"] = img_to_encoding(tf.keras.utils.load_img("images/felix.jpg"), model)
    database["benoit"] = img_to_encoding(tf.keras.utils.load_img("images/benoit.jpg"), model)
    database["arnaud"] = img_to_encoding(tf.keras.utils.load_img("images/arnaud.jpg"), model)
    database["adit"] = img_to_encoding(tf.keras.utils.load_img("images/adit.jpg"), model)

    return database