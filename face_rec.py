import tensorflow as tf
import numpy as np

def img_to_encoding(img, model):
    img = img.resize((160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

def recognize(img, database, model):
    encoding_input = img_to_encoding(img, model)
    
    min_dist = 1000
    for name, enc in database.items():
        dist = np.linalg.norm(encoding_input - enc)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist < 0.6:
        succ = True
        print(f"Welcome {identity}")
    else:
        succ = False
        print("You are unregistered")

    return succ, identity



        