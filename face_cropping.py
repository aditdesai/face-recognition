from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray
from mtcnn.mtcnn import MTCNN

def extract_image(img1):
    # img1 = Image.open(image)            #open the image
    img1 = img1.convert('RGB')          #convert the image to RGB format 
    pixels = asarray(img1)              #convert the image to numpy array
    detector = MTCNN()                  #assign the MTCNN detector
    f = detector.detect_faces(pixels)
    #fetching the (x,y)co-ordinate and (width-->w, height-->h) of the image
    x1,y1,w,h = f[0]['box']             
    x1, y1 = abs(x1), abs(y1)
    x2 = abs(x1+w)
    y2 = abs(y1+h)
    #locate the co-ordinates of face in the image
    store_face = pixels[y1:y2,x1:x2]
    plt.imshow(store_face)
    image1 = Image.fromarray(store_face,'RGB')    #convert the numpy array to object
    image1 = image1.resize((160,160))             #resize the image

    # image1.save(image)
    return image1

def extract_images_from_folder():
    images = ['danielle.png', 'younes.jpg', 'tian.jpg', 'andrew.jpg', 'kian.jpg', 'dan.jpg', 'sebastiano.jpg', 'bertrand.jpg', 'kevin.jpg', 'felix.jpg', 'benoit.jpg', 'arnaud.jpg', 'adit.jpg']

    for image in images:
        extract_image("images/" + image)