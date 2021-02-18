import os
import sys
import math

import cv2

import numpy as np

import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize

from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMG_WIDTH = 128 # for faster computing
IMG_HEIGHT = 128 # for faster computing
IMG_CHANNELS = 3

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def main():
    input_img = Input((IMG_HEIGHT, IMG_WIDTH, 3), name='img')
    model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    model.load_weights('face-segmentation.h5')

    img_path = sys.argv[1]

    image = imread(img_path)[:,:,:IMG_CHANNELS]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("Found {0} Faces!".format(len(faces)))

    dim_x, dim_y = image.shape[0], image.shape[1]
    transpose_x, transpose_y = dim_x *0.05, dim_y * 0.05

    X = np.zeros((len(faces), 128, 128, 3), dtype=np.float32)
    X_positions = []

    index=0
    for (x, y, w, h) in faces:
        x_img = math.floor(x-transpose_x)
        y_img = math.floor(y-transpose_y)
        w_img = math.floor(w+2*transpose_x)
        h_img = math.floor(h+2*transpose_y)
        X_positions.append([x_img, y_img, w_img, h_img])
    #     2d array of cropped image
        roi_color = image[y_img:h_img+y_img, x_img:w_img+x_img]
        X [index] = resize(roi_color, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        index +=1

    preds_test = (model.predict(X, verbose=1)> 0.8).astype(np.uint8)

    upsampled_mask = np.zeros((dim_x, dim_y), dtype=np.uint8)
    for i in range(len(preds_test)):
        coords = X_positions[i]
        section = resize(np.squeeze(preds_test[i]), 
                        (coords[3], coords[2]), mode='constant', preserve_range=True)
        upsampled_mask[coords[1]:coords[3]+coords[1], coords[0]:coords[2]+coords[0]] += section.astype(np.uint8)
    
    plt.imsave(os.path.split(img_path)[0] + '/' + os.path.splitext(os.path.basename(img_path))[0] + '_mask.jpg', upsampled_mask, cmap='Greys_r')

if __name__ == "__main__":
    main()