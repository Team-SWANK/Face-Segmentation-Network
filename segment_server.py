import os
import io
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import math
import redis

# import cv2

import numpy as np
import redis
import uuid
import time
import settings
import helpers
import json

from skimage.io import imread
# from skimage.transform import resize

# from keras.models import Model
# from keras.layers import Input, BatchNormalization, Activation, Dropout
# from keras.layers.convolutional import Conv2D, Conv2DTranspose
# from keras.layers.pooling import MaxPooling2D
# from keras.layers.merge import concatenate
# from keras.optimizers import Adam
# import keras.backend.tensorflow_backend as tb
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


# tb._SYMBOLIC_SCOPE.value = True

app = Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)
api = Api(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)
db.flushall()

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# IMG_WIDTH = 128 # for faster computing
# IMG_HEIGHT = 128 # for faster computing
# IMG_CHANNELS = 3

# def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
#     """Function to add 2 convolutional layers with the parameters passed to it"""
#     # first layer
#     x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
#               kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
#     if batchnorm:
#         x = BatchNormalization()(x)
#     x = Activation('relu')(x)
    
#     # second layer
#     x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
#               kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
#     if batchnorm:
#         x = BatchNormalization()(x)
#     x = Activation('relu')(x)
    
#     return x

# def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    # """Function to define the UNET Model"""
    # Contracting Path
#     c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
#     p1 = MaxPooling2D((2, 2))(c1)
#     p1 = Dropout(dropout)(p1)
    
#     c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
#     p2 = MaxPooling2D((2, 2))(c2)
#     p2 = Dropout(dropout)(p2)
    
#     c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
#     p3 = MaxPooling2D((2, 2))(c3)
#     p3 = Dropout(dropout)(p3)
    
#     c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
#     p4 = MaxPooling2D((2, 2))(c4)
#     p4 = Dropout(dropout)(p4)
    
#     c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
#     # Expansive Path
#     u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
#     u6 = concatenate([u6, c4])
#     u6 = Dropout(dropout)(u6)
#     c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
#     u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
#     u7 = concatenate([u7, c3])
#     u7 = Dropout(dropout)(u7)
#     c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
#     u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
#     u8 = concatenate([u8, c2])
#     u8 = Dropout(dropout)(u8)
#     c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
#     u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
#     u9 = concatenate([u9, c1])
#     u9 = Dropout(dropout)(u9)
#     c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
#     outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
#     model = Model(inputs=[input_img], outputs=[outputs])
#     return model

# def segment_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.3,
#         minNeighbors=3,
#         minSize=(30, 30)
#     )
#     dim_x, dim_y = image.shape[0], image.shape[1]

#     X = np.zeros((len(faces), 128, 128, 3), dtype=np.float32)
#     X_positions = []

#     index=0
#     for (x, y, w, h) in faces:
#         transpose_x, transpose_y = w * 0.75, h * 0.75
#         x_img = math.floor(x-transpose_x) if math.floor(x-transpose_x) >= 0 else 0
#         y_img = math.floor(y-transpose_y) if math.floor(y-transpose_y) >= 0 else 0
#         w_img = math.floor(w+2*transpose_x) if math.floor(w+2*transpose_x) < dim_x-1 else dim_x-1
#         h_img = math.floor(h+2*transpose_y) if math.floor(w+2*transpose_y) < dim_y-1 else dim_y-1
#         X_positions.append([x_img, y_img, w_img, h_img])
#     #     2d array of cropped image
#         roi_color = image[y_img:h_img+y_img, x_img:w_img+x_img]
#         X [index] = resize(roi_color, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#         index +=1

#     preds_test = (model.predict(X, verbose=1) > 0.9).astype(np.uint8)

#     upsampled_mask = np.zeros((dim_x, dim_y), dtype=np.uint8)
#     for i in range(len(preds_test)):
#         coords = X_positions[i]
#         section = resize(np.squeeze(preds_test[i]), 
#                         (coords[3], coords[2]), mode='constant', preserve_range=True)
#         upsampled_mask[coords[1]:coords[3]+coords[1], coords[0]:coords[2]+coords[0]] += section.astype(np.uint8)
#     return upsampled_mask

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")
	# resize the input image and preprocess it
	# image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	# return the processed image
	return image

class Segment(Resource):
    def get(self):
        return {'hello': 'world'}

    def post(self):
        data = {"success": False}
        files = request.files.to_dict()
        if files.get("image"):
            image =Image.open(io.BytesIO(files['image'].read()))
            image = prepare_image(image, (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
            # ensure our NumPy array is C-contiguous as well,
            # otherwise we won't be able to serialize it
            image = image.copy(order="C")
            # generate an ID for the classification then add the
            # classification ID + image to the queue
            k = str(uuid.uuid4())
            image_shape = image.shape
            image = helpers.base64_encode_image(image)
            d = {"id": k, "image": image, "shape": image_shape}
            db.rpush(settings.IMAGE_QUEUE, json.dumps(d))

            # keep looping until our model server returns the output
            # predictions
            while True:
                # attempt to grab the output predictions
                output = db.get(k)

                # check to see if our model has classified the input
                # image
                if output is not None:
                    # add the output predictions to our data
                    # dictionary so we can return it to the client
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)

                    # delete the result from the database and break
                    # from the polling loop
                    db.delete(k)
                    break

                # sleep for a small amount to give the model a chance
                # to classify the input image
                time.sleep(settings.CLIENT_SLEEP)

            # indicate that the request was a success
            data["success"] = True

        # return the data dictionary as a JSON response
        return jsonify(data)


api.add_resource(Segment, '/')

if __name__ == '__main__':
    # input_img = Input((IMG_HEIGHT, IMG_WIDTH, 3), name='img')
    # model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    # model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    # model.load_weights('face-segmentation.h5')
    app.run(host="0.0.0.0",port=5000,threaded=False)
