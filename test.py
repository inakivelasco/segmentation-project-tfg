import os
import cv2
import numpy as np
import tensorflow as tf

from data import loadCityscape

H = 256
W = 256
nClasses = 8

images, masks = loadCityscape()

nTrain = int(len(images)*0.8)
trainImages = images[:nTrain]
trainMasks = masks[:nTrain]
valImages = images[nTrain:]
valMasks = masks[nTrain:]

model = tf.keras.models.load_model("model.h5")

resultPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')

for i in range(5):
    # Train
    x = cv2.imread(trainImages[i], cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    
    p = model.predict(np.expand_dims(x, axis=0))[0]
    p = np.argmax(p, axis=-1)
    p *= 36
    p = np.uint8(p)
    
    final_image = np.hstack((x*255, cv2.cvtColor(np.float32(p), cv2.COLOR_GRAY2RGB)))
    
    cv2.imwrite(os.path.join(resultPath, str(i) + 'train.png'), final_image)
    
    # Val
    x = cv2.imread(valImages[i], cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    
    p = model.predict(np.expand_dims(x, axis=0))[0]
    p = np.argmax(p, axis=-1)
    p *= 36
    p = np.uint8(p)
    
    final_image = np.hstack((x*255, cv2.cvtColor(np.float32(p), cv2.COLOR_GRAY2RGB)))
    
    cv2.imwrite(os.path.join(resultPath, str(i) + 'val.png'), final_image)
    
