import os
import cv2
import time
import pickle
import numpy as np
import tensorflow as tf

from skimage import color
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern

from data import loadCityscape
from modelParameters import models, returnModelParams

def Mean_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes): 
        true_labels = K.equal(true_pixels, i) 
        pred_labels = K.equal(pred_pixels, i) 
        inter = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)
        legal_batches = K.sum(tf.cast(true_labels, tf.int32), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(ious[legal_batches]))
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = iou[legal_labels]
    return K.mean(iou)

radiusLBP = 3
n_pointsLBP = 8*radiusLBP
methodLBP = 'default'

K = 500
m = 12

def readImgToPredict(imgPath, modelName):
    x = cv2.imread(imgPath, cv2.IMREAD_COLOR)

    if modelName == 'model256x512_RGB':
        x = x / 255.0
    elif modelName == 'model256x512_HSV':
        x = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
        x[:,:,0] = x[:,:,0] / 360.0
        x[:,:,1:] = x[:,:,1:] / 100.0
    elif modelName == 'model256x512_Lab':
        x = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
        x[:,:,0] = x[:,:,0] / 100.0
        x[:,:,1:] = x[:,:,1:] + 128.0
        x[:,:,1:] = x[:,:,1:] / 255.0
    elif modelName == 'model256x512_RGB_HSV_Lab':
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = hsv[:,:,0] / 360.0
        hsv[:,:,1:] = hsv[:,:,1:] / 100.0
        lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = lab[:,:,0] / 100.0
        lab[:,:,1:] = lab[:,:,1:] + 128.0
        lab[:,:,1:] = lab[:,:,1:] / 255.0
        x = x / 255.0
        x = np.dstack((x, hsv, lab))
    elif modelName == 'model256x512_RGB_LBP':
        lbp = local_binary_pattern(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), n_pointsLBP, radiusLBP, methodLBP)
        x = x / 255.0
        lbp = lbp / (2**n_pointsLBP-1.0)
        x = np.dstack((x,lbp))
    elif modelName == 'model256x512_RGB_SP':
        sp = slic(x, n_segments=K, sigma=0, compactness=m, enforce_connectivity=False, start_label=1)
        x = color.label2rgb(sp, x, kind='avg', bg_label=-1)
        x = x / 255.0
        x = x.astype(np.float32)

    return x.astype(np.float32)

def obtainPredictionImg(model, img, nClasses):
    p = model.predict(np.expand_dims(img, axis=0))[0]
    p = np.argmax(p, axis=-1)
    p *= int(255/(nClasses-1))
    return np.uint8(p)

def writeImgAndPred(img, pred, folder, name):
    final_image = np.hstack((img, cv2.cvtColor(np.float32(pred), cv2.COLOR_GRAY2RGB)))
    cv2.imwrite(os.path.join(folder, name), final_image)

modelName = models[0]
nClasses, shape, lr, batchSize, epochs = returnModelParams(modelName)
print('Using model', modelName,'\n')

imagesTrain, masksTrain = loadCityscape('train')
imagesVal, masksVal = loadCityscape('val')
imagesTest, masksTest = loadCityscape('test')

with open('histories\\hist_'+modelName, 'rb') as f:
    x = pickle.load(f)

model = tf.keras.models.load_model('models\\'+modelName+".h5", custom_objects={"Mean_IOU": Mean_IOU})

resultPath = 'results\\'+modelName+"_"+str(round(max(x['accuracy']), 3))
if not os.path.exists(resultPath):
    os.makedirs(resultPath)
resultPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), resultPath)

t0 = time.time()
for i in range(10):
    # Train
    x = readImgToPredict(imagesTrain[i], modelName)
    p = obtainPredictionImg(model, x, nClasses)
    writeImgAndPred(cv2.imread(imagesTrain[i], cv2.IMREAD_COLOR), p, resultPath, str(i) + '_train.png')
    
    # Val
    x = readImgToPredict(imagesVal[i], modelName)
    p = obtainPredictionImg(model, x, nClasses)
    writeImgAndPred(cv2.imread(imagesTrain[i], cv2.IMREAD_COLOR), p, resultPath, str(i) + '_val.png')
    
    # Test
    x = readImgToPredict(imagesTest[i], modelName)
    p = obtainPredictionImg(model, x, nClasses)
    writeImgAndPred(cv2.imread(imagesTrain[i], cv2.IMREAD_COLOR), p, resultPath, str(i) + '_test.png')
    
print('tiempo:',time.time() - t0)  
