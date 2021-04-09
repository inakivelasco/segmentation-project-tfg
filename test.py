import os
import cv2
import time
import numpy as np
import tensorflow as tf
from skimage.feature import local_binary_pattern

from data import loadCityscape
# from train import Mean_IOU, gen_dice
from modelParameters import models, returnModelParams

radiusLBP = 3
n_pointsLBP = 8*radiusLBP
methodLBP = 'default'

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

    return x.astype(np.float32)

def obtainPredictionImg(model, img, nClasses):
    p = model.predict(np.expand_dims(img, axis=0))[0]
    p = np.argmax(p, axis=-1)
    p *= int(255/(nClasses-1))
    return np.uint8(p)

def writeImgAndPred(img, pred, folder, name):
    final_image = np.hstack((img[:,:,:3]*255, cv2.cvtColor(np.float32(pred), cv2.COLOR_GRAY2RGB)))
    cv2.imwrite(os.path.join(folder, name), final_image)

modelName = models[0]
nClasses, shape, lr, batchSize, epochs = returnModelParams(modelName)
print('Using model', modelName,'\n')

imagesTrain, masksTrain = loadCityscape('train')
imagesVal, masksVal = loadCityscape('val')
imagesTest, masksTest = loadCityscape('test')

model = tf.keras.models.load_model('models\\'+modelName+".h5", custom_objects={"Mean_IOU": Mean_IOU})

if not os.path.exists('results\\'+modelName):
    os.makedirs('results\\'+modelName)
resultPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results\\'+modelName)

t0 = time.time()
for i in range(10):
    # Train
    x = readImgToPredict(imagesTrain[i], modelName)
    p = obtainPredictionImg(model, x, nClasses)
    writeImgAndPred(x, p, resultPath, str(i) + '_train.png')
    
    # Val
    x = readImgToPredict(imagesVal[i], modelName)
    p = obtainPredictionImg(model, x, nClasses)
    writeImgAndPred(x, p, resultPath, str(i) + '_val.png')
    
    # Test
    x = readImgToPredict(imagesTest[i], modelName)
    p = obtainPredictionImg(model, x, nClasses)
    writeImgAndPred(x, p, resultPath, str(i) + '_test.png')
    
print('tiempo:',time.time() - t0)  
