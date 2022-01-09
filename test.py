import os
import cv2
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import stats
from skimage import color
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern

from data import loadCityscape
from modelParameters import models, returnModelParams, reductionMethods

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

unetSteps = 4
reductionMethod = reductionMethods[0]
imagesTrain, masksTrain = loadCityscape(reductionMethod, 'train')
imagesVal, masksVal = loadCityscape(reductionMethod, 'val')
imagesTest, masksTest = loadCityscape(reductionMethod, 'test')

times = []

for modelName in models:
    nClasses, shape, lr, batchSize, epochs = returnModelParams(modelName, reductionMethod)
    
    print('Using model' + modelName + ' - ' + reductionMethod)
    
    with open(f'histories\\{unetSteps}UnetSteps\\{reductionMethod}\\hist_{modelName}', 'rb') as f:
        x = pickle.load(f)
    
    model = tf.keras.models.load_model(f'models\\{unetSteps}UnetSteps\\{reductionMethod}\\{modelName}.h5', custom_objects={"Mean_IOU": Mean_IOU})
    
    resultPath = f"results\\{unetSteps}UnetSteps\\{reductionMethod}\\{modelName}_{round(max(x['accuracy']), 3)}"
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
        writeImgAndPred(cv2.imread(imagesVal[i], cv2.IMREAD_COLOR), p, resultPath, str(i) + '_val.png')
        
        # Test
        x = readImgToPredict(imagesTest[i], modelName)
        p = obtainPredictionImg(model, x, nClasses)
        writeImgAndPred(cv2.imread(imagesTest[i], cv2.IMREAD_COLOR), p, resultPath, str(i) + '_test.png')
    
    totalTime = time.time() - t0
    print('Total time: ' + str(totalTime) + '\n')  
    times.append(totalTime)

orderedIndex = np.argsort(times)
fig = plt.figure(figsize=(20,10))
plt.barh(np.array(models)[orderedIndex], np.array(times)[orderedIndex])
plt.title('Time to process 30 already reduced images')
plt.xlabel('Time in seconds')
plt.savefig(f'graphs\\{unetSteps}UnetSteps\\{reductionMethod}\\_execution_time.png')

# ----------------------------------------------------------------------------
H = 256
W = 512
id2cat = np.array([0,0,0,0,0,0,0, 1,1,1,1, 0,0,0,0,0,0, 0,0,0,0, 0,0, 0, 3,3, 2,2,2,2,2,2,2,2,2])

trainingPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets\\Cityscape')
imagesPath = os.path.join(trainingPath, 'train')
maskPath = os.path.join(trainingPath, 'train' + 'GT')

times = []
for reductionMethod in reductionMethods:
    t0 = time.time()
    if reductionMethod == 'bilinearInterpolation':
        print(f'Reading with {reductionMethod} method...')
        for city in os.listdir(imagesPath):
            for image in os.listdir(os.path.join(imagesPath, city))[:10]:
                cv2.resize(cv2.imread(os.path.join(imagesPath, city, image)), (W, H))
            for mask in os.listdir(os.path.join(maskPath, city))[:40]:
                if 'label' in mask:
                    id2cat[cv2.resize(cv2.imread(os.path.join(maskPath, city, mask)), (W, H), interpolation = cv2.INTER_NEAREST)]
                    
            break
    elif reductionMethod == 'meanSlidingWindow':
        rows = np.arange(0,H*4,4)
        columns = np.arange(0,W*4,4)
            
        for city in os.listdir(imagesPath):
            for image in os.listdir(os.path.join(imagesPath, city))[:10]:
                originalImg = cv2.imread(os.path.join(imagesPath, city, image))
                newImg = np.zeros([H, W, 3], dtype=np.uint8)
                
                for row in rows:
                    for column in columns:
                        newImg[row//4, column//4, :] = np.mean(originalImg[row:row+4, column:column+4, :], axis=(0,1))
                
            for mask in os.listdir(os.path.join(maskPath, city))[:40]:
                if 'label' in mask:
                    originalMask = cv2.imread(os.path.join(maskPath, city, mask), 0)
                    newMask = np.zeros([H, W], dtype=np.uint8)
                    
                    for row in rows:
                        for column in columns:
                            newMask[row//4, column//4] = stats.mode(originalMask[row:row+4, column:column+4],axis=None)[0][0]
                    id2cat[newMask]
            break
    
    totalTime = time.time() - t0
    print('Total time: ' + str(totalTime) + '\n')  
    times.append(totalTime)
    
orderedIndex = np.argsort(times)
fig = plt.figure(figsize=(20,10))
plt.barh(np.array(reductionMethods)[orderedIndex], np.array(times)[orderedIndex])
plt.title('Time to read 10 images and 10 masks')
plt.xlabel('Time in seconds')
plt.savefig(f'graphs\\{unetSteps}UnetSteps\\reading_time.png')
























