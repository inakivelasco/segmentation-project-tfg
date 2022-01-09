import os
import cv2
import numpy as np
import tensorflow as tf

from skimage import color
from skimage.segmentation import slic
from skimage.feature import local_binary_pattern

radiusLBP = 3
n_pointsLBP = 8*radiusLBP
methodLBP = 'default'

K = 500  # n_segments
m = 12   # compactness
    
def readMask(path):
    path = path.decode("utf-8")
    mask = cv2.imread(path, 0)
    return mask.astype(np.int32)

def readImage256x512_RGB(x):
    x = x.decode("utf-8")
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = x / 255.0
    x = x.astype(np.float32)
    return x
def preprocess256x512_RGB(x, y):
    def f(x, y):
        image = readImage256x512_RGB(x)
        mask = readMask(y)
        return image, mask
    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])    
    mask = tf.one_hot(mask, 4, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 4])
    return image, mask

def readImage256x512_HSV(x):
    x = x.decode("utf-8")
    x = cv2.cvtColor(cv2.imread(x, cv2.IMREAD_COLOR), cv2.COLOR_RGB2HSV)
    x[:,:,0] = x[:,:,0] / 360.0
    x[:,:,1:] = x[:,:,1:] / 100.0
    x = x.astype(np.float32)
    return x
def preprocess256x512_HSV(x, y):
    def f(x, y):
        image = readImage256x512_HSV(x)
        mask = readMask(y)
        return image, mask
    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])    
    mask = tf.one_hot(mask, 4, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 4])
    return image, mask

def readImage256x512_Lab(x):
    x = x.decode("utf-8")
    x = cv2.cvtColor(cv2.imread(x, cv2.IMREAD_COLOR), cv2.COLOR_BGR2LAB)
    x[:,:,0] = x[:,:,0] / 100.0
    x[:,:,1:] = x[:,:,1:] + 128.0
    x[:,:,1:] = x[:,:,1:] / 255.0
    x = x.astype(np.float32)
    return x
def preprocess256x512_Lab(x, y):
    def f(x, y):
        image = readImage256x512_Lab(x)
        mask = readMask(y)
        return image, mask
    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])    
    mask = tf.one_hot(mask, 4, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 4])
    return image, mask

def readImage256x512_RGB_HSV_Lab(x):
    x = x.decode("utf-8")
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    hsv[:,:,0] = hsv[:,:,0] / 360.0
    hsv[:,:,1:] = hsv[:,:,1:] / 100.0
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = lab[:,:,0] / 100.0
    lab[:,:,1:] = lab[:,:,1:] + 128.0
    lab[:,:,1:] = lab[:,:,1:] / 255.0
    x = x / 255.0
    x = np.dstack((x, hsv, lab))
    return x.astype(np.float32)
def preprocess256x512_RGB_HSV_Lab(x, y):
    def f(x, y):
        image = readImage256x512_RGB_HSV_Lab(x)
        mask = readMask(y)
        return image, mask
    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, 4, dtype=tf.int32)
    image.set_shape([H, W, 9])
    mask.set_shape([H, W, 4])
    return image, mask

def readImage256x512_RGB_LBP(x):
    x = x.decode("utf-8")
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    lbp = local_binary_pattern(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), n_pointsLBP, radiusLBP, methodLBP)
    x = x / 255.0
    lbp = lbp / (2**n_pointsLBP-1.0)
    x = np.dstack((x,lbp))
    x = x.astype(np.float32)
    return x
def preprocess256x512_RGB_LBP(x, y):
    def f(x, y):
        image = readImage256x512_RGB_LBP(x)
        mask = readMask(y)
        return image, mask
    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])    
    mask = tf.one_hot(mask, 4, dtype=tf.int32)
    image.set_shape([H, W, 4])
    mask.set_shape([H, W, 4])
    return image, mask

def readImage256x512_RGB_SP(x):
    x = x.decode("utf-8")
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    sp = slic(x, n_segments=K, sigma=0, compactness = m, enforce_connectivity=False, start_label=1)
    x = color.label2rgb(sp, x, kind='avg', bg_label=-1)
    x = x / 255.0
    x = x.astype(np.float32)
    return x
def preprocess256x512_RGB_SP(x, y):
    def f(x, y):
        image = readImage256x512_RGB_SP(x)
        mask = readMask(y)
        return image, mask
    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])    
    mask = tf.one_hot(mask, 4, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 4])
    return image, mask

def tf_dataset(x, y, batch, modelName, reductionMethod): 
    global H
    global W
    if reductionMethod == 'None':
        H = 1024
        W = 2048
    else:
        H = 256
        W = 512
    
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    
    if modelName == 'model256x512_RGB':
        dataset = dataset.map(preprocess256x512_RGB)
    elif modelName == 'model256x512_HSV':
        dataset = dataset.map(preprocess256x512_HSV)
    elif modelName == 'model256x512_Lab':
        dataset = dataset.map(preprocess256x512_Lab)
    elif modelName == 'model256x512_RGB_HSV_Lab':
        dataset = dataset.map(preprocess256x512_RGB_HSV_Lab)
    elif modelName == 'model256x512_RGB_LBP':
        dataset = dataset.map(preprocess256x512_RGB_LBP)
    elif modelName == 'model256x512_RGB_SP':
        dataset = dataset.map(preprocess256x512_RGB_SP)
        
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset

def loadCityscape(reductionMethod, trainValTest):
    if reductionMethod in ['bilinearInterpolation', 'meanSlidingWindow']:
        trainPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets\\CityscapeCorrected256x512_4classes_' + reductionMethod + '\\' + trainValTest)
    else:
        trainPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets\\CityscapeCorrected1024x2048\\' + trainValTest)
    imagesPath = os.path.join(trainPath, 'images')
    maskPath = os.path.join(trainPath, 'masks')
    
    images = []
    masks = []
     
    print('Loading images and masks for ' + reductionMethod + '-' + trainValTest + ' Cityscape dataset...')
    for image in os.listdir(imagesPath):
        images.append(os.path.join(imagesPath, image))
    for mask in os.listdir(maskPath):
        if 'label' in mask:
            masks.append(os.path.join(maskPath, mask))
    print('Loaded {} images\n'.format(len(images)))
    
    return images, masks



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 


