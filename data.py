import os
import cv2
import numpy as np
import tensorflow as tf

H = 256
W = 256
id2cat = np.array([0,0,0,0,0,0,0, 1,1,1,1, 2,2,2,2,2,2, 3,3,3,3, 4,4, 5, 6,6, 7,7,7,7,7,7,7,7,7])
categories = np.array(['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle'])

def readImage(x):
    x = x.decode("utf-8")
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x
    
def readMask(path):
    path = path.decode("utf-8")
    mask = cv2.imread(path, 0)
    mask = cv2.resize(mask, (W, H))
    return mask.astype(np.int32)

def preprocess(x, y):
    def f(x, y):
        image = readImage(x)
        mask = readMask(y)
        return image, mask
    
    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, 8, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 8])
    
    return image, mask
        

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset

def loadCityscape():
    trainPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets\\CityscapeCorregido\\train')
    imagesPath = os.path.join(trainPath, 'images')
    maskPath = os.path.join(trainPath, 'masks')
    
    images = []
    masks = []
     
    print('Loading images and masks for Cityscape dataset...')
    for image in os.listdir(imagesPath):
        images.append(os.path.join(imagesPath, image))
    for mask in os.listdir(maskPath):
        if 'label' in mask:
            masks.append(os.path.join(maskPath, mask))
    print('Loaded {} images\n'.format(len(images)))
    
    return images, masks



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 


