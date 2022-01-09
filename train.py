import os
import time
import pickle
import numpy as np
import random as rn
import tensorflow as tf

from keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from model import build_unet
from data import loadCityscape, tf_dataset
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

seed = 12345
rn.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

unetSteps = 4
modelName = models[0]
reductionMethod = reductionMethods[1]
nClasses, shape, lr, batchSize, epochs = returnModelParams(modelName, reductionMethod)

model = build_unet(shape, nClasses, unetSteps)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr), metrics=['accuracy', Mean_IOU])
model.summary()

imagesTrain, masksTrain = loadCityscape(reductionMethod, 'train')
imagesVal, masksVal = loadCityscape(reductionMethod, 'val')

trainDataset = tf_dataset(np.array(imagesTrain), np.array(masksTrain), batchSize, modelName, reductionMethod)
valDataset = tf_dataset(np.array(imagesVal), np.array(masksVal), batchSize, modelName, reductionMethod)
print('Datasets created\n')
    
trainSteps = len(imagesTrain)//batchSize
valSteps = len(imagesVal)//batchSize

modelPath = f'models\\{unetSteps}UnetSteps\\{reductionMethod}'
if not os.path.exists(modelPath):
        os.makedirs(modelPath)
        
historyPath = f'histories\\{unetSteps}UnetSteps\\{reductionMethod}'
if not os.path.exists(historyPath):
        os.makedirs(historyPath)

callbacks = [
    ModelCheckpoint(f'{modelPath}\\{modelName}.h5', verbose=1, save_best_model=True),
    ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=5, verbose=1),
]

t0 = time.time()
print('Starting training for', modelName)
print(f'\tunetSteps: {unetSteps}\n\treductionMethod: {reductionMethod}\n\tnCLasses: {nClasses}\n\tshape: {shape}\n\tlr: {lr}\n\tbatchSize: {batchSize}\n\tepochs: {epochs}\n')
history = model.fit(trainDataset,
          steps_per_epoch=trainSteps,
          validation_data=valDataset,
          validation_steps=valSteps,
          epochs=epochs,
          callbacks=callbacks)
trainingTime = time.time()-t0
print(f'Training ended: {trainingTime}s')

history.history['training_time'] = trainingTime

with open(f'{historyPath}\\hist_{modelName}', 'wb') as fd:
        pickle.dump(history.history, fd)
        

        





