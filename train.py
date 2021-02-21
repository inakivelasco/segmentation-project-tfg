import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from data import loadCityscape, tf_dataset
from model import build_unet

imagesTrain, masksTrain = loadCityscape('train')
imagesVal, masksVal = loadCityscape('val')

shape = (256, 256, 3)
nClasses = 8
lr = 1e-4
batchSize = 8
epochs = 20

print('nTrain:', len(imagesTrain))
print('nVal:', len(imagesVal))

model = build_unet(shape, nClasses)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr))
model.summary()

trainDataset = tf_dataset(np.array(imagesTrain), np.array(masksTrain), batch=batchSize)
valDataset = tf_dataset(np.array(imagesVal), np.array(masksVal), batch=batchSize)

print('Datasets created')

trainSteps = len(imagesTrain)//batchSize
valSteps = len(imagesVal)//batchSize

callbacks = [
    ModelCheckpoint("model2.h5", verbose=1, save_best_model=True),
    ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=5, verbose=1)    
]

model.fit(trainDataset,
          steps_per_epoch=trainSteps,
          validation_data=valDataset,
          validation_steps=valSteps,
          epochs=epochs,
          callbacks=callbacks)