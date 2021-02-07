import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from data import loadCityscape, tf_dataset
from model import build_unet

images, masks = loadCityscape()

nTrain = int(len(images)*0.8)
shape = (256, 256, 3)
nClasses = 8
lr = 1e-4
batchSize = 8
epochs = 10 

print('nTrain:', nTrain)
print('nVal:', len(images)-nTrain)

trainImages = images[:nTrain]
trainMasks = masks[:nTrain]
valImages = images[nTrain:]
valMasks = masks[nTrain:]

model = build_unet(shape, nClasses)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr))
model.summary()

trainDataset = tf_dataset(np.array(trainImages), np.array(trainMasks), batch=batchSize)
valDataset = tf_dataset(np.array(valImages), np.array(valMasks), batch=batchSize)

print('Datasets created')

trainSteps = len(trainImages)//batchSize
valSteps = len(valImages)//batchSize

callbacks = [
    ModelCheckpoint("model.h5", verbose=1, save_best_model=True),
    ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=5, verbose=1)    
]

model.fit(trainDataset,
          steps_per_epoch=trainSteps,
          validation_data=valDataset,
          validation_steps=valSteps,
          epochs=epochs,
          callbacks=callbacks)