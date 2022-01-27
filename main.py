import os
import cv2
import pickle
import numpy as np
from skimage.measure import regionprops
from skimage import color
from skimage.segmentation import slic, watershed
from skimage.segmentation import mark_boundaries
from skimage.feature import local_binary_pattern
import time
import matplotlib.pyplot as plt

from data import loadCityscape
from modelParameters import models, returnModelParams, reductionMethods

unetSteps = 4
modelName = models[2]
reductionMethod = reductionMethods[1]
nClasses, shape, lr, batchSize, epochs = returnModelParams(modelName, reductionMethod)

print('Using model', modelName)
print('nCLasses: {}\nshape: {}\nlr: {}\nbatchSize: {}\nepochs: {}'.format(nClasses, shape, lr, batchSize, epochs))

H, W = shape[:2]
imagesTrain, masksTrain = loadCityscape('bilinearInterpolation', 'train')


# SAVE HISTORIES PLOTS ---------------------------------------------------------------------------
with open(f'histories\\{unetSteps}UnetSteps\\{reductionMethod}\\hist_{modelName}', 'rb') as f:
    x = pickle.load(f)
    
print(x.keys())
print(x['training_time'])

fig, axs = plt.subplots(3, figsize=(10,20))

axs[0].plot(np.arange(len(x['accuracy'])),x['accuracy'],label = 'Train accuracy', color='b')
axs[0].plot(np.arange(len(x['accuracy'])),x['val_accuracy'], label = 'Val accuracy', color='r')

axs[1].plot(np.arange(len(x['accuracy'])),x['loss'], label = 'Train loss', color='b')
axs[1].plot(np.arange(len(x['accuracy'])),x['val_loss'], label = 'Val loss', color='r')

axs[2].plot(np.arange(len(x['accuracy'])),x['Mean_IOU'], label = 'Mean_IOU', color='b')
axs[2].plot(np.arange(len(x['accuracy'])),x['val_Mean_IOU'], label = 'Val Mean_IOU', color='r')

for i in range(3):
    axs[i].set_xlim([0,len(x['accuracy'])-1])
    axs[i].set_ylim([0,1])
    axs[i].set_xticks(np.arange(len(x['accuracy'])))
    axs[i].grid()
    axs[i].legend(frameon=True)
    
axs[0].set_ylim([0.6,1])

fig.suptitle(f'{unetSteps} unet steps', fontsize=24,y = 0.92)

# print(modelName)
# print(np.max(x['accuracy']))
# print(np.max(x['Mean_IOU']))
    
graphPath = f'graphs\\{unetSteps}UnetSteps\\{reductionMethod}'
if not os.path.exists(graphPath):
    os.makedirs(graphPath)
fig.savefig(f'{graphPath}\\{modelName}_acc_loss_IOU.png')




