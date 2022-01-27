import os
import cv2
import numpy as np
from scipy import stats

def windowMean(window):
    return np.mean(np.mean(window,axis=0), axis=0)

reductionMethod ='meanSlidingWindow'
dataset = 'CityscapeCorrected256x512_4classes'
trainValTest = 'test'

H = 256
W = 512
if dataset == 'CityscapeCorrected256x512':
    id2cat = np.array([0,0,0,0,0,0,0, 1,1,1,1, 2,2,2,2,2,2, 3,3,3,3, 4,4, 5, 6,6, 7,7,7,7,7,7,7,7,7])
    # 8 clases
elif dataset == 'CityscapeCorrected256x512_3classes':
    id2cat = np.array([0,0,0,0,0,0,0, 1,1,1,1, 0,0,0,0,0,0, 0,0,0,0, 0,0, 0, 0,0, 2,2,2,2,2,2,2,2,2])
    # carretera vehiculo
elif dataset == 'CityscapeCorrected256x512_4classes':
    id2cat = np.array([0,0,0,0,0,0,0, 1,1,1,1, 0,0,0,0,0,0, 0,0,0,0, 0,0, 0, 3,3, 2,2,2,2,2,2,2,2,2])
    # carretera vehiculo persona
elif dataset == 'CityscapeCorrected256x512_5classes':
    id2cat = np.array([0,0,0,0,0,0,0, 1,1,1,1, 0,0,0,0,0,0, 4,4,4,4, 0,0, 0, 3,3, 2,2,2,2,2,2,2,2,2])
    #carretera vehiculo persona objetos
    
categories = np.array(['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle'])
    
trainingPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets\\Cityscape')
imagesPath = os.path.join(trainingPath, trainValTest)
maskPath = os.path.join(trainingPath, trainValTest + 'GT')

# originalPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets\\Cityscape')
# newPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets\\CityscapeCorrected1024x2048')

# imagesPath = os.path.join(originalPath, trainValTest)
# newImagesPath = os.path.join(newPath, trainValTest, 'images')

# masksPath = os.path.join(originalPath, f'{trainValTest}GT')
# newMasksPath = os.path.join(newPath, trainValTest, 'masks')

# if not os.path.exists(newImagesPath):
#     os.makedirs(newImagesPath)
# if not os.path.exists(newMasksPath):
#     os.makedirs(newMasksPath)

# for city in os.listdir(imagesPath):
#     print('\tLoading', city)
#     for image in os.listdir(os.path.join(imagesPath, city)):
#         cv2.imwrite(os.path.join(newImagesPath, image), cv2.imread(os.path.join(imagesPath, city, image)))
#     for mask in os.listdir(os.path.join(masksPath, city)):
#         if 'label' in mask:
#             cv2.imwrite(os.path.join(newMasksPath, mask), id2cat[cv2.imread(os.path.join(masksPath, city, mask), 0)])



newPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets\\' + dataset + '_' + reductionMethod)
if not os.path.exists(newPath):
    os.makedirs(newPath)

newPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets\\' + dataset + '_' + reductionMethod + '\\' + trainValTest)
if not os.path.exists(newPath):
    os.makedirs(newPath)
if not os.path.exists(newPath + '\\images'):
    os.makedirs(newPath + '\\images')
if not os.path.exists(newPath + '\\masks'):
    os.makedirs(newPath + '\\masks')

print('Creating dataset: ' + dataset + '_' + reductionMethod + ': ' + trainValTest)

if reductionMethod == 'bilinearInterpolation':
    for city in os.listdir(imagesPath):
        print('\tLoading', city)
        for image in os.listdir(os.path.join(imagesPath, city)):
            cv2.imwrite(os.path.join(newPath, 'images', image), cv2.resize(cv2.imread(os.path.join(imagesPath, city, image)), (W, H)))
        for mask in os.listdir(os.path.join(maskPath, city)):
            if 'label' in mask:
                cv2.imwrite(os.path.join(newPath, 'masks', mask), id2cat[cv2.resize(cv2.imread(os.path.join(maskPath, city, mask)), (W, H), interpolation = cv2.INTER_NEAREST)])

elif reductionMethod == 'meanSlidingWindow':
    rows = np.arange(0,H*4,4)
    columns = np.arange(0,W*4,4)
        
    for city in os.listdir(imagesPath):
        print('\tLoading', city)
        for image in os.listdir(os.path.join(imagesPath, city)):
            originalImg = cv2.imread(os.path.join(imagesPath, city, image))
            newImg = np.zeros([H, W, 3], dtype=np.uint8)
            
            for row in rows:
                for column in columns:
                    newImg[row//4, column//4, :] = np.mean(originalImg[row:row+4, column:column+4, :], axis=(0,1))
            cv2.imwrite(os.path.join(newPath, 'images', image), newImg)
            
        for mask in os.listdir(os.path.join(maskPath, city)):
            if 'label' in mask:
                originalMask = cv2.imread(os.path.join(maskPath, city, mask), 0)
                newMask = np.zeros([H, W], dtype=np.uint8)
                
                for row in rows:
                    for column in columns:
                        newMask[row//4, column//4] = stats.mode(originalMask[row:row+4, column:column+4],axis=None)[0][0]
                cv2.imwrite(os.path.join(newPath, 'masks', mask), id2cat[newMask])
            
