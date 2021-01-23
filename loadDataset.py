import os
import cv2
import numpy as np
import pickle

def loadCityscape():
    # Load Cityscape dataset -----
    
    trainingPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets\Cityscape')
    imagesPath = os.path.join(trainingPath, 'train')
    groundTruthPath = os.path.join(trainingPath, 'trainGT')
    
    images = []
    groundTruth = []
    
    print('Loading images and ground truth for Cityscape dataset...')
    for city in os.listdir(imagesPath):
        print('\tLoading', city)
        for image in os.listdir(os.path.join(imagesPath, city)):
            images.append(cv2.imread(os.path.join(imagesPath, city, image)))
        for gt in os.listdir(os.path.join(groundTruthPath, city)):
            if 'color' in gt:
                groundTruth.append(cv2.imread(os.path.join(groundTruthPath, city, gt)))
        break
    print('Loaded {} images\n'.format(len(images)))
    
    showRandomImg(images, groundTruth)
    
    saveLists(images, 'imagesCityscape', groundTruth, 'groundTruthCityScape')
    
def loadKITTI():
    # Load KITTI dataset -----
    
    trainingPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets\KITTI\\training')
    imagesPath = os.path.join(trainingPath, 'images')
    groundTruthPath = os.path.join(trainingPath, 'semantic_rgb')
    
    images = []
    groundTruth = []
    
    print('Loading images and ground truth for KITTI dataset...')
    for image in os.listdir(imagesPath):
        images.append(cv2.imread(os.path.join(imagesPath, image)))
        groundTruth.append(cv2.imread(os.path.join(groundTruthPath, image)))
    print('Loaded {} images\n'.format(len(images)))
    
    showRandomImg(images, groundTruth)
    
    saveLists(images, 'imagesKITTI', groundTruth, 'groundTruthKITTI')

def showRandomImg(images, groundTruth):
    randomImg = np.random.randint(len(images))
    
    cv2.imshow('Example', cv2.addWeighted(images[randomImg], 0.5, groundTruth[randomImg], 0.5, 0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def saveLists(images, imagesFilename, groundTruth, groundTruthFilename):
    with open(imagesFilename, 'wb') as fp:
        pickle.dump(images, fp)
        
    with open(groundTruthFilename, 'wb') as fp:
        pickle.dump(groundTruth, fp)
  
loadCityscape()
loadKITTI()

print('Finished loading')

    