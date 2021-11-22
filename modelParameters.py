models = ['model256x512_RGB', 'model256x512_HSV', 'model256x512_Lab', 'model256x512_RGB_HSV_Lab', 'model256x512_RGB_LBP', 'model256x512_RGB_SP']
reductionMethods = ['bilinearInterpolation', 'meanSlidingWindow']

def returnModelParams(modelName):
    epochs = 20
    lr = 1e-4 
    batchSize = 4
    nClasses = 4
    
    if modelName in ['model256x512_RGB', 'model256x512_HSV', 'model256x512_Lab', 'model256x512_RGB_SP']:
        shape = (256, 512, 3)
    elif modelName == 'model256x512_RGB_HSV_Lab':
        shape = (256, 512, 9)
    elif modelName == 'model256x512_RGB_LBP':
        shape = (256, 512, 4)
    
    return nClasses, shape, lr, batchSize, epochs
