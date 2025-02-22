models = ['model256x512_RGB', 'model256x512_HSV', 'model256x512_Lab', 'model256x512_RGB_HSV_Lab', 'model256x512_RGB_LBP', 'model256x512_RGB_SP']
reductionMethods = ['None', 'bilinearInterpolation', 'meanSlidingWindow']

def returnModelParams(modelName, reductionMethod):
    epochs = 20
    lr = 1e-4 
    batchSize = 8
    nClasses = 4
    
    if reductionMethod == 'None':
        height = 1024
        width = 2048
    else:
        height = 256
        width = 512
    
    if modelName in ['model256x512_RGB', 'model256x512_HSV', 'model256x512_Lab', 'model256x512_RGB_SP']:
        shape = (height, width, 3)
    elif modelName == 'model256x512_RGB_HSV_Lab':
        shape = (height, width, 9)
    elif modelName == 'model256x512_RGB_LBP':
        shape = (height, width, 4)
        
    return nClasses, shape, lr, batchSize, epochs