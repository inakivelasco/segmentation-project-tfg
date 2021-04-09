models = ['model256x512_RGB', 'model256x512_HSV', 'model256x512_Lab', 'model256x512_RGB_HSV_Lab', 'model256x512_RGB_LBP']

def returnModelParams(modelName):
    epochs = 20
    lr = 1e-4 
    batchSize = 4
    nClasses = 4
    
    if modelName == 'model256x512_RGB':
        shape = (256, 512, 3)
    elif modelName == 'model256x512_HSV':
        shape = (256, 512, 3)
    elif modelName == 'model256x512_Lab':
        shape = (256, 512, 3)
    elif modelName == 'model256x512_RGB_HSV_Lab':
        shape = (256, 512, 9)
    elif modelName == 'model256x512_RGB_LBP':
        shape = (256, 512, 4)

    return nClasses, shape, lr, batchSize, epochs
