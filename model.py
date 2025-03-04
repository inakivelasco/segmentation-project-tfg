from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate


def conv_block(inputs, filters, pool=True):
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if pool == True:
        p = MaxPool2D((2, 2))(x)
        return x, p
    else:
        return x
    
def build_unet(shape, num_classes, steps):
    inputs = Input(shape)

    if steps == 3:
        """ Encoder """
        x1, p1 = conv_block(inputs, 16, pool=True)
        x2, p2 = conv_block(p1, 32, pool=True)
        x3, p3 = conv_block(p2, 64, pool=True)
    
        """ Bridge """
        b1 = conv_block(p3, 128, pool=False)
    
        """ Decoder """    
        u1 = UpSampling2D((2, 2), interpolation="bilinear")(b1)
        c1 = Concatenate()([u1, x3])
        x4 = conv_block(c1, 64, pool=False)
    
        u2 = UpSampling2D((2, 2), interpolation="bilinear")(x4)
        c2 = Concatenate()([u2, x2])
        x5 = conv_block(c2, 32, pool=False)
    
        u3 = UpSampling2D((2, 2), interpolation="bilinear")(x5)
        c3 = Concatenate()([u3, x1])
        x6 = conv_block(c3, 16, pool=False)
    
        """ Output layer """
        output = Conv2D(num_classes, 1, padding="same", activation="softmax")(x6)
        
    if steps == 4:
        """ Encoder """
        x1, p1 = conv_block(inputs, 16, pool=True)
        x2, p2 = conv_block(p1, 32, pool=True)
        x3, p3 = conv_block(p2, 64, pool=True)
        x4, p4 = conv_block(p3, 128, pool=True)
    
        """ Bridge """
        b1 = conv_block(p4, 256, pool=False)
    
        """ Decoder """
        u1 = UpSampling2D((2, 2), interpolation="bilinear")(b1)
        c1 = Concatenate()([u1, x4])
        x5 = conv_block(c1, 128, pool=False)
    
        u2 = UpSampling2D((2, 2), interpolation="bilinear")(x5)
        c2 = Concatenate()([u2, x3])
        x6 = conv_block(c2, 64, pool=False)
    
        u3 = UpSampling2D((2, 2), interpolation="bilinear")(x6)
        c3 = Concatenate()([u3, x2])
        x7 = conv_block(c3, 32, pool=False)
    
        u4 = UpSampling2D((2, 2), interpolation="bilinear")(x7)
        c4 = Concatenate()([u4, x1])
        x8 = conv_block(c4, 16, pool=False)
    
        """ Output layer """
        output = Conv2D(num_classes, 1, padding="same", activation="softmax")(x8)
 
    return Model(inputs, output)