from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation

# start/endGrid.shape = (n, dimensions, dimensions, 1)
def IvysaurusIChooseYou(dimensions, nclasses, nTrackVars, nShowerVars):
    
    ################################
    # U View
    ################################
    startInputsU = Input(shape=(dimensions, dimensions, 1))
    endInputsU = Input(shape=(dimensions, dimensions, 1))
    branchU = CreateViewBranch(dimensions, startInputsU, endInputsU)
    
    ################################
    # V View
    ################################
    startInputsV = Input(shape=(dimensions, dimensions, 1))
    endInputsV = Input(shape=(dimensions, dimensions, 1))
    branchV = CreateViewBranch(dimensions, startInputsV, endInputsV)    
    
    ################################
    # W View
    ################################
    startInputsW = Input(shape=(dimensions, dimensions, 1))
    endInputsW = Input(shape=(dimensions, dimensions, 1))
    branchW = CreateViewBranch(dimensions, startInputsW, endInputsW)
    
    ################################
    # Now combine the U, V and W branches
    ################################
    combined = Concatenate()([branchU, branchV, branchW])

    ################################
    # Now add in the trackVars
    ################################
    trackVarsInputs = Input(shape=(nTrackVars,))
    combined = Concatenate()([combined, trackVarsInputs])

    ################################
    # Now add in the trackVars
    ################################
    showerVarsInputs = Input(shape=(nShowerVars,))
    combined = Concatenate()([combined, showerVarsInputs])
    
    ################################
    # FC layers
    ################################
    combined = Dense(4096, use_bias=False)(combined)
    combined = BatchNormalization()(combined)
    combined = Activation('relu')(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(4096, use_bias=False)(combined)
    combined = BatchNormalization()(combined)
    combined = Activation('relu')(combined)
    combined = Dropout(0.5)(combined)

    ################################
    # Now classify the image
    ################################
    outputs = Dense(nclasses, activation="softmax")(combined)
    model = Model(inputs=[startInputsU, endInputsU, startInputsV, endInputsV, startInputsW, endInputsW, trackVarsInputs, showerVarsInputs], outputs=outputs)
    
    return model

##########################################################################################################################
##########################################################################################################################

def CreateViewBranch(dimensions, startInputs, endInputs):
    ################################
    # Start branch
    ################################
    # 1 - 1 conv
    startBranch = Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(startInputs)
    startBranch = BatchNormalization()(startBranch)
    startBranch = Activation('relu')(startBranch)
    startBranch = MaxPooling2D(pool_size=2, strides=2)(startBranch)
    # 2 - 1 conv
    startBranch = Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(startBranch)
    startBranch = BatchNormalization()(startBranch)
    startBranch = Activation('relu')(startBranch)
    startBranch = MaxPooling2D(pool_size=2, strides=2)(startBranch)
    # 3 - 2 conv
    startBranch = Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(startBranch)
    startBranch = BatchNormalization()(startBranch)
    startBranch = Activation('relu')(startBranch)
    startBranch = Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(startBranch)
    startBranch = BatchNormalization()(startBranch)
    startBranch = Activation('relu')(startBranch)
    startBranch = MaxPooling2D(pool_size=2, strides=2)(startBranch)
    # 4 - 2 conv
    startBranch = Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(startBranch)
    startBranch = BatchNormalization()(startBranch)
    startBranch = Activation('relu')(startBranch)
    startBranch = Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(startBranch)
    startBranch = BatchNormalization()(startBranch)
    startBranch = Activation('relu')(startBranch)
    startBranch = MaxPooling2D(pool_size=2, strides=2)(startBranch)
    # 5 - Flatten
    startBranch = Flatten()(startBranch)

    ################################
    # End branch
    ################################
    # 1 - 1 conv
    endBranch = Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(endInputs)
    endBranch = BatchNormalization()(endBranch)
    endBranch = Activation('relu')(endBranch)
    endBranch = MaxPooling2D(pool_size=2, strides=2)(endBranch)
    # 2 - 1 conv
    endBranch = Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(endBranch)
    endBranch = BatchNormalization()(endBranch)
    endBranch = Activation('relu')(endBranch)
    endBranch = MaxPooling2D(pool_size=2, strides=2)(endBranch)
    # 3 - 2 conv
    endBranch = Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(endBranch)
    endBranch = BatchNormalization()(endBranch)
    endBranch = Activation('relu')(endBranch)
    endBranch = Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(endBranch)
    endBranch = BatchNormalization()(endBranch)
    endBranch = Activation('relu')(endBranch)
    endBranch = MaxPooling2D(pool_size=2, strides=2)(endBranch)
    # 4 - 2 conv
    endBranch = Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(endBranch)
    endBranch = BatchNormalization()(endBranch)
    endBranch = Activation('relu')(endBranch)
    endBranch = Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(endBranch)
    endBranch = BatchNormalization()(endBranch)
    endBranch = Activation('relu')(endBranch)
    endBranch = MaxPooling2D(pool_size=2, strides=2)(endBranch)
    # 5 - Flatten
    endBranch = Flatten()(endBranch)
    
    ################################
    # Now combine the start/end branches
    ################################
    combined = Concatenate()([startBranch, endBranch])
    
    return combined
