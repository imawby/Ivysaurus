from tensorflow import keras
from tensorflow.keras import layers

# start/endGrid.shape = (n, dimensions, dimensions, 1)
def IvysaurusIChooseYou(dimensions, nclasses, nTrackVars, nShowerVars):
    
    ################################
    # U View
    ################################
    startInputsU = keras.Input(shape=(dimensions, dimensions, 1))
    endInputsU = keras.Input(shape=(dimensions, dimensions, 1))
    branchU = CreateViewBranch(dimensions, startInputsU, endInputsU)
    
    ################################
    # V View
    ################################
    startInputsV = keras.Input(shape=(dimensions, dimensions, 1))
    endInputsV = keras.Input(shape=(dimensions, dimensions, 1))
    branchV = CreateViewBranch(dimensions, startInputsV, endInputsV)    
    
    ################################
    # W View
    ################################
    startInputsW = keras.Input(shape=(dimensions, dimensions, 1))
    endInputsW = keras.Input(shape=(dimensions, dimensions, 1))
    branchW = CreateViewBranch(dimensions, startInputsW, endInputsW)
    
    ################################
    # Now combine the U, V and W branches
    ################################
    combined = keras.layers.Concatenate()([branchU, branchV, branchW])
        
    ################################
    # Now add in the trackVars
    ################################
    trackVarsInputs = keras.Input(shape=(nTrackVars,))
    combined = keras.layers.Concatenate()([combined, trackVarsInputs])

    ################################
    # Now add in the showerVars
    ################################
    trackVarsInputs = keras.Input(shape=(nShowerVars,))
    combined = keras.layers.Concatenate()([combined, showerVarsInputs])
    
    ################################
    # FC layers
    ################################
    combined = keras.layers.Dense(4096, use_bias=False)(combined)
    combined = keras.layers.BatchNormalization()(combined)
    combined = keras.layers.Activation('relu')(combined)
    combined = keras.layers.Dropout(0.5)(combined)
    combined = keras.layers.Dense(4096, use_bias=False)(combined)
    combined = keras.layers.BatchNormalization()(combined)
    combined = keras.layers.Activation('relu')(combined)
    combined = keras.layers.Dropout(0.5)(combined)

    ################################
    # Now classify the image
    ################################
    outputs = keras.layers.Dense(nclasses, activation="softmax")(combined)
    model = keras.Model(inputs=[startInputsU, endInputsU, startInputsV, endInputsV, startInputsW, endInputsW, trackVarsInputs, showerVarsInputs], outputs=outputs)
    
    return model

##########################################################################################################################
##########################################################################################################################

def CreateViewBranch(dimensions, startInputs, endInputs):
    ################################
    # Start branch
    ################################
    # 1 - 1 conv
    startBranch = layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(startInputs)
    startBranch = layers.BatchNormalization()(startBranch)
    startBranch = layers.Activation('relu')(startBranch)
    startBranch = layers.MaxPooling2D(pool_size=2, strides=2)(startBranch)
    # 2 - 1 conv
    startBranch = layers.Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(startBranch)
    startBranch = layers.BatchNormalization()(startBranch)
    startBranch = layers.Activation('relu')(startBranch)
    startBranch = layers.MaxPooling2D(pool_size=2, strides=2)(startBranch)
    # 3 - 2 conv
    startBranch = layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(startBranch)
    startBranch = layers.BatchNormalization()(startBranch)
    startBranch = layers.Activation('relu')(startBranch)
    startBranch = layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(startBranch)
    startBranch = layers.BatchNormalization()(startBranch)
    startBranch = layers.Activation('relu')(startBranch)
    startBranch = layers.MaxPooling2D(pool_size=2, strides=2)(startBranch)
    # 4 - 2 conv
    startBranch = layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(startBranch)
    startBranch = layers.BatchNormalization()(startBranch)
    startBranch = layers.Activation('relu')(startBranch)
    startBranch = layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(startBranch)
    startBranch = layers.BatchNormalization()(startBranch)
    startBranch = layers.Activation('relu')(startBranch)
    startBranch = layers.MaxPooling2D(pool_size=2, strides=2)(startBranch)
    # 5 - Flatten
    startBranch = layers.Flatten()(startBranch)

    ################################
    # End branch
    ################################
    # 1 - 1 conv
    endBranch = layers.Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False)(endInputs)
    endBranch = layers.BatchNormalization()(endBranch)
    endBranch = layers.Activation('relu')(endBranch)
    endBranch = layers.MaxPooling2D(pool_size=2, strides=2)(endBranch)
    # 2 - 1 conv
    endBranch = layers.Conv2D(filters=32, kernel_size=3, padding="same", use_bias=False)(endBranch)
    endBranch = layers.BatchNormalization()(endBranch)
    endBranch = layers.Activation('relu')(endBranch)
    endBranch = layers.MaxPooling2D(pool_size=2, strides=2)(endBranch)
    # 3 - 2 conv
    endBranch = layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(endBranch)
    endBranch = layers.BatchNormalization()(endBranch)
    endBranch = layers.Activation('relu')(endBranch)
    endBranch = layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(endBranch)
    endBranch = layers.BatchNormalization()(endBranch)
    endBranch = layers.Activation('relu')(endBranch)
    endBranch = layers.MaxPooling2D(pool_size=2, strides=2)(endBranch)
    # 4 - 2 conv
    endBranch = layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(endBranch)
    endBranch = layers.BatchNormalization()(endBranch)
    endBranch = layers.Activation('relu')(endBranch)
    endBranch = layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(endBranch)
    endBranch = layers.BatchNormalization()(endBranch)
    endBranch = layers.Activation('relu')(endBranch)
    endBranch = layers.MaxPooling2D(pool_size=2, strides=2)(endBranch)
    # 5 - Flatten
    endBranch = layers.Flatten()(endBranch)
    
    ################################
    # Now combine the start/end branches
    ################################
    combined = keras.layers.Concatenate()([startBranch, endBranch])
    
    return combined
