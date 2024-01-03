from tensorflow import keras
from tensorflow.keras import layers

# start/endGrid.shape = (n, dimensions, dimensions, 1)
def IvysaurusIChooseYou(dimensions, nclasses, nTrackVars):
    
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
    # FC layers
    ################################
    combined = keras.layers.Dense(4096, activation="relu")(combined)
    combined = keras.layers.Dropout(0.5)(combined)
    combined = keras.layers.Dense(4096, activation="relu")(combined)
    combined = keras.layers.Dropout(0.5)(combined)

    ################################
    # Now classify the image
    ################################
    outputs = keras.layers.Dense(nclasses, activation="softmax")(combined)
    
    model = keras.Model(inputs=[startInputsU, endInputsU, startInputsV, endInputsV, startInputsW, endInputsW, trackVarsInputs], outputs=outputs)
    
    return model

##########################################################################################################################
##########################################################################################################################

def CreateViewBranch(dimensions, startInputs, endInputs):
    ################################
    # Start branch
    ################################
    ################################
    # Stem
    ################################
    # 1 - 1 conv
    startBranch = layers.Conv2D(filters=64, kernel_size=7, activation="relu", padding="same")(startInputs)
    startBranch = layers.MaxPooling2D(pool_size=2, strides=1)(startBranch)
    # 2 - 2 conv
    startBranch = layers.Conv2D(filters=64, kernel_size=1, activation="relu", padding="same")(startBranch)
    startBranch = layers.Conv2D(filters=192, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = layers.MaxPooling2D(pool_size=2, strides=2)(startBranch) # Down to 12 x 12
    ################################
    # Body
    ################################   
    # 3 - 2 x Inception
    startBranch = PassThroughInceptionBlock(startBranch, 64, (96, 128), (16, 32), 32)
    startBranch = PassThroughInceptionBlock(startBranch, 128, (128, 192), (32, 96), 64)
    startBranch = layers.MaxPooling2D(pool_size=2, strides=2)(startBranch) # Down to 6 x 6
    # 4 - 5 x Inception
    startBranch = PassThroughInceptionBlock(startBranch, 192, (96, 208), (16, 48), 64)
    startBranch = PassThroughInceptionBlock(startBranch, 160, (112, 224), (24, 64), 64)
    startBranch = PassThroughInceptionBlock(startBranch, 128, (128, 256), (24, 64), 64)
    startBranch = PassThroughInceptionBlock(startBranch, 112, (114, 288), (32, 64), 64)    
    startBranch = PassThroughInceptionBlock(startBranch, 256, (160, 320), (32, 128), 128)
    startBranch = layers.MaxPooling2D(pool_size=2, strides=2)(startBranch) # Down to 3 x 3
    # 5 - 2 x Inception
    startBranch = PassThroughReducedInceptionBlock(startBranch, 256, (160, 320), 128) 
    startBranch = PassThroughReducedInceptionBlock(startBranch, 384, (192, 384), 128)
    ################################
    # Head
    ################################  
    startBranch = layers.GlobalAvgPool2D()(startBranch) # Down to 1 x 1
    startBranch = layers.Flatten()(startBranch)


    ################################
    # End branch
    ################################
    ################################
    # Stem
    ################################
    # 1 - 1 conv
    endBranch = layers.Conv2D(filters=64, kernel_size=7, activation="relu", padding="same")(endInputs)
    endBranch = layers.MaxPooling2D(pool_size=2, strides=1)(endBranch)
    # 2 - 2 conv
    endBranch = layers.Conv2D(filters=64, kernel_size=1, activation="relu", padding="same")(endBranch)
    endBranch = layers.Conv2D(filters=192, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = layers.MaxPooling2D(pool_size=2, strides=2)(endBranch) # Down to 12 x 12
    ################################
    # Body
    ################################   
    # 3 - 2 x Inception
    endBranch = PassThroughInceptionBlock(endBranch, 64, (96, 128), (16, 32), 32)
    endBranch = PassThroughInceptionBlock(endBranch, 128, (128, 192), (32, 96), 64)
    endBranch = layers.MaxPooling2D(pool_size=2, strides=2)(endBranch) # Down to 6 x 6
    # 4 - 5 x Inception
    endBranch = PassThroughInceptionBlock(endBranch, 192, (96, 208), (16, 48), 64)
    endBranch = PassThroughInceptionBlock(endBranch, 160, (112, 224), (24, 64), 64)
    endBranch = PassThroughInceptionBlock(endBranch, 128, (128, 256), (24, 64), 64)
    endBranch = PassThroughInceptionBlock(endBranch, 112, (114, 288), (32, 64), 64)    
    endBranch = PassThroughInceptionBlock(endBranch, 256, (160, 320), (32, 128), 128)
    endBranch = layers.MaxPooling2D(pool_size=2, strides=2)(endBranch) # Down to 3 x 3
    # 5 - 2 x Inception
    endBranch = PassThroughReducedInceptionBlock(endBranch, 256, (160, 320), 128) 
    endBranch = PassThroughReducedInceptionBlock(endBranch, 384, (192, 384), 128)
    ################################
    # Head
    ################################  
    endBranch = layers.GlobalAvgPool2D()(endBranch) # Down to 1 x 1
    endBranch = layers.Flatten()(endBranch)
    
    ################################
    # Now combine the start/end branches
    ################################
    combined = keras.layers.Concatenate()([startBranch, endBranch])
    
    return combined

##########################################################################################################################
##########################################################################################################################


def PassThroughInceptionBlock(flow, firstBrChannels, secondBrChannels, thirdBrChannels, forthBrChannels):
    # First branch
    firstOutput = layers.Conv2D(filters=firstBrChannels, kernel_size=1, activation="relu", padding="same")(flow)
    # Second branch
    secondOutput = layers.Conv2D(filters=secondBrChannels[0], kernel_size=1, activation="relu", padding="same")(flow)
    secondOutput = layers.Conv2D(filters=secondBrChannels[1], kernel_size=3, activation="relu", padding="same")(secondOutput)
    # Third branch
    thirdOutput = layers.Conv2D(filters=thirdBrChannels[0], kernel_size=1, activation="relu", padding="same")(flow)
    thirdOutput = layers.Conv2D(filters=thirdBrChannels[1], kernel_size=5, activation="relu", padding="same")(thirdOutput)
    # Forth branch
    forthOutput = layers.MaxPooling2D(pool_size=2, strides=1, padding='same')(flow)
    forthOutput = layers.Conv2D(filters=forthBrChannels, kernel_size=1, activation="relu", padding="same")(forthOutput)
    # Concatenate
    combinedOutput = layers.Concatenate()([firstOutput, secondOutput, thirdOutput, forthOutput])
    
    return combinedOutput

##########################################################################################################################
##########################################################################################################################

# For lower dimension inputs, get rid of the third branch
def PassThroughReducedInceptionBlock(flow, firstBrChannels, secondBrChannels, forthBrChannels):
    # First branch
    firstOutput = layers.Conv2D(filters=firstBrChannels, kernel_size=1, activation="relu", padding="same")(flow)
    # Second branch
    secondOutput = layers.Conv2D(filters=secondBrChannels[0], kernel_size=1, activation="relu", padding="same")(flow)
    secondOutput = layers.Conv2D(filters=secondBrChannels[1], kernel_size=3, activation="relu", padding="same")(secondOutput)
    # Forth branch
    forthOutput = layers.MaxPooling2D(pool_size=2, strides=1, padding='same')(flow)
    forthOutput = layers.Conv2D(filters=forthBrChannels, kernel_size=1, activation="relu", padding="same")(forthOutput)
    # Concatenate
    combinedOutput = layers.Concatenate()([firstOutput, secondOutput, forthOutput])
    
    return combinedOutput
    
    
    
    
    
    
    
    
    
    
    
    
    
