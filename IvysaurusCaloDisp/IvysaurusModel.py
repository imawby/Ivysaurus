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
    # FC layer??
    ################################
    combined = keras.layers.Dense(128, activation="relu")(combined)
    
    ################################
    # Now add in the trackVars
    ################################
    trackVarsInputs = keras.Input(shape=(nTrackVars,))
    combined = keras.layers.Concatenate()([combined, trackVarsInputs])
    combined = keras.layers.Dense(128, activation="relu")(combined)

    ################################
    # Now classify the image
    ################################
    outputs = keras.layers.Dense(nclasses, activation="softmax")(combined)
    
    model = keras.Model(inputs=[startInputsU, endInputsU, startInputsV, endInputsV, startInputsW, endInputsW, trackVarsInputs], outputs=outputs)
    
    return model

################################################################################################################################################################
################################################################################################################################################################

def CreateViewBranch(dimensions, startInputs, endInputs):
    ################################
    # Start branch
    ################################
    # 1
    startBranch = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(startInputs)
    startBranch = layers.MaxPooling2D(pool_size=2)(startBranch)
    # 2
    startBranch = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = layers.MaxPooling2D(pool_size=2)(startBranch)
    # 3
    startBranch = layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(startBranch)
    # 4
    startBranch = layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(startBranch)
    # 5
    startBranch = layers.Flatten()(startBranch)

    ################################
    # End branch
    ################################
    # 1
    endBranch = layers.Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(endInputs)
    endBranch = layers.MaxPooling2D(pool_size=2)(endBranch)
    # 2
    endBranch = layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = layers.MaxPooling2D(pool_size=2)(endBranch)    
    # 3
    endBranch = layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(endBranch)
    # 4
    endBranch = layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(endBranch)
    # 5
    endBranch = layers.Flatten()(endBranch)
    
    ################################
    # Now combine the start/end branches
    ################################
    combined = keras.layers.Concatenate()([startBranch, endBranch])
    combined = keras.layers.Dense(128, activation="relu")(combined)
    
    return combined
