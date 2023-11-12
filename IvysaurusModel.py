from tensorflow import keras
from tensorflow.keras import layers

# start/endGrid.shape = (n, dimensions, dimensions, 1)
def IvysaurusIChooseYou(dimensions, nclasses, meanStartU, varStartU, meanStartV, varStartV, meanStartW, varStartW, meanEndU, varEndU, meanEndV, varEndV, meanEndW, varEndW):
    
    ################################
    # U View
    ################################
    startInputsU = keras.layers.Normalization(-1, mean=meanStartU, variance=varStartU)
    startInputsU = keras.Input(shape=(dimensions, dimensions, 1))
    endInputsU = keras.layers.Normalization(-1, mean=meanEndU, variance=varEndU)
    endInputsU = keras.Input(shape=(dimensions, dimensions, 1))
    branchU = CreateViewBranch(dimensions, startInputsU, endInputsU)
    
    ################################
    # V View
    ################################
    startInputsV = keras.layers.Normalization(-1, mean=meanStartV, variance=varStartV)
    startInputsV = keras.Input(shape=(dimensions, dimensions, 1))
    endInputsV = keras.layers.Normalization(-1, mean=meanEndV, variance=varEndV)
    endInputsV = keras.Input(shape=(dimensions, dimensions, 1))
    branchV = CreateViewBranch(dimensions, startInputsV, endInputsV)    
    
    ################################
    # W View
    ################################
    startInputsW = keras.layers.Normalization(-1, mean=meanStartW, variance=varStartW)
    startInputsW = keras.Input(shape=(dimensions, dimensions, 1))
    endInputsW = keras.layers.Normalization(-1, mean=meanEndW, variance=varEndW)
    endInputsW = keras.Input(shape=(dimensions, dimensions, 1))
    branchW = CreateViewBranch(dimensions, startInputsW, endInputsW)
    
    # Now combine the U, V and W branches
    combined = keras.layers.Concatenate()([branchU, branchV, branchW])
    
    # FC layer??
    combined = keras.layers.Dense(128, activation="relu")(combined)
    
    # Now classify the image
    outputs = keras.layers.Dense(nclasses, activation="softmax")(combined)
    
    model = keras.Model(inputs=[startInputsU, endInputsU, startInputsV, endInputsV, startInputsW, endInputsW], outputs=outputs)
    
    return model


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
    #startBranch = layers.MaxPooling2D(pool_size=2)(startBranch)
    # 4
    startBranch = layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(startBranch)
    #startBranch = layers.MaxPooling2D(pool_size=2)(startBranch)
    # 5
    #startBranch = layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(startBranch)
    #startBranch = layers.MaxPooling2D(pool_size=2)(startBranch) 
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
    #endBranch = layers.MaxPooling2D(pool_size=2)(endBranch)    
    # 4
    endBranch = layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(endBranch)
    #endBranch = layers.MaxPooling2D(pool_size=2)(endBranch)    
    # 5
    #endBranch = layers.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(endBranch)
    #endBranch = layers.MaxPooling2D(pool_size=2)(endBranch)
    endBranch = layers.Flatten()(endBranch)
    
    ################################
    # Now combine the start/end branches
    ################################
    combined = keras.layers.Concatenate()([startBranch, endBranch])
    
    return combined