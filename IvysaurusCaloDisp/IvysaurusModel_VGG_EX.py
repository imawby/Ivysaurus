from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Conv2D, MaxPooling2D, Flatten

# start/endGrid.shape = (n, dimensions, dimensions, 1)
def IvysaurusIChooseYou(dimensions, nclasses):
    
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
    # FC layers
    ################################
    combined = Dense(4096, activation="relu")(combined)
    combined = Dense(4096, activation="relu")(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(1000, activation="relu")(combined)
    combined = Dropout(0.5)(combined)
    
    ################################
    # Now classify the image
    ################################
    outputs = Dense(nclasses, activation="softmax")(combined)
    
    model = Model(inputs=[startInputsU, endInputsU, startInputsV, endInputsV, startInputsW, endInputsW], outputs=outputs)
    
    return model

##########################################################################################################################
##########################################################################################################################

def CreateViewBranch(dimensions, startInputs, endInputs):
    ################################
    # Start branch
    ################################
    # 1 - 2 conv
    startBranch = Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(startInputs)
    startBranch = Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = MaxPooling2D(pool_size=2, strides=1)(startBranch)
    startBranch = Dropout(0.5)(startBranch)
    # 2 - 1 conv
    startBranch = Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = MaxPooling2D(pool_size=2, strides=2)(startBranch)
    # 3 - 2 conv
    startBranch = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = MaxPooling2D(pool_size=2, strides=2)(startBranch)
    # 3 - 2 conv
    startBranch = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = MaxPooling2D(pool_size=2, strides=2)(startBranch)    
    # 4 - 2 conv
    startBranch = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(startBranch)
    startBranch = MaxPooling2D(pool_size=2, strides=2)(startBranch)
    # 5 - Flatten
    startBranch = Flatten()(startBranch)

    ################################
    # End branch
    ################################
    # 1 - 1 conv
    endBranch = Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(endInputs)
    endBranch = Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = MaxPooling2D(pool_size=2, strides=1)(endBranch)
    endBranch = Dropout(0.5)(endBranch)    
    # 2 - 1 conv
    endBranch = Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = Conv2D(filters=64, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = MaxPooling2D(pool_size=2, strides=2)(endBranch)
    # 3 - 2 conv
    endBranch = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = MaxPooling2D(pool_size=2, strides=2)(endBranch)
    # 3 - 2 conv
    endBranch = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = Conv2D(filters=128, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = MaxPooling2D(pool_size=2, strides=2)(endBranch)    
    # 4 - 2 conv
    endBranch = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(endBranch)
    endBranch = MaxPooling2D(pool_size=2, strides=2)(endBranch)
    # 5 - Flatten
    endBranch = Flatten()(endBranch)
    
    ################################
    # Now combine the start/end branches
    ################################
    combined = Concatenate()([startBranch, endBranch])
    
    return combined
