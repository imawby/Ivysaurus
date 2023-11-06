from tensorflow import keras
from tensorflow.keras import layers

# start/endGrid.shape = (n, dimensions, dimensions, 1)
def IvysaurusIChooseYou(dimensions, nclasses):
    
    # TODO: also need to do a normalisation - I think i've done that with energies
    
    ################################
    # U View
    ################################
    # Start U branch
    startInputsU = keras.Input(shape=(dimensions, dimensions, 1))
    startBranchU = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(startInputsU)
    startBranchU = layers.Flatten()(startBranchU)

    # End U branch
    endInputsU = keras.Input(shape=(dimensions, dimensions, 1))
    endBranchU = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(endInputsU)
    endBranchU = layers.Flatten()(endBranchU)
    
    # Now combine the U branches
    combinedU = keras.layers.Concatenate()([startBranchU, endBranchU])
    
    # Now classify the image
    outputs = layers.Dense(nclasses, activation="softmax")(combinedU)
    
    model = keras.Model(inputs=[startInputsU, endInputsU], outputs=outputs)
    
    return model