from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout

def IvysaurusIChooseYou(nvars, nclasses):

    varsInputs = Input(shape=(nvars,))
    
    ################################
    # FC layers
    ################################
    combined = Dense(4096, activation="relu")(combined)
    combined = Dropout(0.5)(combined)
    combined = Dense(4096, activation="relu")(combined)
    combined = Dropout(0.5)(combined)

    ################################
    # Now classify the image
    ################################
    outputs = Dense(nclasses, activation="softmax")(combined)

    ################################
    # Define model
    ################################    
    model = Model(inputs=varsInputs, outputs=outputs)
    
    return model

##########################################################################################################################
##########################################################################################################################
