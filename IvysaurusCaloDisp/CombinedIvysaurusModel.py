from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout

def IvysaurusIChooseYou(nvars, nclasses):

    varsInputs = Input(shape=(nvars,))
    
    ################################
    # Now classify the image
    ################################
    outputs = Dense(nclasses, activation="softmax")(varsInputs)

    ################################
    # Define model
    ################################    
    model = Model(inputs=varsInputs, outputs=outputs)
    
    return model

##########################################################################################################################
##########################################################################################################################
