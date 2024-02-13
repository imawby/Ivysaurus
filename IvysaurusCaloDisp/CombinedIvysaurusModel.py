from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout

def IvysaurusIChooseYou(nvars, nclasses):

    varsInputs = Input(shape=(nvars,))
    
    ################################
    # Now classify the image
    ################################
    x = Dense(128, activation="relu", kernel_initializer='lecun_uniform')(varsInputs)
    #x = Dropout(dropoutRate)(x)
    x = Dense(128, activation="relu", kernel_initializer='lecun_uniform')(x)
    #x = Dropout(dropoutRate)(x)
    x = Dense(64, activation="relu", kernel_initializer='lecun_uniform')(x)
    #x = Dropout(dropoutRate)(x)
    
    outputs = Dense(nclasses, activation="softmax")(x)

    ################################
    # Define model
    ################################    
    model = Model(inputs=varsInputs, outputs=outputs)
    
    return model

##########################################################################################################################
##########################################################################################################################
