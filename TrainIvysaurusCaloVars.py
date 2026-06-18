import argparse
import numpy as np
import math
import glob
import sys

# RUN ON GPU:1 (GPU:1 will be renamed to GPU:0)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight

import IvysaurusModel_VGG_BN_VARS

###########################################################

dimensions = 24
nClasses = 5

# nTrackChildren, nTrackChildren_valid, nShowerChildren, nShowerChildren_valid, nGrandChildren, nGrandChildren_valid, nChildHits, nChildHits_valid,
# childEnergy, childEnergy_valid, childTrackScore, childTrackScore_valid, trackLength, trackLength_valid, trackWobble, trackWobble_valid,
# trackScore, trackScore_valid, momComparison, momComparison_valid
nTrackVars = 21

# displacement, displacement_valid, dca, dca_valid, trackStubLength, trackStubLength_valid,
# nuVertexAvSeparation, nuVertexAvSeparation_valid, nuVertexChargeAsymmetry, nuVertexChargeAsymmetry_valid
nShowerVars = 10

batchSize = 64
learningRate = 1e-4

###########################################################

def WhoseThatPokemon(ndimensions, nclasses, nTrackVars, nShowerVars) :
    return IvysaurusModel_VGG_BN_VARS.IvysaurusIChooseYou(ndimensions, nclasses, nTrackVars, nShowerVars)

###########################################################
###########################################################

def main(args) :

    physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')

    print('physical_devices:', physical_devices)

    # Make sure TF doesnt hog GPU resources
    if len(physical_devices) > 0 :
        for gpu in physical_devices :
            tensorflow.config.experimental.set_memory_growth(gpu, True)


    # Here we'll get our information...
    startGridU_calo_train = np.empty((0, dimensions, dimensions, 1))
    startGridV_calo_train = np.empty((0, dimensions, dimensions, 1))
    startGridW_calo_train = np.empty((0, dimensions, dimensions, 1))
    endGridU_calo_train = np.empty((0, dimensions, dimensions, 1))
    endGridV_calo_train = np.empty((0, dimensions, dimensions, 1))
    endGridW_calo_train = np.empty((0, dimensions, dimensions, 1))
    startGridU_calo_test = np.empty((0, dimensions, dimensions, 1))
    startGridV_calo_test = np.empty((0, dimensions, dimensions, 1))
    startGridW_calo_test = np.empty((0, dimensions, dimensions, 1))
    endGridU_calo_test = np.empty((0, dimensions, dimensions, 1))
    endGridV_calo_test = np.empty((0, dimensions, dimensions, 1))
    endGridW_calo_test = np.empty((0, dimensions, dimensions, 1))
    trackVars_train = np.empty((0, nTrackVars))
    showerVars_train = np.empty((0, nShowerVars))
    trackVars_test = np.empty((0, nTrackVars))
    showerVars_test = np.empty((0, nShowerVars))
    y_train = np.empty((0, nClasses))
    y_test = np.empty((0, nClasses))

    # Get training file
    trainFileNames = glob.glob(f'/home/imawby/Ivysaurus/files/filtered_0_{"Contained" if args.is_contained else "Exiting"}.npz')
    print(trainFileNames)

    for trainFileName in trainFileNames :
        print('Reading file: ', str(trainFileName),', This may take a while...')
    
        data = np.load(trainFileName)

        startGridU_calo_train = np.concatenate((startGridU_calo_train, data['startGridU_train']), axis=0)
        startGridV_calo_train = np.concatenate((startGridV_calo_train, data['startGridV_train']), axis=0)
        startGridW_calo_train = np.concatenate((startGridW_calo_train, data['startGridW_train']), axis=0)
        endGridU_calo_train = np.concatenate((endGridU_calo_train, data['endGridU_train']), axis=0)
        endGridV_calo_train = np.concatenate((endGridV_calo_train, data['endGridV_train']), axis=0)
        endGridW_calo_train = np.concatenate((endGridW_calo_train, data['endGridW_train']), axis=0)
        startGridU_calo_test = np.concatenate((startGridU_calo_test, data['startGridU_test']), axis=0)
        startGridV_calo_test = np.concatenate((startGridV_calo_test, data['startGridV_test']), axis=0) 
        startGridW_calo_test = np.concatenate((startGridW_calo_test, data['startGridW_test']), axis=0)
        endGridU_calo_test = np.concatenate((endGridU_calo_test, data['endGridU_test']), axis=0)
        endGridV_calo_test = np.concatenate((endGridV_calo_test, data['endGridV_test']), axis=0)
        endGridW_calo_test = np.concatenate((endGridW_calo_test, data['endGridW_test']), axis=0)
        trackVars_train = np.concatenate((trackVars_train, data['trackVars_train']), axis=0)
        trackVars_test = np.concatenate((trackVars_test, data['trackVars_test']), axis=0)
        showerVars_train = np.concatenate((showerVars_train, data['showerVars_train']), axis=0)
        showerVars_test = np.concatenate((showerVars_test, data['showerVars_test']), axis=0)
        y_train = np.concatenate((y_train, data['y_train']), axis=0)
        y_test = np.concatenate((y_test, data['y_test']), axis=0)

    print('startGridU_calo_train: ', startGridU_calo_train.shape)
    print('startGridV_calo_train: ', startGridV_calo_train.shape)
    print('startGridW_calo_train: ', startGridW_calo_train.shape)
    print('endGridU_calo_train: ', endGridU_calo_train.shape)    
    print('endGridV_calo_train: ', endGridV_calo_train.shape)
    print('endGridW_calo_train: ', endGridW_calo_train.shape)
    print('startGridU_calo_test: ', startGridU_calo_test.shape)
    print('startGridV_calo_test: ', startGridV_calo_test.shape)
    print('startGridW_calo_test: ', startGridW_calo_test.shape)
    print('endGridU_calo_test: ', endGridU_calo_test.shape)     
    print('endGridV_calo_test: ', endGridV_calo_test.shape)     
    print('endGridW_calo_test: ', endGridW_calo_test.shape)
    print('trackVars_train: ', trackVars_train.shape)    
    print('showerVars_train: ', showerVars_train.shape)  
    print('trackVars_test: ', trackVars_test.shape)
    print('showerVars_test: ', showerVars_test.shape)  
    print('y_train: ', y_train.shape)
    print('y_test', y_test.shape)

    ivysaurusCalo = WhoseThatPokemon(dimensions, nClasses, nTrackVars, nShowerVars)
    #ivysaurusCalo.summary()

    # Define the optimiser and compile the model
    optimiser = Adam(learning_rate=learningRate)
    ivysaurusCalo.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

    # Create class weights
    # muons = 0, protons = 1, pions = 2, electrons = 3, photons = 4, other = 5    
    indexVector = np.argmax(y_test, axis=1)
    nMuons = np.count_nonzero(indexVector == 0)    
    nProtons = np.count_nonzero(indexVector == 1)  
    nPions = np.count_nonzero(indexVector == 2)  
    nElectrons = np.count_nonzero(indexVector == 3)  
    nPhotons = np.count_nonzero(indexVector == 4)  
    maxParticle = max(nMuons, nProtons, nPions, nElectrons, nPhotons)
    classWeights = {0: maxParticle/nMuons, 1: maxParticle/nProtons, 2: maxParticle/nPions, 3: maxParticle/nElectrons, 4: maxParticle/nPhotons}

    print('Class Weights: ')
    print(classWeights)

    # Fit that model!
    filePath = f'/home/imawby/Ivysaurus/models/my_model_{"contained" if args.is_contained else "exiting"}_VGG_BN_VARS'

    # checkpoint
    checkpoint = ModelCheckpoint(filePath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # Reduce the learning rate by a factor of ten when required
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6, verbose=1)

    callbacks_list = [checkpoint, reduce_lr]
    
    history = ivysaurusCalo.fit([startGridU_calo_train, endGridU_calo_train, startGridV_calo_train,
                                 endGridV_calo_train, startGridW_calo_train, endGridW_calo_train,
                                 trackVars_train, showerVars_train], y_train,
                                batch_size = batchSize,
                                validation_data=([startGridU_calo_test, endGridU_calo_test, startGridV_calo_test,
                                                  endGridV_calo_test, startGridW_calo_test, endGridW_calo_test,
                                                  trackVars_test, showerVars_test], y_test),
                                
    shuffle=True, epochs=args.n_epochs, class_weight=classWeights, callbacks=callbacks_list, verbose=2)

##########################################################################################################
            
def parse_cli():
    parser = argparse.ArgumentParser(description="Ivysaurus PID")
    parser.add_argument("--is_contained", type=bool, required=True, help="Training for contained particles?")
    parser.add_argument("--n_epochs", type=int, required=True, help="Number of epochs")    
    return parser.parse_args()

##########################################################################################################

if __name__ == "__main__":
    args = parse_cli()
    main(args)
