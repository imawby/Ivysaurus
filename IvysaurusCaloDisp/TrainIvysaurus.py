print('111')

import numpy as np
import math
import glob
import sys

print('222')

import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

print('333')

from sklearn.utils import class_weight

print('444')

import IvysaurusModel_VGG
import CombinedIvysaurusModel

print('555')

###########################################################
###########################################################

print('AAAAAAA')

physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0 :
    for gpu in physical_devices :
       tensorflow.config.experimental.set_memory_growth(gpu, True)

print('1111111')

###########################################################

useCaloModel = True
useDispModel = True
useTrackVars = True
useShowerVars = True 

nModels = 0 # calorimetry, displacement

for model in [useCaloModel, useDispModel] :
    if model :
        nModels = nModels + 1

if (nModels == 0) :
    print("NO GRID BASED MODEL!")
    exit()
        
nTrackVars = 10 # nTrackChildren, nShowerChildren, nGrandChildren, nChildHits, childEnergy, childTrackScore, trackLength, trackWobble, trackScore, momComparison
nShowerVars = 16 # displacement, dca, trackStubLength

if (not useTrackVars) :
    nTrackVars = 0

if (not useShowerVars) :
    nShowerVars = 0

nVars = (nModels * 5) + nTrackVars + nShowerVars

dimensions = 24
nClasses = 5
nEpochs = 10
batchSize = 64
learningRate = 1e-4

###########################################################

# Here we'll get our information...

print('BBBBBBB')

# Calo grids
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

# Displacement grids
startGridU_disp_train = np.empty((0, dimensions, dimensions, 1))
startGridV_disp_train = np.empty((0, dimensions, dimensions, 1))
startGridW_disp_train = np.empty((0, dimensions, dimensions, 1))

endGridU_disp_train = np.empty((0, dimensions, dimensions, 1))
endGridV_disp_train = np.empty((0, dimensions, dimensions, 1))
endGridW_disp_train = np.empty((0, dimensions, dimensions, 1))

startGridU_disp_test = np.empty((0, dimensions, dimensions, 1))
startGridV_disp_test = np.empty((0, dimensions, dimensions, 1))
startGridW_disp_test = np.empty((0, dimensions, dimensions, 1))

endGridU_disp_test = np.empty((0, dimensions, dimensions, 1))
endGridV_disp_test = np.empty((0, dimensions, dimensions, 1))
endGridW_disp_test = np.empty((0, dimensions, dimensions, 1))

# Track and shower vars
trackVars_train = np.empty((0, nTrackVars))
showerVars_train = np.empty((0, nShowerVars))

trackVars_test = np.empty((0, nTrackVars))
showerVars_test = np.empty((0, nShowerVars))

# Truth
y_train = np.empty((0, nClasses))
y_test = np.empty((0, nClasses))

print('CCCCCCC')

# Get training file
trainFileNames = glob.glob('/storage/users/mawbyi1/Ivysaurus/files/gaussian/*/ivysaurus_*.npz')
print(trainFileNames)

for trainFileName in trainFileNames :
    print('Reading file: ', str(trainFileName),', This may take a while...')
    
    data = np.load(trainFileName)

    # Calo grids
    if (useCaloModel) :
        startGridU_calo_train = np.concatenate((startGridU_calo_train, data['startGridU_calo_train']), axis=0)
        startGridV_calo_train = np.concatenate((startGridV_calo_train, data['startGridV_calo_train']), axis=0)
        startGridW_calo_train = np.concatenate((startGridW_calo_train, data['startGridW_calo_train']), axis=0)

        endGridU_calo_train = np.concatenate((endGridU_calo_train, data['endGridU_calo_train']), axis=0)
        endGridV_calo_train = np.concatenate((endGridV_calo_train, data['endGridV_calo_train']), axis=0)
        endGridW_calo_train = np.concatenate((endGridW_calo_train, data['endGridW_calo_train']), axis=0)
    
        startGridU_calo_test = np.concatenate((startGridU_calo_test, data['startGridU_calo_test']), axis=0)
        startGridV_calo_test = np.concatenate((startGridV_calo_test, data['startGridV_calo_test']), axis=0) 
        startGridW_calo_test = np.concatenate((startGridW_calo_test, data['startGridW_calo_test']), axis=0)
    
        endGridU_calo_test = np.concatenate((endGridU_calo_test, data['endGridU_calo_test']), axis=0)
        endGridV_calo_test = np.concatenate((endGridV_calo_test, data['endGridV_calo_test']), axis=0)
        endGridW_calo_test = np.concatenate((endGridW_calo_test, data['endGridW_calo_test']), axis=0)

    # Disp grids
    if (useDispModel) :
        startGridU_disp_train = np.concatenate((startGridU_disp_train, data['startGridU_disp_train']), axis=0)
        startGridV_disp_train = np.concatenate((startGridV_disp_train, data['startGridV_disp_train']), axis=0)
        startGridW_disp_train = np.concatenate((startGridW_disp_train, data['startGridW_disp_train']), axis=0)

        endGridU_disp_train = np.concatenate((endGridU_disp_train, data['endGridU_disp_train']), axis=0)
        endGridV_disp_train = np.concatenate((endGridV_disp_train, data['endGridV_disp_train']), axis=0)
        endGridW_disp_train = np.concatenate((endGridW_disp_train, data['endGridW_disp_train']), axis=0)
    
        startGridU_disp_test = np.concatenate((startGridU_disp_test, data['startGridU_disp_test']), axis=0)
        startGridV_disp_test = np.concatenate((startGridV_disp_test, data['startGridV_disp_test']), axis=0) 
        startGridW_disp_test = np.concatenate((startGridW_disp_test, data['startGridW_disp_test']), axis=0)
    
        endGridU_disp_test = np.concatenate((endGridU_disp_test, data['endGridU_disp_test']), axis=0)
        endGridV_disp_test = np.concatenate((endGridV_disp_test, data['endGridV_disp_test']), axis=0)
        endGridW_disp_test = np.concatenate((endGridW_disp_test, data['endGridW_disp_test']), axis=0)
    
    # Track and shower vars
    if (useTrackVars) :
        trackVars_train = np.concatenate((trackVars_train, data['trackVars_train']), axis=0)
        trackVars_test = np.concatenate((trackVars_test, data['trackVars_test']), axis=0)

    if (useShowerVars) :
        showerVars_train = np.concatenate((showerVars_train, data['showerVars_train']), axis=0)
        showerVars_test = np.concatenate((showerVars_test, data['showerVars_test']), axis=0)

    # Truth
    y_train = np.concatenate((y_train, data['y_train']), axis=0)
    y_test = np.concatenate((y_test, data['y_test']), axis=0) 
    
print('DDDDDDD')

###########################################################

# I need to normalise the displacement grid here..

print('Normalising displacement grid wrt all other grids...')

dispLimit = 295.0

startGridU_disp_train[startGridU_disp_train > dispLimit] = dispLimit
startGridU_disp_train = startGridU_disp_train / dispLimit
    
startGridV_disp_train[startGridV_disp_train > dispLimit] = dispLimit
startGridV_disp_train = startGridV_disp_train / dispLimit
    
startGridW_disp_train[startGridW_disp_train > dispLimit] = dispLimit
startGridW_disp_train = startGridW_disp_train / dispLimit
    
endGridU_disp_train[endGridU_disp_train > dispLimit] = dispLimit
endGridU_disp_train = endGridU_disp_train / dispLimit
    
endGridV_disp_train[endGridV_disp_train > dispLimit] = dispLimit
endGridV_disp_train = endGridV_disp_train / dispLimit
    
endGridW_disp_train[endGridW_disp_train > dispLimit] = dispLimit
endGridW_disp_train = endGridW_disp_train / dispLimit

startGridU_disp_test[startGridU_disp_test > dispLimit] = dispLimit
startGridU_disp_test = startGridU_disp_test / dispLimit
    
startGridV_disp_test[startGridV_disp_test > dispLimit] = dispLimit
startGridV_disp_test = startGridV_disp_test / dispLimit
    
startGridW_disp_test[startGridW_disp_test > dispLimit] = dispLimit
startGridW_disp_test = startGridW_disp_test / dispLimit
    
endGridU_disp_test[endGridU_disp_test > dispLimit] = dispLimit
endGridU_disp_test = endGridU_disp_test / dispLimit
    
endGridV_disp_test[endGridV_disp_test > dispLimit] = dispLimit
endGridV_disp_test = endGridV_disp_test / dispLimit
    
endGridW_disp_test[endGridW_disp_test > dispLimit] = dispLimit
endGridW_disp_test = endGridW_disp_test / dispLimit

###########################################################

# Calo grid
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

# Disp grid
print('startGridU_disp_train: ', startGridU_disp_train.shape)
print('startGridV_disp_train: ', startGridV_disp_train.shape)
print('startGridW_disp_train: ', startGridW_disp_train.shape)

print('endGridU_disp_train: ', endGridU_disp_train.shape)    
print('endGridV_disp_train: ', endGridV_disp_train.shape)
print('endGridW_disp_train: ', endGridW_disp_train.shape)

print('startGridU_disp_test: ', startGridU_disp_test.shape)
print('startGridV_disp_test: ', startGridV_disp_test.shape)
print('startGridW_disp_test: ', startGridW_disp_test.shape)
  
print('endGridU_disp_test: ', endGridU_disp_test.shape)     
print('endGridV_disp_test: ', endGridV_disp_test.shape)     
print('endGridW_disp_test: ', endGridW_disp_test.shape) 

# Track and shower vars
print('trackVars_train: ', trackVars_train.shape)    
print('showerVars_train: ', showerVars_train.shape)  

print('trackVars_test: ', trackVars_test.shape)
print('showerVars_test: ', showerVars_test.shape)  

print('y_train: ', y_train.shape)
print('y_test', y_test.shape)

###########################################################
# Calculate calo scores

# Predict scores
if (useCaloModel) :
    ivysaurusCalo = load_model('/storage/users/mawbyi1/Ivysaurus/files/gaussian/my_model_calo_VGG')
    print("Loaded calo model from disk")
    
    ivysaurusScores_calo_train = ivysaurusCalo.predict([startGridU_calo_train, endGridU_calo_train, startGridV_calo_train, endGridV_calo_train, startGridW_calo_train, endGridW_calo_train])
    ivysaurusScores_calo_test = ivysaurusCalo.predict([startGridU_calo_test, endGridU_calo_test, startGridV_calo_test, endGridV_calo_test, startGridW_calo_test, endGridW_calo_test])

###########################################################
# Calculate disp scores

# load model
if (useDispModel) :
    ivysaurusDisp = load_model('/storage/users/mawbyi1/Ivysaurus/files/gaussian/my_model_disp_VGG')
    print("Loaded disp model from disk")

    ivysaurusScores_disp_train = ivysaurusDisp.predict([startGridU_disp_train, endGridU_disp_train, startGridV_disp_train, endGridV_disp_train, startGridW_disp_train, endGridW_disp_train])
    ivysaurusScores_disp_test = ivysaurusDisp.predict([startGridU_disp_test, endGridU_disp_test, startGridV_disp_test, endGridV_disp_test, startGridW_disp_test, endGridW_disp_test])

###########################################################
# Prep model input

if (useCaloModel) :
    combinedVars_train = ivysaurusScores_calo_train
    combinedVars_test = ivysaurusScores_calo_test

    if (useDispModel) :
        combinedVars_train = np.concatenate((combinedVars_train, ivysaurusScores_disp_train), axis=1)
        combinedVars_test = np.concatenate((combinedVars_test, ivysaurusScores_disp_test), axis=1)

elif (useDispModel) :
    combinedVars_train = ivysaurusScores_disp_train
    combinedVars_test = ivysaurusScores_disp_test

    if (useCaloModel) :
        combinedVars_train = np.concatenate((combinedVars_train, ivysaurusScores_calo_train), axis=1)
        combinedVars_test = np.concatenate((combinedVars_test, ivysaurusScores_calo_test), axis=1)


if (useTrackVars) :
    combinedVars_train = np.concatenate((combinedVars_train, trackVars_train), axis=1)
    combinedVars_test = np.concatenate((combinedVars_test, trackVars_test), axis=1)

if (useShowerVars) :
    combinedVars_train = np.concatenate((combinedVars_train, showerVars_train), axis=1)
    combinedVars_test = np.concatenate((combinedVars_test, showerVars_test), axis=1)

###########################################################

mirrored_strategy = tensorflow.distribute.MultiWorkerMirroredStrategy()

print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

with mirrored_strategy.scope():
   combinedIvysaurus = CombinedIvysaurusModel.IvysaurusIChooseYou(nVars, nClasses)
   combinedIvysaurus.summary()

   # Define the optimiser and compile the model
   optimiser = Adam(learning_rate=learningRate)
   combinedIvysaurus.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

###########################################################
# Create class weights

indexVector = np.argmax(y_train, axis=1)

# muons = 0, protons = 1, pions = 2, electrons = 3, photons = 4, other = 5
nMuons = np.count_nonzero(indexVector == 0)    
nProtons = np.count_nonzero(indexVector == 1)  
nPions = np.count_nonzero(indexVector == 2)  
nElectrons = np.count_nonzero(indexVector == 3)  
nPhotons = np.count_nonzero(indexVector == 4)  

# Normalise to largest
maxParticle = max(nMuons, nProtons, nPions, nElectrons, nPhotons)

classWeights = {0: maxParticle/nMuons, 1: maxParticle/nProtons, 2: maxParticle/nPions, 3: maxParticle/nElectrons, 4: maxParticle/nPhotons}

print('Class Weights: ')
print(classWeights)

###########################################################

# Fit that model!

if (useCaloModel and useDispModel) :
    filePath = '/storage/users/mawbyi1/Ivysaurus/files/gaussian/my_model_combined_VGG'
elif (useCaloModel) :
    filePath = '/storage/users/mawbyi1/Ivysaurus/files/gaussian/my_model_combined_caloNoDisp_VGG'
elif (useDispModel) :
    filePath = '/storage/users/mawbyi1/Ivysaurus/files/gaussian/my_model_combined_dispNoCalo_VGG'
    
# checkpoint
checkpoint = ModelCheckpoint(filePath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Reduce the learning rate by a factor of ten when required
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6, verbose=1)

callbacks_list = [checkpoint, reduce_lr]

# Train!
history = combinedIvysaurus.fit(combinedVars_train, y_train, 
    batch_size = batchSize, validation_data=(combinedVars_test, y_test), 
    shuffle=True, epochs=nEpochs, class_weight=classWeights, callbacks=callbacks_list, verbose=2)

###########################################################

# FIN!

print('DONE!')


