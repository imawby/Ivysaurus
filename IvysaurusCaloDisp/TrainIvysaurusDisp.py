print('111')

import numpy as np
import math
import glob
import sys

print('222')

import tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

print('333')

from sklearn.utils import class_weight

print('444')

import IvysaurusModel
import IvysaurusModel_VGG
import IvysaurusModel_VGG_BN
import IvysaurusModel_VGG_SEP
import IvysaurusModel_VGG_EX
import IvysaurusModel_VGG_RES

print('555')

###########################################################
###########################################################

def WhoseThatPokemon(ndimensions, nclasses, mode) :
    
   if (mode == '0') :
      print('IvysaurusModel_VGG')
      return IvysaurusModel_VGG.IvysaurusIChooseYou(ndimensions, nclasses)
   elif (mode == '1') :
      print('IvysaurusModel_VGG_BN')
      return IvysaurusModel_VGG_BN.IvysaurusIChooseYou(ndimensions, nclasses)
   elif (mode == '2') :
      print('IvysaurusModel_VGG_SEP')
      return IvysaurusModel_VGG_SEP.IvysaurusIChooseYou(ndimensions, nclasses)
   elif (mode == '3') :
      print('IvysaurusModel_VGG_EX')
      return IvysaurusModel_VGG_EX.IvysaurusIChooseYou(ndimensions, nclasses)
   elif (mode == '4') :
      print('IvysaurusModel_VGG_RES')
      return IvysaurusModel_VGG_RES.IvysaurusIChooseYou(ndimensions, nclasses)

   return IvysaurusModel_VGG.IvysaurusIChooseYou(ndimensions, nclasses)
      

###########################################################
###########################################################

print('AAAAAAA')

physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0 :
    for gpu in physical_devices :
       tensorflow.config.experimental.set_memory_growth(gpu, True)

print('1111111')

###########################################################

# VGG, VGG_BN, VGG_SEP, VGG_EX, VGG_RES
MODE_VGG = sys.argv[1]

print('MODE_VGG: ', MODE_VGG)

dimensions = 24
nClasses = 5

nEpochs = 10
batchSize = 64
learningRate = 1e-4

###########################################################

# Here we'll get our information...

print('BBBBBBB')

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

    # Disp grids
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

# Truth
print('y_train: ', y_train.shape)
print('y_test', y_test.shape)

###########################################################

mirrored_strategy = tensorflow.distribute.MultiWorkerMirroredStrategy()

print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

with mirrored_strategy.scope():
   # VGG, VGG_BN, VGG_SEP, VGG_EX, VGG_RES
   ivysaurusDisp = WhoseThatPokemon(dimensions, nClasses, MODE_VGG)
   ivysaurusDisp.summary()

   # Define the optimiser and compile the model
   optimiser = Adam(learning_rate=learningRate)
   ivysaurusDisp.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

###########################################################

# Create class weights
indexVector = np.argmax(y_test, axis=1)

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

if (MODE_VGG == '0') :
   filePath = '/storage/users/mawbyi1/Ivysaurus/files/gaussian/my_model_disp_VGG'
elif (MODE_VGG == '1') :
   filePath = '/storage/users/mawbyi1/Ivysaurus/files/gaussian/my_model_disp_VGG_BN'
elif (MODE_VGG == '2') :
   filePath = '/storage/users/mawbyi1/Ivysaurus/files/gaussian/my_model_disp_VGG_SEP'
elif (MODE_VGG == '3') :
   filePath = '/storage/users/mawbyi1/Ivysaurus/files/gaussian/my_model_disp_VGG_EX'
elif (MODE_VGG == '4') :
   filePath = '/storage/users/mawbyi1/Ivysaurus/files/gaussian/my_model_disp_VGG_RES'

# checkpoint
checkpoint = ModelCheckpoint(filePath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Reduce the learning rate by a factor of ten when required
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6, verbose=1)

callbacks_list = [checkpoint, reduce_lr]
   
history = ivysaurusDisp.fit([startGridU_disp_train, endGridU_disp_train, startGridV_disp_train, endGridV_disp_train, startGridW_disp_train, endGridW_disp_train], y_train, 
    batch_size = batchSize, validation_data=([startGridU_disp_test, endGridU_disp_test, startGridV_disp_test, endGridV_disp_test, startGridW_disp_test, endGridW_disp_test], y_test), 
    shuffle=True, epochs=nEpochs, class_weight=classWeights, callbacks=callbacks_list, verbose=2)

###########################################################

# FIN!

print('DONE!')
