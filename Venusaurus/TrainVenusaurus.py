print('111')

import numpy as np
import math
import glob
import sys

print('222')

import tensorflow
from tensorflow import float32 as tffloat32
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import newaxis, cast

print('333')

from sklearn.utils import class_weight

print('444')

import VenusaurusTransformer
import VenusaurusFileHelper

print('555')

###########################################################

print('666')

physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0 :
    for gpu in physical_devices :
        tensorflow.config.experimental.set_memory_growth(gpu, True)

print('777')

###########################################################

# Some parameters for the training

# To create profiles - TODO (actually add them to the function call)
binWidth_l = 0.5    # Units cm - wire pitch?
targetNBins_l = 50  # This equates to 25cm in length
binWidth_t = 0.5    # Units cm - wire pitch?
targetNBins_t = 20  # This equates to 10cm in length - moliere radius

# To turn profiles into integer tokens - I think l and t should be the same?
maxEnergyValue_l = 0.009
nEnergyBins_l = 100
energyBinWidth_l = float(maxEnergyValue_l) / float(nEnergyBins_l)

maxEnergyValue_t = 0.22
nEnergyBins_t = 500
energyBinWidth_t = float(maxEnergyValue_t) / float(nEnergyBins_t)

# Transformer parameters
nVocab_l = nEnergyBins_l + 2 # Number of bins for energy depositions (transformer expects an integer tokens)
nVocab_t = nEnergyBins_t + 2 # Number of bins for energy depositions (transformer expects an integer tokens)
embedDim_l = 32              # Position embedding dimensions
embedDim_t = 50              # Position embedding dimensions
sequenceLength_l = targetNBins_l * int(2)
sequenceLength_t = targetNBins_t

nClasses = 5        # Number of types for classification

nEpochs = 20         # Number of epochs to train for
batchSize = 64     # Batch size
learningRate = 1e-4 # Initial learning rate

print('888')

###########################################################

# Here we'll get our information...

# Profiles
longProfiles_start_train = np.empty((0, targetNBins_l, 1))
longProfiles_end_train = np.empty((0, targetNBins_l, 1))
transProfiles_train = np.empty((0, targetNBins_t, 1))

longProfiles_start_test = np.empty((0, targetNBins_l, 1))
longProfiles_end_test = np.empty((0, targetNBins_l, 1))
transProfiles_test = np.empty((0, targetNBins_t, 1))

# Truth
y_train = np.empty((0, nClasses))
y_test = np.empty((0, nClasses))

# Get training file(s)
trainFileNames = glob.glob('/Users/isobel/Desktop/DUNE/Ivysaurus/Venusaurus/files/*/*.npz')
print(trainFileNames)

for trainFileName in trainFileNames :
    print('Reading file: ', str(trainFileName),', This may take a while...')
    
    data = np.load(trainFileName)

    # Profiles
    longProfiles_start_train =  np.concatenate((longProfiles_start_train, data['longProfiles_start_train']), axis=0)
    longProfiles_end_train = np.concatenate((longProfiles_end_train, data['longProfiles_end_train']), axis=0)
    transProfiles_train = np.concatenate((transProfiles_train, data['transProfiles_train']), axis=0)
                           
    longProfiles_start_test =  np.concatenate((longProfiles_start_test, data['longProfiles_start_test']), axis=0)
    longProfiles_end_test = np.concatenate((longProfiles_end_test, data['longProfiles_end_test']), axis=0)
    transProfiles_test = np.concatenate((transProfiles_test, data['transProfiles_test']), axis=0)

    # Truth
    y_train = np.concatenate((y_train, data['y_train']), axis=0)
    y_test = np.concatenate((y_test, data['y_test']), axis=0)

    
print('999')

###########################################################

# check everything went smoothly
print('longProfiles_start_train: ', longProfiles_start_train.shape)
print('longProfiles_end_train: ', longProfiles_end_train.shape)
print('transProfiles_train: ', transProfiles_train.shape)

print('longProfiles_start_test: ', longProfiles_start_test.shape)
print('longProfiles_end_test: ', longProfiles_end_test.shape)
print('transProfiles_test: ', transProfiles_test.shape)

# Truth
print('y_train: ', y_train.shape)
print('y_test', y_test.shape)

###########################################################

# To turn profiles into integer tokens

zeroComparison = 0.00001 

# longProfiles_start_train
ls_train_mask_above = longProfiles_start_train > maxEnergyValue_l        # Mark those that surpase upper limit
ls_train_mask_zero = longProfiles_start_train < zeroComparison           # Mark the padded or 'no value' tokens
longProfiles_start_train = np.floor(longProfiles_start_train / energyBinWidth_l).astype('int64')
longProfiles_start_train[ls_train_mask_above] = int(-1)                  # 1 typically marks OOB tokens
longProfiles_start_train[ls_train_mask_zero] = int(-2)                   # 0 typically marks 'no value' tokens
longProfiles_start_train = longProfiles_start_train + 2
longProfiles_start_train[longProfiles_start_train > nVocab_l] = nVocab_l # Just to make sure (.f precision)

# longProfiles_end_train
le_train_mask_above = longProfiles_end_train > maxEnergyValue_l          # Mark those that surpase upper limit
le_train_mask_zero = longProfiles_end_train < zeroComparison             # Mark the padded or 'no value' tokens
longProfiles_end_train = np.floor(longProfiles_end_train / energyBinWidth_l).astype('int64')
longProfiles_end_train[le_train_mask_above] = int(-1)                    # 1 typically marks OOB tokens
longProfiles_end_train[le_train_mask_zero] = int(-2)                     # 0 typically marks 'no value' tokens
longProfiles_end_train = longProfiles_end_train + 2
longProfiles_end_train[longProfiles_end_train > nVocab_l] = nVocab_l     # Just to make sure (.f precision)

# transProfiles_train
t_train_mask_above = transProfiles_train > maxEnergyValue_t              # Mark those that surpase upper limit
t_train_mask_zero = transProfiles_train < zeroComparison                 # Mark the padded or 'no value' tokens
transProfiles_train = np.floor(transProfiles_train / energyBinWidth_t).astype('int64')
transProfiles_train[t_train_mask_above] = int(-1)                        # 1 typically marks OOB tokens
transProfiles_train[t_train_mask_zero] = int(-2)                         # 0 typically marks 'no value' tokens
transProfiles_train = transProfiles_train + 2
transProfiles_train[transProfiles_train > nVocab_t] = nVocab_t           # Just to make sure (.f precision)

# longProfiles_start_test
ls_test_mask_above = longProfiles_start_test > maxEnergyValue_l          # Mark those that surpase upper limit
ls_test_mask_zero = longProfiles_start_test < zeroComparison             # Mark the padded or 'no value' tokens
longProfiles_start_test = np.floor(longProfiles_start_test / energyBinWidth_l).astype('int64')
longProfiles_start_test[ls_test_mask_above] = int(-1)                    # 1 typically marks OOB tokens
longProfiles_start_test[ls_test_mask_zero] = int(-2)                     # 0 typically marks 'no value' tokens
longProfiles_start_test = longProfiles_start_test + 2
longProfiles_start_test[longProfiles_start_test > nVocab_l] = nVocab_l   # Just to make sure (.f precision)

# longProfiles_end_test
le_test_mask_above = longProfiles_end_test > maxEnergyValue_l            # Mark those that surpase upper limit
le_test_mask_zero = longProfiles_end_test < zeroComparison               # Mark the padded or 'no value' tokens
longProfiles_end_test = np.floor(longProfiles_end_test / energyBinWidth_l).astype('int64')
longProfiles_end_test[le_test_mask_above] = int(-1)                      # 1 typically marks OOB tokens
longProfiles_end_test[le_test_mask_zero] = int(-2)                       # 0 typically marks 'no value' tokens
longProfiles_end_test = longProfiles_end_test + 2
longProfiles_end_test[longProfiles_end_test > nVocab_l] = nVocab_l       # Just to make sure (.f precision)

# transProfiles_test
t_test_mask_above = transProfiles_test > maxEnergyValue_t             # Mark those that surpase upper limit
t_test_mask_zero = transProfiles_test < zeroComparison                # Mark the padded or 'no value' tokens
transProfiles_test = np.floor(transProfiles_test / energyBinWidth_t).astype('int64')
transProfiles_test[t_test_mask_above] = int(-1)                       # 1 typically marks OOB tokens
transProfiles_test[t_test_mask_zero] = int(-2)                        # 0 typically marks 'no value' tokens
transProfiles_test = transProfiles_test + 2
transProfiles_test[transProfiles_test > nVocab_l] = nVocab_l          # Just to make sure (.f precision)

print('aaa')

###########################################################

# Combine start and end profiles

longProfiles_train = np.concatenate((longProfiles_start_train, longProfiles_end_train), axis=1)
longProfiles_test = np.concatenate((longProfiles_start_test, longProfiles_end_test), axis=1)

print('bbb')

###########################################################

# Check everything went smoothly

# Profiles
print('longProfiles_train: ', longProfiles_train.shape)
print('transProfiles_train: ', transProfiles_train.shape)

print('longProfiles_test: ', longProfiles_test.shape)
print('transProfiles_test: ', transProfiles_test.shape)

# Truth
print('y_train: ', y_train.shape)
print('y_test', y_test.shape)

###########################################################

# Reshape, because transformers are weird

longProfiles_train = longProfiles_train.reshape(longProfiles_train.shape[0], sequenceLength_l)
transProfiles_train = transProfiles_train.reshape(transProfiles_train.shape[0], sequenceLength_t)

longProfiles_test = longProfiles_test.reshape(longProfiles_test.shape[0], sequenceLength_l)
transProfiles_test = transProfiles_test.reshape(transProfiles_test.shape[0], sequenceLength_t)

print('ccc')

###########################################################

mirrored_strategy = tensorflow.distribute.MultiWorkerMirroredStrategy()

print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

with mirrored_strategy.scope():
    venusaurusModel = VenusaurusTransformer.TransformerModel(sequenceLength_l, nVocab_l, nClasses, embedDim_l)
    venusaurusModel.summary()

    # Define the optimiser and compile the model
    optimiser = Adam(learning_rate=learningRate)
    venusaurusModel.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

print('ddd')

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

#classWeights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}


print('Class Weights: ')
print(classWeights)

###########################################################

# Fit that model!

# checkpoint
checkpoint = ModelCheckpoint('/Users/isobel/Desktop/DUNE/Ivysaurus/Venusaurus/files/test', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Reduce the learning rate by a factor of ten when required
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6, verbose=1)

callbacks_list = [checkpoint, reduce_lr]

history = venusaurusModel.fit(longProfiles_train, y_train, 
    batch_size = batchSize, validation_data=(longProfiles_test, y_test), 
    shuffle=True, epochs=nEpochs, class_weight=classWeights, callbacks=callbacks_list, verbose=2)

###########################################################

# FIN!

print('DONE!')


















