print('111')

import numpy as np
import math

print('222')

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau

print('333')

import sklearn 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.utils import class_weight

print('444')

import IvysaurusModel
import IvysaurusModel_VGG
import IvysaurusModel_Inception

print('555')

###########################################################

print('AAAAAAA')

dimensions = 24
nClasses = 6

nTrackVars = 10 # nTrackChildren, nShowerChildren, nGrandChildren, nChildHits, childEnergy, childTrackScore, trackLength, trackWobble, trackScore, momComparison
nShowerVars = 3 # displacement, dca, trackStubLength

ntrain = 928250
ntest  = 103139

nEpochs = 10
batchSize = 64
learningRate = 1e-4

###########################################################

# Here we'll get our information...

print('BBBBBBB')

trainVarFile = '/storage/users/mawbyi1/Ivysaurus/files/grid24/trainVarArrays.npz'
data = np.load(trainVarFile)
    
startGridU_train = data['startGridU_train']
startGridV_train = data['startGridV_train']
startGridW_train = data['startGridW_train']
    
startGridU_test = data['startGridU_test']
startGridV_test = data['startGridV_test'] 
startGridW_test = data['startGridW_test']
    
endGridU_train = data['endGridU_train']
endGridV_train = data['endGridV_train']
endGridW_train = data['endGridW_train']
    
endGridU_test = data['endGridU_test']
endGridV_test = data['endGridV_test']
endGridW_test = data['endGridW_test']
    
trackVars_train = data['trackVars_train']
trackVars_test = data['trackVars_test']

showerVars_train = data['showerVars_train']
showerVars_test = data['showerVars_test']
    
y_train = data['y_train']
y_test = data['y_test']

print('CCCCCC')

###########################################################

print('startGridU_train: ', startGridU_train.shape)
print('startGridV_train: ', startGridV_train.shape)
print('startGridW_train: ', startGridW_train.shape)
print('startGridU_test: ', startGridU_test.shape)
print('startGridV_test: ', startGridV_test.shape)
print('startGridW_test: ', startGridW_test.shape)
   
print('endGridU_train: ', endGridU_train.shape)    
print('endGridV_train: ', endGridV_train.shape)
print('endGridW_train: ', endGridW_train.shape)
print('endGridU_test: ', endGridU_test.shape)     
print('endGridV_test: ', endGridV_test.shape)     
print('endGridW_test: ', endGridW_test.shape) 
    
print('trackVars_train: ', trackVars_train.shape)    
print('trackVars_test: ', trackVars_test.shape)

print('showerVars_train: ', showerVars_train.shape)    
print('showerVars_test: ', showerVars_test.shape)

print('y_train: ', y_train.shape)
print('y_test', y_test.shape)

###########################################################

# Set the most deviating graph to zero?
'''
nU_train = (startGridU_train > 0.00000000000001).sum(axis=1).sum(axis=1)
nV_train = (startGridV_train > 0.00000000000001).sum(axis=1).sum(axis=1)
nW_train = (startGridW_train > 0.00000000000001).sum(axis=1).sum(axis=1)

nU_test = (startGridU_test > 0.00000000000001).sum(axis=1).sum(axis=1)
nV_test = (startGridV_test > 0.00000000000001).sum(axis=1).sum(axis=1)
nW_test = (startGridW_test > 0.00000000000001).sum(axis=1).sum(axis=1)

deltaUV_train = np.fabs(nU_train - nV_train)
deltaUW_train = np.fabs(nU_train - nW_train)
deltaVU_train = np.fabs(nU_train - nV_train)
deltaVW_train = np.fabs(nV_train - nW_train)
deltaWU_train = np.fabs(nU_train - nW_train)
deltaWV_train = np.fabs(nV_train - nW_train)

deltaUV_test = np.fabs(nU_test - nV_test)
deltaUW_test = np.fabs(nU_test - nW_test)
deltaVU_test = np.fabs(nU_test - nV_test)
deltaVW_test = np.fabs(nV_test - nW_test)
deltaWU_test = np.fabs(nU_test - nW_test)
deltaWV_test = np.fabs(nV_test - nW_test)

deltaU_train = deltaUV_train + deltaUW_train
deltaV_train = deltaVU_train + deltaVW_train
deltaW_train = deltaWU_train + deltaWV_train

deltaU_test = deltaUV_test + deltaUW_test
deltaV_test = deltaVU_test + deltaVW_test
deltaW_test = deltaWU_test + deltaWV_test

brokenUIndex_train = (deltaU_train > deltaV_train) & (deltaU_train > deltaW_train)
brokenVIndex_train = (deltaV_train > deltaU_train) & (deltaV_train > deltaW_train)
brokenWIndex_train = (deltaW_train > deltaU_train) & (deltaW_train > deltaV_train)
ambiguousIndex_train = (brokenUIndex_train == False) & (brokenVIndex_train == False) & (brokenWIndex_train == False) 

brokenUIndex_test = (deltaU_test > deltaV_test) & (deltaU_test > deltaW_test)
brokenVIndex_test = (deltaV_test > deltaU_test) & (deltaV_test > deltaW_test)
brokenWIndex_test = (deltaW_test > deltaU_test) & (deltaW_test > deltaV_test)
ambiguousIndex_test = (brokenUIndex_test == False) & (brokenVIndex_test == False) & (brokenWIndex_test == False) 

startGridU_train[np.where(ambiguousIndex_train)[:][:]] = 0
startGridU_train[np.where(brokenUIndex_train)[:][:]] = 0
startGridV_train[np.where(brokenVIndex_train)[:][:]] = 0
startGridW_train[np.where(brokenWIndex_train)[:][:]] = 0
endGridU_train[np.where(ambiguousIndex_train)[:][:]] = 0
endGridU_train[np.where(brokenUIndex_train)[:][:]] = 0
endGridV_train[np.where(brokenVIndex_train)[:][:]] = 0
endGridW_train[np.where(brokenWIndex_train)[:][:]] = 0

startGridU_test[np.where(ambiguousIndex_test)[:][:]] = 0
startGridU_test[np.where(brokenUIndex_test)[:][:]] = 0
startGridV_test[np.where(brokenVIndex_test)[:][:]] = 0
startGridW_test[np.where(brokenWIndex_test)[:][:]] = 0
endGridU_test[np.where(ambiguousIndex_test)[:][:]] = 0
endGridU_test[np.where(brokenUIndex_test)[:][:]] = 0
endGridV_test[np.where(brokenVIndex_test)[:][:]] = 0
endGridW_test[np.where(brokenWIndex_test)[:][:]] = 0
'''
###########################################################

# Normalise start/end grids

energyLimit = 0.01

startGridU_train[startGridU_train > energyLimit] = energyLimit
startGridU_train = startGridU_train / energyLimit

startGridV_train[startGridV_train > energyLimit] = energyLimit
startGridV_train = startGridV_train / energyLimit

startGridW_train[startGridW_train > energyLimit] = energyLimit
startGridW_train = startGridW_train / energyLimit

endGridU_train[endGridU_train > energyLimit] = energyLimit
endGridU_train = endGridU_train / energyLimit

endGridV_train[endGridV_train > energyLimit] = energyLimit
endGridV_train = endGridV_train / energyLimit

endGridW_train[endGridW_train > energyLimit] = energyLimit
endGridW_train = endGridW_train / energyLimit

startGridU_test[startGridU_test > energyLimit] = energyLimit
startGridU_test = startGridU_test / energyLimit

startGridV_test[startGridV_test > energyLimit] = energyLimit
startGridV_test = startGridV_test / energyLimit

startGridW_test[startGridW_test > energyLimit] = energyLimit
startGridW_test = startGridW_test / energyLimit

endGridU_test[endGridU_test > energyLimit] = energyLimit
endGridU_test = endGridU_test / energyLimit

endGridV_test[endGridV_test > energyLimit] = energyLimit
endGridV_test = endGridV_test / energyLimit

endGridW_test[endGridW_test > energyLimit] = energyLimit
endGridW_test = endGridW_test / energyLimit

###########################################################

ivysaurusCNN = IvysaurusModel_VGG.IvysaurusIChooseYou(dimensions, nClasses, nTrackVars, nShowerVars)
ivysaurusCNN.summary()

###########################################################

# Define the optimiser and compile the model
optimiser = optimizers.Adam(learning_rate=learningRate)
ivysaurusCNN.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])

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

# Reduce the learning rate by a factor of ten when required
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6, verbose=1)
history = ivysaurusCNN.fit([startGridU_train, endGridU_train, startGridV_train, endGridV_train, startGridW_train, endGridW_train, trackVars_train, showerVars_train], y_train, 
    batch_size = batchSize, validation_data=([startGridU_test, endGridU_test, startGridV_test, endGridV_test, startGridW_test, endGridW_test, trackVars_test, showerVars_test], y_test), 
    shuffle=True, epochs=nEpochs, class_weight=classWeights, callbacks=[reduce_lr], verbose=2) 

###########################################################

# Save the model

print('Saving model...')
'''
model_json = ivysaurusCNN.to_json()

with open("/home/hpc/30/mawbyi1/Ivysaurus/files/grid24/model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
ivysaurusCNN.save_weights("/home/hpc/30/mawbyi1/Ivysaurus/files/grid24/model.h5")
'''

#ivysaurusCNN.save('/home/hpc/30/mawbyi1/Ivysaurus/files/grid24/my_model')

ivysaurusCNN.save('/home/hpc/30/mawbyi1/Ivysaurus/files/grid24/my_model_ShowerVars_VGG')


