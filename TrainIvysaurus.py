#!/usr/bin/env python
# coding: utf-8

# In[1]:


import uproot
import matplotlib.pyplot as plt
import numpy as np
import math
import glob

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau

import sklearn 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.utils import class_weight



import IvysaurusModel
import FileHelper


# In[10]:


fileNames = glob.glob('/Users/isobel/Desktop/DUNE/Ivysaurus/files/grid24/ivysaurus_*.root')
trainVarFile = '/Users/isobel/Desktop/DUNE/Ivysaurus/files/grid24/trainVarArrays.npz'
print(fileNames)


# In[ ]:





# In[6]:


# Here we'll put some hyperparameters...

dimensions = 24
nClasses = 6
nTrackVars = 6 # nTrackChildren, nShowerChildren, nGrandChildren, trackLength, trackWobble, trackScore
         
ntrain = 300507
ntest  = 75127

nEpochs = 10
batchSize = 64
learningRate = 1e-4


# In[ ]:



    
    


# In[ ]:





# In[11]:


# Here we'll get our information...

useExistingVariableFile = False

if not (useExistingVariableFile):

    # Read tree
    eventID, startGridU, startGridV, startGridW, endGridU, endGridV, endGridW, trackVars, y = FileHelper.readTree(fileNames, dimensions, nClasses)

    print(startGridU.shape)
    print(trackVars.shape)
    
    # This should shuffle things so that the indicies are still linked
    startGridU, startGridV, startGridW, endGridU, endGridV, endGridW, trackVars, y = sklearn.utils.shuffle(startGridU, startGridV, startGridW, endGridU, endGridV, endGridW, trackVars, y)

    #eventID_train = eventID[:ntrain]
    #eventID_test = eventID[ntrain:(ntrain + ntest)]
    
    startGridU_train = startGridU[:ntrain]
    startGridV_train = startGridV[:ntrain]
    startGridW_train = startGridW[:ntrain]

    startGridU_test = startGridU[ntrain:(ntrain + ntest)]
    startGridV_test = startGridV[ntrain:(ntrain + ntest)]
    startGridW_test = startGridW[ntrain:(ntrain + ntest)]

    endGridU_train = endGridU[:ntrain]
    endGridV_train = endGridV[:ntrain]
    endGridW_train = endGridW[:ntrain]

    endGridU_test = endGridU[ntrain:(ntrain + ntest)]
    endGridV_test = endGridV[ntrain:(ntrain + ntest)]
    endGridW_test = endGridW[ntrain:(ntrain + ntest)]
    
    trackVars_train = trackVars[:ntrain]
    trackVars_test = trackVars[ntrain:(ntrain + ntest)]

    y_train = y[:ntrain]
    y_test = y[ntrain:(ntrain + ntest)]
    
    np.savez(trainVarFile, startGridU_train=startGridU_train, startGridV_train=startGridV_train, startGridW_train=startGridW_train, startGridU_test=startGridU_test, startGridV_test=startGridV_test, startGridW_test=startGridW_test, endGridU_train=endGridU_train, endGridV_train=endGridV_train, endGridW_train=endGridW_train, trackVars_train=trackVars_train, endGridU_test=endGridU_test, endGridV_test=endGridV_test, endGridW_test=endGridW_test, trackVars_test=trackVars_test, y_train=y_train, y_test=y_test)   


# In[ ]:





# In[ ]:



    


# In[ ]:





# In[ ]:





# In[5]:


if (useExistingVariableFile):
    data = np.load(trainVarFile)
    
    eventID_train = data['eventID_train']
    eventID_test = data['eventID_test']
    
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
    
    y_train = data['y_train']
    y_test = data['y_test']


# In[6]:


print('eventID_train: ', eventID_train.shape)
print('eventID_train: ', eventID_test.shape)

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

print('y_train: ', y_train.shape)
print('y_test', y_test.shape)


# In[7]:


# Work out the mean and variance

meanStartU = np.mean(startGridU_train)
meanStartV = np.mean(startGridV_train)
meanStartW = np.mean(startGridW_train)

meanEndU = np.mean(endGridU_train)
meanEndV = np.mean(endGridV_train)
meanEndW = np.mean(endGridW_train)

varStartU = np.var(startGridU_train)
varStartV = np.var(startGridV_train)
varStartW = np.var(startGridW_train)

varEndU = np.var(endGridU_train)
varEndV = np.var(endGridV_train)
varEndW = np.var(endGridW_train)

print('meanStartU: ', meanStartU)
print('meanStartV: ', meanStartV)
print('meanStartW: ', meanStartW)

print('varStartU: ', varStartU)
print('varStartV: ', varStartV)
print('varStartW: ', varStartW)

print('meanEndU: ', meanEndU)
print('meanEndV: ', meanEndV)
print('meanEndW: ', meanEndW)

print('varEndU: ', varEndU)
print('varEndV: ', varEndV)
print('varEndW: ', varEndW)


# In[8]:


ivysaurusCNN = IvysaurusModel.IvysaurusIChooseYou(dimensions, nClasses, nTrackVars, meanStartU, varStartU, meanStartV, varStartV, meanStartW, varStartW, meanEndU, varEndU, meanEndV, varEndV, meanEndW, varEndW)
#ivysaurusCNN.summary()


# In[9]:


# Define the optimiser and compile the model
optimiser = optimizers.legacy.Adam(learning_rate=learningRate)
ivysaurusCNN.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])


# In[10]:


# Create class weights

print(y_test)
indexVector = np.argmax(y_test, axis=1)

    # muons = 0, protons = 1, pions = 2, electrons = 3, photons = 4, other = 5

nMuons = np.count_nonzero(indexVector == 0)    
nProtons = np.count_nonzero(indexVector == 1)  
nPions = np.count_nonzero(indexVector == 2)  
nElectrons = np.count_nonzero(indexVector == 3)  
nPhotons = np.count_nonzero(indexVector == 4)  
nOther = np.count_nonzero(indexVector == 5)  


# Normalise to largest
maxParticle = max(nMuons, nProtons, nPions, nElectrons, nPhotons)

classWeights = {0: maxParticle/nMuons, 1: maxParticle/nProtons, 2: maxParticle/nPions, 3: maxParticle/nElectrons, 4: maxParticle/nPhotons, 5:0}

print(classWeights)


# In[11]:


# Fit that model!

# Reduce the learning rate by a factor of ten when required
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6, verbose=1)
history = ivysaurusCNN.fit([startGridU_train, endGridU_train, startGridV_train, endGridV_train, startGridW_train, endGridW_train, trackVars_train], y_train, 
    batch_size = batchSize, validation_data=([startGridU_test, endGridU_test, startGridV_test, endGridV_test, startGridW_test, endGridW_test, trackVars_test], y_test), 
    shuffle=True, epochs=nEpochs, class_weight=classWeights, callbacks=[reduce_lr]) 


# In[ ]:


# Evaluate training

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


# Use the network to predict the category of the test sample

y_pred = ivysaurusCNN.predict([startGridU_test, endGridU_test, startGridV_test, endGridV_test, startGridW_test, endGridW_test])



# In[ ]:


incorrectIndicies = []

for i in range (y_pred.shape[0]) :
    prediction = np.argmax(y_pred[i])
    truth = np.argmax(y_test[i])
    if (prediction != truth) :
        incorrectIndicies.append([i, prediction, truth])
    
print(incorrectIndicies)                


# In[ ]:


# Let's look at the confusion matrix

confMatrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

trueSums = np.sum(confMatrix, axis=1)
predSums = np.sum(confMatrix, axis=0)

print('trueSums: ', trueSums)
print('predSums: ', predSums)

trueNormalised = np.zeros(shape=(nClasses, nClasses))
predNormalised = np.zeros(shape=(nClasses, nClasses))

for trueIndex in range(nClasses) : 
    for predIndex in range(nClasses) :
        nEntries = confMatrix[trueIndex][predIndex]
        if trueSums[trueIndex] > 0 :
            trueNormalised[trueIndex][predIndex] = float(nEntries) / float(trueSums[trueIndex])
        if predSums[predIndex] > 0 :
            predNormalised[trueIndex][predIndex] = float(nEntries) / float(predSums[predIndex])

displayTrueNorm = ConfusionMatrixDisplay(confusion_matrix=trueNormalised, display_labels=["Muon", "Proton", "Pion", "Electron", "Photon", "Other"])
displayTrueNorm.plot()

displayPredNorm = ConfusionMatrixDisplay(confusion_matrix=predNormalised, display_labels=["Muon", "Proton", "Pion", "Electron", "Photon", "Other"])
displayPredNorm.plot()

print(confMatrix)


# In[ ]:


# Compute ROC curve and ROC area for each class

falsePositive = dict()
bkgRejection = dict()
truePositive = dict()
roc = dict()

for i in range(nClasses):
    falsePositive[i], truePositive[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    bkgRejection[i] = 1 - falsePositive[i]
    roc[i] = sklearn.metrics.auc(falsePositive[i], bkgRejection[i])

# Plot of a ROC curve for a specific class

rocCurveTitles = ["Muon", "Proton", "Pion", "Electron", "Photon", "Other"]

for i in range(nClasses):
    plt.figure()
    plt.plot(truePositive[i], bkgRejection[i], label='ROC curve (area = %0.2f)' % roc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xlabel('Efficiency')
    plt.ylabel('BG Rejection')
    plt.title(rocCurveTitles[i])
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


# In[ ]:


particleColors = ('b', 'g', 'k', 'r', 'tab:orange', 'tab:gray')
histTitles = ('CNN Muon Score', 'CCN Proton Score', 'CNN Pion Score', 'CNN Electron Score', 'CNN Photon Score', 'CNN Other Score')

for i in range(nClasses) :
    for j in range(nClasses) :
        nTrueParticles = trueSums[j]
        weights = np.full(nTrueParticles, 1.0/nTrueParticles)
        plt.hist(y_pred[y_test[:,j] == 1][:,i], bins=40, weights=weights, color=particleColors[j], histtype='step')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel(histTitles[i])
        plt.ylabel('Proportion of Tracks')
        plt.legend(['Muon', 'Proton', 'Pion', 'Electron', 'Photon', 'Other'])
    plt.show()


# In[ ]:




