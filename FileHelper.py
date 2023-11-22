import numpy as np
import uproot

from tensorflow.keras.utils import to_categorical

def readTreeOLD(branches, dimensions, nClasses, nEntries):

    print('Reading tree... this may take a while...')
    
    # This is dumb... I have to do this because of empty entries in the vectors - SAD

    print()
    
    startIndex = -1

    for i in range(0, len(branches)) :
        
        #print("A: Index: ", str(i))
        
        if (len(branches['StartGridU'][i]) == 0)|(len(branches['StartGridV'][i]) == 0)|(len(branches['StartGridW'][i]) == 0) :
            continue
            
        if (len(branches['TruePDG'][i]) == 0):
            continue
        
        startIndex = i
        break
        
    print("Start Index: ", str(startIndex))
    
    startGridU = np.asarray(branches['StartGridU'][startIndex]).reshape(1, dimensions, dimensions, 1)
    startGridV = np.asarray(branches['StartGridV'][startIndex]).reshape(1, dimensions, dimensions, 1)
    startGridW = np.asarray(branches['StartGridW'][startIndex]).reshape(1, dimensions, dimensions, 1)
    endGridU = np.asarray(branches['EndGridU'][startIndex]).reshape(1, dimensions, dimensions, 1)
    endGridV = np.asarray(branches['EndGridV'][startIndex]).reshape(1, dimensions, dimensions, 1)
    endGridW = np.asarray(branches['EndGridW'][startIndex]).reshape(1, dimensions, dimensions, 1)
    particlePDG = np.asarray(branches['TruePDG'][startIndex]).reshape(1, 1)

    count = 1
    
    for i in range((startIndex + 1), len(branches)) :
        
        #print("B: Index: ", str(i))
        
        #print('StartGridU: ', str(len(branches['StartGridU'][i])))
        #print('StartGridV: ', str(len(branches['StartGridV'][i])))
        #print('StartGridW: ', str(len(branches['StartGridW'][i])))
        #print('TruePDG: ', str(len(branches['TruePDG'][i])))
        #print('---------------------------')
        
        if (len(branches['StartGridU'][i]) == 0)|(len(branches['StartGridV'][i]) == 0)|(len(branches['StartGridW'][i]) == 0) :
            continue
            
        if (len(branches['TruePDG'][i]) == 0):
            continue
            
        # First, completeness and nHit
        completeness = branches['Completeness'][i]
        purity = branches['Purity'][i]

        if (completeness < 0.5) | (purity < 0.5) :
            continue
    
        thisStartGridU = np.asarray(branches['StartGridU'][i]).reshape(1, dimensions, dimensions, 1)
        thisStartGridV = np.asarray(branches['StartGridV'][i]).reshape(1, dimensions, dimensions, 1)
        thisStartGridW = np.asarray(branches['StartGridW'][i]).reshape(1, dimensions, dimensions, 1)
        
        startGridU = np.concatenate((startGridU, thisStartGridU), 0)
        startGridV = np.concatenate((startGridV, thisStartGridV), 0)
        startGridW = np.concatenate((startGridW, thisStartGridW), 0)
    
        thisEndGridU = np.asarray(branches['EndGridU'][i]).reshape(1, dimensions, dimensions, 1)
        thisEndGridV = np.asarray(branches['EndGridV'][i]).reshape(1, dimensions, dimensions, 1)
        thisEndGridW = np.asarray(branches['EndGridW'][i]).reshape(1, dimensions, dimensions, 1)
        
        endGridU = np.concatenate((endGridU, thisEndGridU), 0)
        endGridV = np.concatenate((endGridV, thisEndGridV), 0)
        endGridW = np.concatenate((endGridW, thisEndGridW), 0)
        
        thisParticlePDG = np.asarray(branches['TruePDG'][i]).reshape(1, 1)
        particlePDG = np.concatenate((particlePDG, thisParticlePDG))
        
        count += 1
        
        #print('Count: ', str(count))
        
        if (count >= nEntries):
            break
            
    print('Read: ', str(count), ' entries from tree')


    print('We have...')
    print('nMuons: ', np.count_nonzero(abs(particlePDG) == 13))
    print('nProtons: ', np.count_nonzero(abs(particlePDG) == 2212))    
    print('nPions: ', np.count_nonzero(abs(particlePDG) == 211))     
    print('nElectrons: ', np.count_nonzero(abs(particlePDG) == 11))     
    print('nPhotons: ', np.count_nonzero(abs(particlePDG) == 22))     
    print('nOther: ', np.count_nonzero((abs(particlePDG) != 13) & (abs(particlePDG) != 2212) & (abs(particlePDG) != 221) & (abs(particlePDG) != 11) &  (abs(particlePDG) != 22)))
    # Need to alter the format & shape of particle PDG

    # muons = 0, protons = 1, pions = 2, electrons = 3, photons = 4, other = 5
    particlePDG[abs(particlePDG) == 13] = 0
    particlePDG[abs(particlePDG) == 2212] = 1
    particlePDG[abs(particlePDG) == 211] = 2
    particlePDG[abs(particlePDG) == 11] = 3
    particlePDG[abs(particlePDG) == 22] = 4
    particlePDG[(abs(particlePDG) != 0) & (abs(particlePDG) != 1) & (abs(particlePDG) != 2) & (abs(particlePDG) != 3) &  (abs(particlePDG) != 4)] = 5

    y = to_categorical(particlePDG, nClasses)

    return startGridU, startGridV, startGridW, endGridU, endGridV, endGridW, y

#################################################################################################################
#################################################################################################################

def readTree(fileNames, dimensions, nClasses) :

    # Grid lists
    startGridU = []
    startGridV = []
    startGridW = []
    
    endGridU = []
    endGridV = []
    endGridW = []
    particlePDG = []
    
    eventID = []
    
    # TrackVar lists
    trackVarsSuccessful = []
    nTrackChildren = []
    nShowerChildren = []
    nGrandChildren = []
    trackLength = []
    trackWobble = []
    trackScore = []

    
    nEntries = 0
    
    for fileName in fileNames :
        print('Reading tree: ', str(fileName),', This may take a while...')
    
        treeFile = uproot.open(fileName)
        tree = treeFile['ivyTrain/ivysaur']
        branches = tree.arrays()
        
        nEntries += len(branches)
    
        startGridU.extend(branches['StartGridU'])
        startGridV.extend(branches['StartGridV'])
        startGridW.extend(branches['StartGridW'])
        endGridU.extend(branches['EndGridU'])
        endGridV.extend(branches['EndGridV'])
        endGridW.extend(branches['EndGridW'])
        
        trackVarsSuccessful.extend(branches['TrackVarsSuccessful'])
        nTrackChildren.extend(branches['NTrackChildren'])
        nShowerChildren.extend(branches['NShowerChildren'])
        nGrandChildren.extend(branches['NGrandChildren'])
        trackLength.extend(branches['TrackLength'])
        trackWobble.extend(branches['TrackWobble'])
        trackScore.extend(branches['TrackScore'])
        
        particlePDG.extend(branches['TruePDG'])
        
        
    # Now turn things into numpy arrays
    startGridU = np.array(startGridU)
    startGridV = np.array(startGridV)    
    startGridW = np.array(startGridW)    
    endGridU = np.array(endGridU)
    endGridV = np.array(endGridV)
    endGridW = np.array(endGridW)
    
    trackVarsSuccessful = np.array(trackVarsSuccessful)
    nTrackChildren = np.array(nTrackChildren)
    nShowerChildren = np.array(nShowerChildren)
    nGrandChildren = np.array(nGrandChildren)
    trackLength = np.array(trackLength)
    trackWobble = np.array(trackWobble)
    trackScore = np.array(trackScore)
    particlePDG = np.array(particlePDG)

    
    # Reshape
    startGridU.reshape(nEntries, dimensions, dimensions, 1)
    startGridV.reshape(nEntries, dimensions, dimensions, 1)
    startGridW.reshape(nEntries, dimensions, dimensions, 1)
    endGridU.reshape(nEntries, dimensions, dimensions, 1)
    endGridV.reshape(nEntries, dimensions, dimensions, 1)
    endGridW.reshape(nEntries, dimensions, dimensions, 1)
    nTrackChildren.reshape(nEntries, 1)
    nShowerChildren.reshape(nEntries, 1)
    nGrandChildren.reshape(nEntries, 1)
    trackLength.reshape(nEntries, 1)
    trackWobble.reshape(nEntries, 1)
    trackScore.reshape(nEntries, 1)
    particlePDG.reshape(nEntries, 1)
    
    # Normalise track vars
    
    nTrackChildrenLimit = 5.0
    nShowerChildrenLimit = 3.0
    nGrandChildrenLimit = 3.0
    trackLengthLimit = 500.0
    wobbleLimit = 15.0

    nTrackChildren[nTrackChildren > nTrackChildrenLimit] = nTrackChildrenLimit
    nTrackChildren = nTrackChildren / nTrackChildrenLimit
    
    nShowerChildren[nShowerChildren > nShowerChildrenLimit] = nShowerChildrenLimit
    nShowerChildren = nShowerChildren / nShowerChildrenLimit
    
    nGrandChildren[nGrandChildren > nGrandChildrenLimit] = nGrandChildrenLimit
    nGrandChildren = nGrandChildren / nGrandChildrenLimit
    
    trackLength[trackLength > trackLengthLimit] = trackLengthLimit
    trackLength = trackLength / trackLengthLimit
    
    trackWobble[trackWobble > wobbleLimit] = wobbleLimit
    trackWobble = trackWobble / wobbleLimit  
    
    trackVars = np.concatenate((nTrackChildren, nShowerChildren, nGrandChildren, trackLength, trackWobble, trackScore), axis=1)
    
    # Refinement of the particlePDG vector
    print('We have ', str(nEntries), ' PFParticles overall!')
    print('nMuons: ', np.count_nonzero(abs(particlePDG) == 13))
    print('nProtons: ', np.count_nonzero(abs(particlePDG) == 2212))    
    print('nPions: ', np.count_nonzero(abs(particlePDG) == 211))     
    print('nElectrons: ', np.count_nonzero(abs(particlePDG) == 11))     
    print('nPhotons: ', np.count_nonzero(abs(particlePDG) == 22))     
    print('nOther: ', np.count_nonzero((abs(particlePDG) != 13) & (abs(particlePDG) != 2212) & (abs(particlePDG) != 211) & (abs(particlePDG) != 11) &  (abs(particlePDG) != 22)))
   
    # muons = 0, protons = 1, pions = 2, electrons = 3, photons = 4, other = 5
    particlePDG[abs(particlePDG) == 13] = 0
    particlePDG[abs(particlePDG) == 2212] = 1
    particlePDG[abs(particlePDG) == 211] = 2
    particlePDG[abs(particlePDG) == 11] = 3
    particlePDG[abs(particlePDG) == 22] = 4
    particlePDG[(abs(particlePDG) != 0) & (abs(particlePDG) != 1) & (abs(particlePDG) != 2) & (abs(particlePDG) != 3) &  (abs(particlePDG) != 4)] = 5
    
    # We don't have enough 'other' events to train them...
    #mask = particlePDG < nClasses
    #startGridU = startGridU[mask]
    #startGridV = startGridV[mask]    
    #startGridW = startGridW[mask]
    
    #endGridU = endGridU[mask]
    #endGridV = endGridV[mask]
    #endGridW = endGridW[mask]
    
    #particlePDG = particlePDG[mask] 

    y = to_categorical(particlePDG, nClasses)
    
    return eventID, startGridU, startGridV, startGridW, endGridU, endGridV, endGridW, trackVars, y

#################################################################################################################
#################################################################################################################

def readTrackVars(fileNames) :
    
    # TrackVar lists
    trackVarsSuccessful = []
    nTrackChildren = []
    nShowerChildren = []
    nGrandChildren = []
    nChildHits = []
    childEnergy = []
    childTrackScore = []
    trackLength = []
    trackWobble = []
    trackScore = []
    momComparison = []
    
    particlePDG = []
    
    nEntries = 0
    
    for fileName in fileNames :
        print('Reading tree: ', str(fileName),', This may take a while...')
    
        treeFile = uproot.open(fileName)
        tree = treeFile['ivyTrain/ivysaur']
        branches = tree.arrays()
        
        nEntries += len(branches)
    
        trackVarsSuccessful.extend(branches['TrackVarsSuccessful'])
        nTrackChildren.extend(branches['NTrackChildren'])
        nShowerChildren.extend(branches['NShowerChildren'])
        nGrandChildren.extend(branches['NGrandChildren'])
        nChildHits.extend(branches['NChildHits'])
        childEnergy.extend(branches['ChildEnergy'])
        childTrackScore.extend(branches['ChildTrackScore'])
        trackLength.extend(branches['TrackLength'])
        trackWobble.extend(branches['TrackWobble'])
        trackScore.extend(branches['TrackScore'])
     
        particlePDG.extend(branches['TruePDG'])

    trackVarsSuccessful = np.array(trackVarsSuccessful)
    nTrackChildren = np.array(nTrackChildren)
    nShowerChildren = np.array(nShowerChildren)
    nGrandChildren = np.array(nGrandChildren)
    trackLength = np.array(trackLength)
    trackWobble = np.array(trackWobble)
    trackScore = np.array(trackScore)
    particlePDG = np.array(particlePDG)
    
    particlePDG[abs(particlePDG) == 13] = 0
    particlePDG[abs(particlePDG) == 2212] = 1
    particlePDG[abs(particlePDG) == 211] = 2
    particlePDG[abs(particlePDG) == 11] = 3
    particlePDG[abs(particlePDG) == 22] = 4
    particlePDG[(abs(particlePDG) != 0) & (abs(particlePDG) != 1) & (abs(particlePDG) != 2) & (abs(particlePDG) != 3) &  (abs(particlePDG) != 4)] = 5
    
    return trackVarsSuccessful, nTrackChildren, nShowerChildren, nGrandChildren, trackLength, trackWobble, trackScore, particlePDG
    
    
    
    
    
    
    
    
    
    
    
    
    


