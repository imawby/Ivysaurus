import numpy as np

from tensorflow.keras.utils import to_categorical

def readTree(branches, dimensions, nClasses):

    # This is dumb... I have to do this because of empty entries in the vectors - SAD

    startIndex = -1

    for i in range(0, len(branches)) :
        if (len(branches['StartGridU'][i]) == 0)|(len(branches['StartGridV'][i]) == 0)|(len(branches['StartGridW'][i]) == 0) :
            continue
        
        startIndex = i
        break
    
    startGridU = np.asarray(branches['StartGridU'][startIndex]).reshape(1, dimensions, dimensions, 1)
    startGridV = np.asarray(branches['StartGridV'][startIndex]).reshape(1, dimensions, dimensions, 1)
    startGridW = np.asarray(branches['StartGridW'][startIndex]).reshape(1, dimensions, dimensions, 1)
    endGridU = np.asarray(branches['EndGridU'][startIndex]).reshape(1, dimensions, dimensions, 1)
    endGridV = np.asarray(branches['EndGridV'][startIndex]).reshape(1, dimensions, dimensions, 1)
    endGridW = np.asarray(branches['EndGridW'][startIndex]).reshape(1, dimensions, dimensions, 1)
    particlePDG = np.asarray(branches['TruePDG'][startIndex]).reshape(1, 1)

    for i in range((startIndex + 1), len(branches)) :
        if (len(branches['StartGridU'][i]) == 0)|(len(branches['StartGridV'][i]) == 0)|(len(branches['StartGridW'][i]) == 0) :
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


    # Need to alter the format & shape of particle PDG

    # muons = 0, protons = 1, pions = 2, electrons = 3, photons = 4, other = 5
    particlePDG[abs(particlePDG) == 13] = 0
    particlePDG[abs(particlePDG) == 2212] = 1
    particlePDG[abs(particlePDG) == 211] = 2
    particlePDG[abs(particlePDG) == 11] = 3
    particlePDG[abs(particlePDG) == 22] = 4
    particlePDG[(abs(particlePDG) != 0) & (abs(particlePDG) != 1) & (abs(particlePDG) != 2) & (abs(particlePDG) != 3) &  (abs(particlePDG) != 4)] = 5

    y = to_categorical(particlePDG, nClasses)

    return startGridU, startGridV, startGridW, endGridU, endGridV, endGridW, particlePDG