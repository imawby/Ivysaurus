import numpy as np
import uproot

from tensorflow.keras.utils import to_categorical


def readTree(fileNames, dimensions) :

    ###################################
    # Grid lists
    ###################################
    startGridU = []
    startGridV = []
    startGridW = []
    
    endGridU = []
    endGridV = []
    endGridW = []
    particlePDG = []

    ###################################
    # PFPVar lists
    ###################################
    nHits2D = []
    endX = []
    endY = []
    endZ = []
    
    ###################################
    # TrackVar lists
    ###################################
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
    
    ###################################
    # ShowerVar lists  
    ###################################
    displacement = []
    dca = []
    trackStubLength = []
    nuVertexAvSeparation = []
    nuVertexChargeAsymmetry = []

    nEntries = 0
    
    for fileName in fileNames :
        print('Reading tree: ', str(fileName),', This may take a while...')
    
        treeFile = uproot.open(fileName)
        tree = treeFile['ivysaur']
        branches = tree.arrays()
        
        nEntries += len(branches)
    
        startGridU.extend(branches['StartGridU'])
        startGridV.extend(branches['StartGridV'])
        startGridW.extend(branches['StartGridW'])
        endGridU.extend(branches['EndGridU'])
        endGridV.extend(branches['EndGridV'])
        endGridW.extend(branches['EndGridW'])

        nHits2D.extend(branches['NHits2D'])
        endX.extend(branches['EndX'])
        endY.extend(branches['EndY'])
        endZ.extend(branches['EndZ'])
        
        nTrackChildren.extend(branches['NTrackChildren'])
        nShowerChildren.extend(branches['NShowerChildren'])
        nGrandChildren.extend(branches['NGrandChildren'])
        nChildHits.extend(branches['NChildHits'])
        childEnergy.extend(branches['ChildEnergy'])
        childTrackScore.extend(branches['ChildTrackScore'])
        trackLength.extend(branches['TrackLength'])
        trackWobble.extend(branches['TrackWobble'])
        trackScore.extend(branches['TrackScore'])
        momComparison.extend(branches['TrackMomComparison'])
        
        displacement.extend(branches['ShowerDisplacement'])
        dca.extend(branches['ShowerDCA'])
        trackStubLength.extend(branches['ShowerTrackStubLength'])
        nuVertexAvSeparation.extend(branches['ShowerNuVertexAvSeparation'])
        nuVertexChargeAsymmetry.extend(branches['ShowerNuVertexChargeAsymmetry'])
        
        particlePDG.extend(branches['TruePDG'])
        
        
    ###################################
    # Now turn things into numpy arrays
    ###################################
    startGridU = np.array(startGridU)
    startGridV = np.array(startGridV)    
    startGridW = np.array(startGridW)    
    endGridU = np.array(endGridU)
    endGridV = np.array(endGridV)
    endGridW = np.array(endGridW)
    nHits2D = np.array(nHits2D)
    endX = np.array(endX)
    endY = np.array(endY)
    endZ = np.array(endZ)
    nTrackChildren = np.array(nTrackChildren)
    nShowerChildren = np.array(nShowerChildren)
    nGrandChildren = np.array(nGrandChildren)
    nChildHits = np.array(nChildHits)
    childEnergy = np.array(childEnergy)
    childTrackScore = np.array(childTrackScore)
    trackLength = np.array(trackLength)
    trackWobble = np.array(trackWobble)
    trackScore = np.array(trackScore)
    momComparison = np.array(momComparison)
    displacement = np.array(displacement)
    dca = np.array(dca)
    trackStubLength = np.array(trackStubLength)
    nuVertexAvSeparation = np.array(nuVertexAvSeparation)
    nuVertexChargeAsymmetry = np.array(nuVertexChargeAsymmetry)
    particlePDG = np.array(particlePDG)

    ###################################
    # Handle grids
    ###################################
    # work out validity
    startGridU_valid = startGridU > 0.0
    startGridV_valid = startGridV > 0.0
    startGridW_valid = startGridW > 0.0
    endGridU_valid = endGridU > 0.0
    endGridV_valid = endGridV > 0.0
    endGridW_valid = endGridW > 0.0

    # log them
    startGridU[startGridU_valid] = np.log1p(startGridU[startGridU_valid])
    startGridV[startGridV_valid] = np.log1p(startGridV[startGridV_valid])
    startGridW[startGridW_valid] = np.log1p(startGridW[startGridW_valid])
    endGridU[endGridU_valid] = np.log1p(endGridU[endGridU_valid])
    endGridV[endGridV_valid] = np.log1p(endGridV[endGridV_valid])
    endGridW[endGridW_valid] = np.log1p(endGridW[endGridW_valid])
    
    # print('--------------------------------------------------')
    # print(f'startGridU mean: {round(float(np.mean(startGridU[startGridU_valid])), 4)}')
    # print(f'startGridU std: {round(float(np.std(startGridU[startGridU_valid])), 4)}')    
    # print('--------------------------------------------------')
    # print(f'startGridV mean: {round(float(np.mean(startGridV[startGridV_valid])), 4)}')
    # print(f'startGridV std: {round(float(np.std(startGridV[startGridV_valid])), 4)}')    
    # print('--------------------------------------------------')
    # print(f'startGridW mean: {round(float(np.mean(startGridW[startGridW_valid])), 4)}')
    # print(f'startGridW std: {round(float(np.std(startGridW[startGridW_valid])), 4)}')    
    # print('--------------------------------------------------')
    # print(f'endGridU mean: {round(float(np.mean(endGridU[endGridU_valid])), 4)}')
    # print(f'endGridU std: {round(float(np.std(endGridU[endGridU_valid])), 4)}')    
    # print('--------------------------------------------------')
    # print(f'endGridV mean: {round(float(np.mean(endGridV[endGridV_valid])), 4)}')
    # print(f'endGridV std: {round(float(np.std(endGridV[endGridV_valid])), 4)}')    
    # print('--------------------------------------------------')
    # print(f'endGridW mean: {round(float(np.mean(endGridW[endGridW_valid])), 4)}')
    # print(f'endGridW std: {round(float(np.std(endGridW[endGridW_valid])), 4)}')    
    # print('--------------------------------------------------')    

    # normalise them
    startGridU_mean = 0.0768
    startGridU_std = 0.0955
    startGridV_mean = 0.0772
    startGridV_std = 0.0964
    startGridW_mean = 0.089
    startGridW_std = 0.1087
    endGridU_mean = 0.0703
    endGridU_std = 0.0935
    endGridV_mean = 0.0714
    endGridV_std = 0.0949
    endGridW_mean = 0.0809
    endGridW_std = 0.1084

    startGridU[startGridU_valid] = (startGridU[startGridU_valid] - startGridU_mean) / startGridU_std
    startGridV[startGridV_valid] = (startGridV[startGridV_valid] - startGridV_mean) / startGridV_std
    startGridW[startGridW_valid] = (startGridW[startGridW_valid] - startGridW_mean) / startGridW_std
    endGridU[endGridU_valid] = (endGridU[endGridU_valid] - endGridU_mean) / endGridU_std
    endGridV[endGridV_valid] = (endGridV[endGridV_valid] - endGridV_mean) / endGridV_std
    endGridW[endGridW_valid] = (endGridW[endGridW_valid] - endGridW_mean) / endGridW_std
    
    ###################################
    # PFP vars
    ###################################
    # print('--------------------------------------------------')
    # print(f'nHits2D mean: {round(float(np.mean(nHits2D)), 4)}')
    # print(f'nHits2D std: {round(float(np.std(nHits2D)), 4)}')
    # print('--------------------------------------------------')

    nHits2D_mean = 818.53
    nHits2D_std = 1406.8101
    nHits2D = (nHits2D - nHits2D_mean) / nHits2D_std
    
    ###################################
    # Track vars (invalid = -1)
    ###################################
    nTrackChildren_valid = nTrackChildren > -0.5
    nShowerChildren_valid = nShowerChildren > -0.5
    nGrandChildren_valid = nGrandChildren > -0.5
    nChildHits_valid = nChildHits > -0.5
    childEnergy_valid = childEnergy > -0.5
    childTrackScore_valid = childTrackScore > -0.5
    trackLength_valid = trackLength > -0.5
    trackWobble_valid = trackWobble > -0.5
    trackScore_valid = trackScore > -0.5
    momComparison_valid = momComparison > -0.5
    
    # print('--------------------------------------------------')
    # print(f'nTrackChildren mean: {round(float(np.mean(nTrackChildren[nTrackChildren_valid])), 4)}')
    # print(f'nTrackChildren std: {round(float(np.std(nTrackChildren[nTrackChildren_valid])), 4)}')
    # print('--------------------------------------------------')
    # print(f'nShowerChildren mean: {round(float(np.mean(nShowerChildren[nShowerChildren_valid])), 4)}')
    # print(f'nShowerChildren std: {round(float(np.std(nShowerChildren[nShowerChildren_valid])), 4)}')
    # print('--------------------------------------------------')
    # print(f'nGrandChildren mean: {round(float(np.mean(nGrandChildren[nGrandChildren_valid])), 4)}')
    # print(f'nGrandChildren std: {round(float(np.std(nGrandChildren[nGrandChildren_valid])), 4)}')
    # print('--------------------------------------------------')
    # print(f'nChildHits mean: {round(float(np.mean(nChildHits[nChildHits_valid])), 4)}')
    # print(f'nChildHits std: {round(float(np.std(nChildHits[nChildHits_valid])), 4)}')
    # print('--------------------------------------------------')
    # print(f'childEnergy mean: {round(float(np.mean(childEnergy[childEnergy_valid])), 4)}')
    # print(f'childEnergy std: {round(float(np.std(childEnergy[childEnergy_valid])), 4)}')
    # print('--------------------------------------------------')
    # print(f'childTrackScore mean: {round(float(np.mean(childTrackScore[childTrackScore_valid])), 4)}')
    # print(f'childTrackScore std: {round(float(np.std(childTrackScore[childTrackScore_valid])), 4)}')
    # print('--------------------------------------------------')
    # print(f'trackLength mean: {round(float(np.mean(trackLength[trackLength_valid])), 4)}')
    # print(f'trackLength std: {round(float(np.std(trackLength[trackLength_valid])), 4)}')
    # print('--------------------------------------------------')
    # print(f'trackWobble mean: {round(float(np.mean(trackWobble[trackWobble_valid])), 4)}')
    # print(f'trackWobble std: {round(float(np.std(trackWobble[trackWobble_valid])), 4)}')
    # print('--------------------------------------------------')
    # print(f'trackScore mean: {round(float(np.mean(trackScore[trackScore_valid])), 4)}')
    # print(f'trackScore std: {round(float(np.std(trackScore[trackScore_valid])), 4)}')
    # print('--------------------------------------------------')
    # print(f'momComparison mean: {round(float(np.mean(momComparison[momComparison_valid])), 4)}')
    # print(f'momComparison std: {round(float(np.std(momComparison[momComparison_valid])), 4)}')
    # print('--------------------------------------------------')

    nTrackChildren_mean = 0.1309
    nTrackChildren_std = 0.3938
    nShowerChildren_mean = 0.0466
    nShowerChildren_std = 0.2163
    nGrandChildren_mean = 0.023
    nGrandChildren_std = 0.1787
    nChildHits_mean = 205.8004
    nChildHits_std = 438.8792
    childEnergy_mean = 0.32
    childEnergy_std = 0.3568
    childTrackScore_mean = 0.7307
    childTrackScore_std = 0.1003
    trackLength_mean = 258.3441
    trackLength_std = 1426.0234
    trackWobble_mean = 6.7975
    trackWobble_std = 4.4681
    trackScore_mean = 0.5998
    trackScore_std = 0.1783
    momComparison_mean = 3.8436
    momComparison_std = 3.1121
 
    nTrackChildren[nTrackChildren_valid] = (nTrackChildren[nTrackChildren_valid] - nTrackChildren_mean) / nTrackChildren_std
    nShowerChildren[nShowerChildren_valid] = (nShowerChildren[nShowerChildren_valid] - nShowerChildren_mean) / nShowerChildren_std
    nGrandChildren[nGrandChildren_valid] = (nGrandChildren[nGrandChildren_valid] - nGrandChildren_mean) / nGrandChildren_std
    nChildHits[nChildHits_valid] = (nChildHits[nChildHits_valid] - nChildHits_mean) / nChildHits_std
    childEnergy[childEnergy_valid] = (childEnergy[childEnergy_valid] - childEnergy_mean) / childEnergy_std
    childTrackScore[childTrackScore_valid] = (childTrackScore[childTrackScore_valid] - childTrackScore_mean) / childTrackScore_std
    trackLength[trackLength_valid] = (trackLength[trackLength_valid] - trackLength_mean) / trackLength_std
    trackWobble[trackWobble_valid] = (trackWobble[trackWobble_valid] - trackWobble_mean) / trackWobble_std
    trackScore[trackScore_valid] = (trackScore[trackScore_valid] - trackScore_mean) / trackScore_std
    momComparison[momComparison_valid] = (momComparison[momComparison_valid] - momComparison_mean) / momComparison_std
    
    ###################################
    # Shower vars
    ###################################
    displacement_valid = displacement > -0.5
    dca_valid = dca > -0.5
    trackStubLength_valid = trackStubLength > -0.5
    nuVertexAvSeparation_valid = nuVertexAvSeparation > -0.5
    nuVertexChargeAsymmetry_valid = nuVertexChargeAsymmetry > -0.5
    
    # print(f'displacement mean: {round(float(np.mean(displacement[displacement_valid])), 4)}')
    # print(f'displacement std: {round(float(np.std(displacement[displacement_valid])), 4)}')
    # print('--------------------------------------------------')
    # print(f'dca mean: {round(float(np.mean(dca[dca_valid])), 4)}')
    # print(f'dca std: {round(float(np.std(dca[dca_valid])), 4)}')
    # print('--------------------------------------------------')
    # print(f'trackStubLength mean: {round(float(np.mean(trackStubLength[trackStubLength_valid])), 4)}')
    # print(f'trackStubLength std: {round(float(np.std(trackStubLength[trackStubLength_valid])), 4)}')
    # print('--------------------------------------------------')
    # print(f'nuVertexAvSeparation mean: {round(float(np.mean(nuVertexAvSeparation[nuVertexAvSeparation_valid])), 4)}')
    # print(f'nuVertexAvSeparation std: {round(float(np.std(nuVertexAvSeparation[nuVertexAvSeparation_valid])), 4)}')
    # print('--------------------------------------------------')
    # print(f'nuVertexChargeAsymmetry mean: {round(float(np.mean(nuVertexChargeAsymmetry[nuVertexChargeAsymmetry_valid])), 4)}')
    # print(f'nuVertexChargeAsymmetry std: {round(float(np.std(nuVertexChargeAsymmetry[nuVertexChargeAsymmetry_valid])), 4)}')
    # print('--------------------------------------------------')

    displacement_mean = 35.9138
    displacement_std = 78.6049
    dca_mean = 19.5538
    dca_std = 53.3346
    trackStubLength_mean = 47.3008
    trackStubLength_std = 63.3723
    nuVertexAvSeparation_mean = 19.8212
    nuVertexAvSeparation_std = 20.6977
    nuVertexChargeAsymmetry_mean = 0.8874
    nuVertexChargeAsymmetry_std = 0.1908

    displacement[displacement_valid] = (displacement[displacement_valid] - displacement_mean) / displacement_std
    dca[dca_valid] = (dca[dca_valid] - dca_mean) / dca_std
    trackStubLength[trackStubLength_valid] = (trackStubLength[trackStubLength_valid] - trackStubLength_mean) / trackStubLength_std
    nuVertexAvSeparation[nuVertexAvSeparation_valid] = (nuVertexAvSeparation[nuVertexAvSeparation_valid] - nuVertexAvSeparation_mean) / nuVertexAvSeparation_std
    nuVertexChargeAsymmetry[nuVertexChargeAsymmetry_valid] = (nuVertexChargeAsymmetry[nuVertexChargeAsymmetry_valid] - nuVertexChargeAsymmetry_mean) / nuVertexChargeAsymmetry_std

    ###################################
    # Turn valid to floats
    ###################################
    startGridU_valid = startGridU_valid.astype(np.float32)
    startGridV_valid = startGridV_valid.astype(np.float32)
    startGridW_valid = startGridW_valid.astype(np.float32)
    endGridU_valid = endGridU_valid.astype(np.float32)
    endGridV_valid = endGridV_valid.astype(np.float32)
    endGridW_valid = endGridW_valid.astype(np.float32)

    nTrackChildren_valid = nTrackChildren_valid.astype(np.float32)
    nShowerChildren_valid = nShowerChildren_valid.astype(np.float32)
    nGrandChildren_valid = nGrandChildren_valid.astype(np.float32)
    nChildHits_valid = nChildHits_valid.astype(np.float32)
    childEnergy_valid = childEnergy_valid.astype(np.float32)
    childTrackScore_valid = childTrackScore_valid.astype(np.float32)
    trackLength_valid = trackLength_valid.astype(np.float32)
    trackWobble_valid = trackWobble_valid.astype(np.float32)
    trackScore_valid = trackScore_valid.astype(np.float32)
    momComparison_valid = momComparison_valid.astype(np.float32)

    displacement_valid = displacement_valid.astype(np.float32)
    dca_valid = dca_valid.astype(np.float32)
    trackStubLength_valid = trackStubLength_valid.astype(np.float32)
    nuVertexAvSeparation_valid = nuVertexAvSeparation_valid.astype(np.float32)
    nuVertexChargeAsymmetry_valid = nuVertexChargeAsymmetry_valid.astype(np.float32)
    
    ###################################
    # Reshape
    ###################################
    startGridU = startGridU.reshape((nEntries, dimensions, dimensions, 1))
    startGridV = startGridV.reshape((nEntries, dimensions, dimensions, 1))
    startGridW = startGridW.reshape((nEntries, dimensions, dimensions, 1))
    endGridU = endGridU.reshape((nEntries, dimensions, dimensions, 1))
    endGridV = endGridV.reshape((nEntries, dimensions, dimensions, 1))
    endGridW = endGridW.reshape((nEntries, dimensions, dimensions, 1))

    startGridU_valid = startGridU_valid.reshape((nEntries, dimensions, dimensions, 1))
    startGridV_valid = startGridV_valid.reshape((nEntries, dimensions, dimensions, 1))
    startGridW_valid = startGridW_valid.reshape((nEntries, dimensions, dimensions, 1))
    endGridU_valid = endGridU_valid.reshape((nEntries, dimensions, dimensions, 1))
    endGridV_valid = endGridV_valid.reshape((nEntries, dimensions, dimensions, 1))
    endGridW_valid = endGridW_valid.reshape((nEntries, dimensions, dimensions, 1))

    nHits2D = nHits2D.reshape((nEntries, 1))
    endX = endX.reshape((nEntries, 1))
    endY = endY.reshape((nEntries, 1))
    endZ = endZ.reshape((nEntries, 1))

    nTrackChildren = nTrackChildren.reshape((nEntries, 1))
    nShowerChildren = nShowerChildren.reshape((nEntries, 1))
    nGrandChildren = nGrandChildren.reshape((nEntries, 1))
    nChildHits = nChildHits.reshape((nEntries, 1))
    childEnergy = childEnergy.reshape((nEntries, 1))
    childTrackScore = childTrackScore.reshape((nEntries, 1))
    trackLength = trackLength.reshape((nEntries, 1))
    trackWobble = trackWobble.reshape((nEntries, 1))
    trackScore = trackScore.reshape((nEntries, 1))
    momComparison = momComparison.reshape((nEntries, 1))

    nTrackChildren_valid = nTrackChildren_valid.reshape((nEntries, 1))
    nShowerChildren_valid = nShowerChildren_valid.reshape((nEntries, 1))
    nGrandChildren_valid = nGrandChildren_valid.reshape((nEntries, 1))
    nChildHits_valid = nChildHits_valid.reshape((nEntries, 1))
    childEnergy_valid = childEnergy_valid.reshape((nEntries, 1))
    childTrackScore_valid = childTrackScore_valid.reshape((nEntries, 1))
    trackLength_valid = trackLength_valid.reshape((nEntries, 1))
    trackWobble_valid = trackWobble_valid.reshape((nEntries, 1))
    trackScore_valid = trackScore_valid.reshape((nEntries, 1))
    momComparison_valid = momComparison_valid.reshape((nEntries, 1))
    
    displacement = displacement.reshape((nEntries, 1))
    dca = dca.reshape((nEntries, 1))
    trackStubLength = trackStubLength.reshape((nEntries, 1))
    nuVertexAvSeparation = nuVertexAvSeparation.reshape((nEntries, 1))
    nuVertexChargeAsymmetry = nuVertexChargeAsymmetry.reshape((nEntries, 1))

    displacement_valid = displacement_valid.reshape((nEntries, 1))
    dca_valid = dca_valid.reshape((nEntries, 1))
    trackStubLength_valid = trackStubLength_valid.reshape((nEntries, 1))
    nuVertexAvSeparation_valid = nuVertexAvSeparation_valid.reshape((nEntries, 1))
    nuVertexChargeAsymmetry_valid = nuVertexChargeAsymmetry_valid.reshape((nEntries, 1))
    
    particlePDG = particlePDG.reshape((nEntries, 1))
    
    ###################################
    # Concatenate
    ###################################
    pfpVars = np.concatenate((endX, endY, endZ), axis=1)
    trackVars = np.concatenate((nTrackChildren, nTrackChildren_valid,
                                nShowerChildren, nShowerChildren_valid,
                                nGrandChildren, nGrandChildren_valid,
                                nChildHits, nChildHits_valid,
                                childEnergy, childEnergy_valid,
                                childTrackScore, childTrackScore_valid,
                                trackLength, trackLength_valid,
                                trackWobble, trackWobble_valid,
                                trackScore, trackScore_valid,
                                momComparison, momComparison_valid, nHits2D), axis=1)
    showerVars = np.concatenate((displacement, displacement_valid,
                                 dca, dca_valid,
                                 trackStubLength, trackStubLength_valid,
                                 nuVertexAvSeparation, nuVertexAvSeparation_valid,
                                 nuVertexChargeAsymmetry, nuVertexChargeAsymmetry_valid), axis=1)

    # Refinement of the particlePDG vector
    print('We have ', str(nEntries), ' PFParticles overall!')
    print('nMuons: ', np.count_nonzero(abs(particlePDG) == 13))
    print('nProtons: ', np.count_nonzero(abs(particlePDG) == 2212))    
    print('nPions: ', np.count_nonzero(abs(particlePDG) == 211))     
    print('nElectrons: ', np.count_nonzero(abs(particlePDG) == 11))     
    print('nPhotons: ', np.count_nonzero(abs(particlePDG) == 22))     
    #print('nOther: ', np.count_nonzero((abs(particlePDG) != 13) & (abs(particlePDG) != 2212) & (abs(particlePDG) != 211) & (abs(particlePDG) != 11) &  (abs(particlePDG) != 22)))
   
    # muons = 0, protons = 1, pions = 2, electrons = 3, photons = 4
    particlePDG[abs(particlePDG) == 13] = 0
    particlePDG[abs(particlePDG) == 2212] = 1
    particlePDG[abs(particlePDG) == 211] = 2
    particlePDG[abs(particlePDG) == 11] = 3
    particlePDG[abs(particlePDG) == 22] = 4
    #particlePDG[(abs(particlePDG) != 0) & (abs(particlePDG) != 1) & (abs(particlePDG) != 2) & (abs(particlePDG) != 3) &  (abs(particlePDG) != 4)] = 5
    
    y = to_categorical(particlePDG, 5)
    
    print('startGridU: ', startGridU.shape)
    print('startGridV: ', startGridV.shape)
    
    return nEntries, startGridU, startGridU_valid, startGridV, startGridV_valid, startGridW, startGridW_valid, endGridU, endGridU_valid, endGridV, endGridV_valid, endGridW, endGridW_valid, pfpVars, trackVars, showerVars, y

#################################################################################################################
#################################################################################################################

def readGrids(fileNames, displacement) :

    ###################################
    # Grid lists
    ###################################
    startGridU = []
    startGridV = []
    startGridW = []
    
    startGridU_disp = []
    startGridV_disp = []
    startGridW_disp = []
    
    endGridU = []
    endGridV = []
    endGridW = []
    
    endGridU_disp = []
    endGridV_disp = []
    endGridW_disp = []
    
    particlePDG = []
    
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
        
        startGridU_disp.extend(branches['StartGridUDisp'])
        startGridV_disp.extend(branches['StartGridVDisp'])
        startGridW_disp.extend(branches['StartGridWDisp'])
        endGridU_disp.extend(branches['EndGridUDisp'])
        endGridV_disp.extend(branches['EndGridVDisp'])
        endGridW_disp.extend(branches['EndGridWDisp'])
        
        particlePDG.extend(branches['TruePDG'])
        
        
    ###################################
    # Now turn things into numpy arrays
    ###################################
    startGridU = np.array(startGridU)
    startGridV = np.array(startGridV)    
    startGridW = np.array(startGridW)   
    endGridU = np.array(endGridU)
    endGridV = np.array(endGridV)
    endGridW = np.array(endGridW)
    
    startGridU_disp = np.array(startGridU_disp)
    startGridV_disp = np.array(startGridV_disp)    
    startGridW_disp = np.array(startGridW_disp)   
    endGridU_disp = np.array(endGridU_disp)
    endGridV_disp = np.array(endGridV_disp)
    endGridW_disp = np.array(endGridW_disp)
    
    particlePDG = np.array(particlePDG)
    
    ###################################
    # Reshape
    ###################################
    startGridU = startGridU.reshape((nEntries, dimensions, dimensions, 1))
    startGridV = startGridV.reshape((nEntries, dimensions, dimensions, 1))
    startGridW = startGridW.reshape((nEntries, dimensions, dimensions, 1))
    endGridU = endGridU.reshape((nEntries, dimensions, dimensions, 1))
    endGridV = endGridV.reshape((nEntries, dimensions, dimensions, 1))
    endGridW = endGridW.reshape((nEntries, dimensions, dimensions, 1))

    startGridU_disp = startGridU_disp.reshape((nEntries, dimensions, dimensions, 1))
    startGridV_disp = startGridV_disp.reshape((nEntries, dimensions, dimensions, 1))
    startGridW_disp = startGridW_disp.reshape((nEntries, dimensions, dimensions, 1))
    endGridU_disp = endGridU_disp.reshape((nEntries, dimensions, dimensions, 1))
    endGridV_disp = endGridV_disp.reshape((nEntries, dimensions, dimensions, 1))
    endGridW_disp = endGridW_disp.reshape((nEntries, dimensions, dimensions, 1))
    
    
    particlePDG = particlePDG.reshape((nEntries, 1))
    
    ###################################
    # Normalise the start and end grids
    ###################################
    '''
    energyLimit = 0.01

    startGridU[startGridU > energyLimit] = energyLimit
    startGridU = startGridU / energyLimit
    
    startGridV[startGridV > energyLimit] = energyLimit
    startGridV = startGridV / energyLimit
    
    startGridW[startGridW > energyLimit] = energyLimit
    startGridW = startGridW / energyLimit
    
    endGridU[endGridU > energyLimit] = energyLimit
    endGridU = endGridU / energyLimit
    
    endGridV[endGridV > energyLimit] = energyLimit
    endGridV = endGridV / energyLimit
    
    endGridW[endGridW > energyLimit] = energyLimit
    endGridW = endGridW / energyLimit
    '''

    # Refinement of the particlePDG vector
    print('We have ', str(nEntries), ' PFParticles overall!')
    print('nMuons: ', np.count_nonzero(abs(particlePDG) == 13))
    print('nProtons: ', np.count_nonzero(abs(particlePDG) == 2212))    
    print('nPions: ', np.count_nonzero(abs(particlePDG) == 211))     
    print('nElectrons: ', np.count_nonzero(abs(particlePDG) == 11))     
    print('nPhotons: ', np.count_nonzero(abs(particlePDG) == 22))     
   
    # muons = 0, protons = 1, pions = 2, electrons = 3, photons = 4
    particlePDG[abs(particlePDG) == 13] = 0
    particlePDG[abs(particlePDG) == 2212] = 1
    particlePDG[abs(particlePDG) == 211] = 2
    particlePDG[abs(particlePDG) == 11] = 3
    particlePDG[abs(particlePDG) == 22] = 4
    
    y = to_categorical(particlePDG, nClasses)
    
    return nEntries, startGridU, startGridV, startGridW, endGridU, endGridV, endGridW, startGridU_disp, startGridV_disp, startGridW_disp, endGridU_disp, endGridV_disp, endGridW_disp, y


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
        momComparison.extend(branches['TrackMomComparison'])
        particlePDG.extend(branches['TruePDG'])

    trackVarsSuccessful = np.array(trackVarsSuccessful)
    nTrackChildren = np.array(nTrackChildren)
    nShowerChildren = np.array(nShowerChildren)
    nGrandChildren = np.array(nGrandChildren)
    nChildHits = np.array(nChildHits)
    childEnergy = np.array(childEnergy)
    childTrackScore = np.array(childTrackScore)
    trackLength = np.array(trackLength)
    trackWobble = np.array(trackWobble)
    trackScore = np.array(trackScore)
    momComparison = np.array(momComparison)
                             
    particlePDG = np.array(particlePDG)
    
    particlePDG[abs(particlePDG) == 13] = 0
    particlePDG[abs(particlePDG) == 2212] = 1
    particlePDG[abs(particlePDG) == 211] = 2
    particlePDG[abs(particlePDG) == 11] = 3
    particlePDG[abs(particlePDG) == 22] = 4
    particlePDG[(abs(particlePDG) != 0) & (abs(particlePDG) != 1) & (abs(particlePDG) != 2) & (abs(particlePDG) != 3) &  (abs(particlePDG) != 4)] = 5
    
    return trackVarsSuccessful, nTrackChildren, nShowerChildren, nGrandChildren, nChildHits, childEnergy, childTrackScore, trackLength, trackWobble, trackScore, momComparison, particlePDG
    
    
#################################################################################################################
#################################################################################################################

def readShowerVars(fileNames) :
    
    # ShowerVar lists
    displacement = []
    dca = []
    trackStubLength = []
    nuVertexAvSeparation = []
    nuVertexChargeAsymmetry = []
    
    initialGapSize = []
    largestGapSize = []
    pathwayLength = []
    pathwayScatteringAngle2D = []
    showerNHits = []
    foundHitRatio = []
    scatterAngle = []
    openingAngle = []
    nuVertexEnergyAsymmetry = []
    nuVertexEnergyWeightedMeanRadialDistance = []
    showerStartEnergyAsymmetry = []
    showerStartMoliereRadius = []
    nAmbiguousViews = []
    unaccountedEnergy = []
    
    particlePDG = []
    
    for fileName in fileNames :
        print('Reading tree: ', str(fileName),', This may take a while...')
    
        treeFile = uproot.open(fileName)
        tree = treeFile['ivyTrain/ivysaur']
        branches = tree.arrays()
    
        displacement.extend(branches['ShowerDisplacement'])
        dca.extend(branches['ShowerDCA'])
        trackStubLength.extend(branches['ShowerTrackStubLength'])
        nuVertexAvSeparation.extend(branches['ShowerNuVertexAvSeparation'])
        nuVertexChargeAsymmetry.extend(branches['ShowerNuVertexChargeAsymmetry'])
        
        # Or I could just do the BDT score... (not in the files anymore...) maybe just a, can it be found?
        initialGapSize.extend(branches['ShowerInitialGapSize'])
        largestGapSize.extend(branches['ShowerLargestGapSize'])
        pathwayLength.extend(branches['ShowerPathwayLength'])
        pathwayScatteringAngle2D.extend(branches['ShowerPathwayScatteringAngle2D'])
        showerNHits.extend(branches['ShowerNHits'])
        foundHitRatio.extend(branches['ShowerFoundHitRatio'])
        scatterAngle.extend(branches['ShowerScatterAngle'])
        openingAngle.extend(branches['ShowerOpeningAngle'])
        nuVertexEnergyAsymmetry.extend(branches['ShowerNuVertexEnergyAsymmetry'])
        nuVertexEnergyWeightedMeanRadialDistance.extend(branches['ShowerNuVertexEnergyWeightedMeanRadialDistance'])
        showerStartEnergyAsymmetry.extend(branches['ShowerStartEnergyAsymmetry'])
        showerStartMoliereRadius.extend(branches['ShowerStartMoliereRadius'])
        nAmbiguousViews.extend(branches['ShowerNAmbiguousViews'])
        unaccountedEnergy.extend(branches['ShowerUnaccountedEnergy'])
        
        particlePDG.extend(branches['TruePDG'])

    displacement = np.array(displacement)
    dca = np.array(dca)
    trackStubLength = np.array(trackStubLength)
    nuVertexAvSeparation = np.array(nuVertexAvSeparation)
    nuVertexChargeAsymmetry = np.array(nuVertexChargeAsymmetry)
    
    initialGapSize = np.array(initialGapSize)
    largestGapSize = np.array(largestGapSize)
    pathwayLength = np.array(pathwayLength)
    pathwayScatteringAngle2D = np.array(pathwayScatteringAngle2D)
    showerNHits = np.array(showerNHits)
    foundHitRatio = np.array(foundHitRatio)
    scatterAngle = np.array(scatterAngle)
    openingAngle = np.array(openingAngle)
    nuVertexEnergyAsymmetry = np.array(nuVertexEnergyAsymmetry)
    nuVertexEnergyWeightedMeanRadialDistance = np.array(nuVertexEnergyWeightedMeanRadialDistance)
    showerStartEnergyAsymmetry = np.array(showerStartEnergyAsymmetry)
    showerStartMoliereRadius = np.array(showerStartMoliereRadius)
    nAmbiguousViews = np.array(nAmbiguousViews)
    unaccountedEnergy = np.array(unaccountedEnergy)
    
    particlePDG = np.array(particlePDG)     
    
    particlePDG[abs(particlePDG) == 13] = 0
    particlePDG[abs(particlePDG) == 2212] = 1
    particlePDG[abs(particlePDG) == 211] = 2
    particlePDG[abs(particlePDG) == 11] = 3
    particlePDG[abs(particlePDG) == 22] = 4
    particlePDG[(abs(particlePDG) != 0) & (abs(particlePDG) != 1) & (abs(particlePDG) != 2) & (abs(particlePDG) != 3) &  (abs(particlePDG) != 4)] = 5
    
    return displacement, dca, trackStubLength, nuVertexAvSeparation, nuVertexChargeAsymmetry, initialGapSize, largestGapSize, pathwayLength, pathwayScatteringAngle2D, showerNHits, foundHitRatio, scatterAngle, openingAngle, nuVertexEnergyAsymmetry, nuVertexEnergyWeightedMeanRadialDistance, showerStartEnergyAsymmetry, showerStartMoliereRadius, nAmbiguousViews, unaccountedEnergy, particlePDG
    


    
    
    
    
    
    
    


