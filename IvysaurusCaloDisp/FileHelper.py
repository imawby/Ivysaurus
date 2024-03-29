import numpy as np
import uproot

from tensorflow.keras.utils import to_categorical


def readTree(fileNames, dimensions, nClasses) :

    ###################################
    # Grid lists
    ###################################
    startGridU_calo = []
    startGridV_calo = []
    startGridW_calo = []
    
    endGridU_calo = []
    endGridV_calo = []
    endGridW_calo = []
    
    startGridU_disp = []
    startGridV_disp = []
    startGridW_disp = []
    
    endGridU_disp = []
    endGridV_disp = []
    endGridW_disp = []    
    
    particlePDG = []
    
    ###################################
    # TrackVar lists
    ###################################
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
    
    ###################################
    # ShowerVar lists  
    ###################################
    showerVarsSuccessful = []
    displacement = []
    dca = []
    trackStubLength = []
    nuVertexAvSeparation = []
    nuVertexChargeAsymmetry = []

    nEntries = 0
    
    for fileName in fileNames :
        print('Reading tree: ', str(fileName),', This may take a while...')
    
        treeFile = uproot.open(fileName)
        tree = treeFile['ivyTrain/ivysaur']
        branches = tree.arrays()
        
        nEntries += len(branches)
    
        startGridU_calo.extend(branches['StartGridU'])
        startGridV_calo.extend(branches['StartGridV'])
        startGridW_calo.extend(branches['StartGridW'])
        endGridU_calo.extend(branches['EndGridU'])
        endGridV_calo.extend(branches['EndGridV'])
        endGridW_calo.extend(branches['EndGridW'])
        
        startGridU_disp.extend(branches['StartGridUDisp'])
        startGridV_disp.extend(branches['StartGridVDisp'])
        startGridW_disp.extend(branches['StartGridWDisp'])
        endGridU_disp.extend(branches['EndGridUDisp'])
        endGridV_disp.extend(branches['EndGridVDisp'])
        endGridW_disp.extend(branches['EndGridWDisp'])
        
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
        
        showerVarsSuccessful.extend(branches['ShowerVarsSuccessful'])
        displacement.extend(branches['ShowerDisplacement'])
        dca.extend(branches['ShowerDCA'])
        trackStubLength.extend(branches['ShowerTrackStubLength'])
        nuVertexAvSeparation.extend(branches['ShowerNuVertexAvSeparation'])
        nuVertexChargeAsymmetry.extend(branches['ShowerNuVertexChargeAsymmetry'])
        
        particlePDG.extend(branches['TruePDG'])
        
        
    ###################################
    # Now turn things into numpy arrays
    ###################################
    startGridU_calo = np.array(startGridU_calo)
    startGridV_calo = np.array(startGridV_calo)    
    startGridW_calo = np.array(startGridW_calo)    
    endGridU_calo = np.array(endGridU_calo)
    endGridV_calo = np.array(endGridV_calo)
    endGridW_calo = np.array(endGridW_calo)
    
    startGridU_disp = np.array(startGridU_disp)
    startGridV_disp = np.array(startGridV_disp)    
    startGridW_disp = np.array(startGridW_disp)    
    endGridU_disp = np.array(endGridU_disp)
    endGridV_disp = np.array(endGridV_disp)
    endGridW_disp = np.array(endGridW_disp)
    
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
    
    displacement = np.array(displacement)
    dca = np.array(dca)
    trackStubLength = np.array(trackStubLength)
    nuVertexAvSeparation = np.array(nuVertexAvSeparation)
    nuVertexChargeAsymmetry = np.array(nuVertexChargeAsymmetry)
    
    particlePDG = np.array(particlePDG)
    
    ###################################
    # Reshape
    ###################################
    startGridU_calo = startGridU_calo.reshape((nEntries, dimensions, dimensions, 1))
    startGridV_calo = startGridV_calo.reshape((nEntries, dimensions, dimensions, 1))
    startGridW_calo = startGridW_calo.reshape((nEntries, dimensions, dimensions, 1))
    endGridU_calo = endGridU_calo.reshape((nEntries, dimensions, dimensions, 1))
    endGridV_calo = endGridV_calo.reshape((nEntries, dimensions, dimensions, 1))
    endGridW_calo = endGridW_calo.reshape((nEntries, dimensions, dimensions, 1))
    
    startGridU_disp = startGridU_disp.reshape((nEntries, dimensions, dimensions, 1))
    startGridV_disp = startGridV_disp.reshape((nEntries, dimensions, dimensions, 1))
    startGridW_disp = startGridW_disp.reshape((nEntries, dimensions, dimensions, 1))
    endGridU_disp = endGridU_disp.reshape((nEntries, dimensions, dimensions, 1))
    endGridV_disp = endGridV_disp.reshape((nEntries, dimensions, dimensions, 1))
    endGridW_disp = endGridW_disp.reshape((nEntries, dimensions, dimensions, 1))
    
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
    
    displacement = displacement.reshape((nEntries, 1))
    dca = dca.reshape((nEntries, 1))
    trackStubLength = trackStubLength.reshape((nEntries, 1))
    nuVertexAvSeparation = nuVertexAvSeparation.reshape((nEntries, 1))
    nuVertexChargeAsymmetry = nuVertexChargeAsymmetry.reshape((nEntries, 1))
    
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
    
    ###################################
    # Normalise track vars
    ###################################
    '''
    nTrackChildrenLimit = 5.0
    nShowerChildrenLimit = 3.0
    nGrandChildrenLimit = 3.0
    nChildHitLimit = 100.0
    childEnergyLimit = 1.0
    childTrackScoreLimit = 1.0
    trackLengthLimit = 500.0
    wobbleLimit = 15.0
    momComparisonLimit = 10.0

    nTrackChildren[nTrackChildren > nTrackChildrenLimit] = nTrackChildrenLimit
    nTrackChildren[nTrackChildren < -0.001] = (-1.0 * nTrackChildrenLimit)    
    nTrackChildren = nTrackChildren / nTrackChildrenLimit
    
    nShowerChildren[nShowerChildren > nShowerChildrenLimit] = nShowerChildrenLimit
    nShowerChildren[nShowerChildren < -0.001] = (-1.0 * nShowerChildrenLimit)  
    nShowerChildren = nShowerChildren / nShowerChildrenLimit
    
    nGrandChildren[nGrandChildren > nGrandChildrenLimit] = nGrandChildrenLimit
    nGrandChildren[nGrandChildren < -0.001] = (-1.0 * nGrandChildrenLimit)  
    nGrandChildren = nGrandChildren / nGrandChildrenLimit
    
    nChildHits[nChildHits > nChildHitLimit] = nChildHitLimit
    nChildHits[nChildHits < -0.001] = (-1.0 * nChildHitLimit)  
    nChildHits = nChildHits / nChildHitLimit
    
    childEnergy[childEnergy > childEnergyLimit] = childEnergyLimit
    childEnergy[childEnergy < -0.001] = (-1.0 * childEnergyLimit)  
    childEnergy = childEnergy / childEnergyLimit
    
    childTrackScore[childTrackScore > childTrackScoreLimit] = childTrackScoreLimit
    childTrackScore[childTrackScore < -0.001] = (-1.0 * childTrackScore)  
    childTrackScore = childTrackScore / childTrackScoreLimit
    
    trackLength[trackLength > trackLengthLimit] = trackLengthLimit
    trackLength[trackLength < -0.001] = (-1.0 * trackLengthLimit)  
    trackLength = trackLength / trackLengthLimit
    
    trackWobble[trackWobble > wobbleLimit] = wobbleLimit
    trackWobble[trackWobble < -0.001] = (-1.0 * wobbleLimit)  
    trackWobble = trackWobble / wobbleLimit  
    
    momComparison[momComparison > momComparisonLimit] = momComparisonLimit
    momComparison[momComparison < -0.001] = (-1.0 * momComparisonLimit) 
    momComparison = momComparison / momComparisonLimit
    '''
    ###################################
    # Normalise shower vars
    ###################################
    '''
    displacementLimit = 150.0
    dcaLimit = 150.0
    trackStubLengthLimit = 100.0
    
    displacement[displacement > displacementLimit] = displacementLimit
    displacement[displacement < 0.0] = (-1.0) * displacementLimit
    displacement = displacement / displacementLimit
    
    dca[dca > dcaLimit] = dcaLimit
    dca[dca < 0.0] = (-1.0) * dcaLimit
    dca = dca / dcaLimit
    
    trackStubLength[trackStubLength > trackStubLengthLimit] = trackStubLengthLimit
    trackStubLength[trackStubLength < 0.0] = (-1.0) * trackStubLengthLimit
    trackStubLength = trackStubLength / trackStubLengthLimit
    '''

    ###################################
    # Concatenate
    ###################################          
    trackVars = np.concatenate((nTrackChildren, nShowerChildren, nGrandChildren, nChildHits, childEnergy, childTrackScore, trackLength, trackWobble, trackScore, momComparison), axis=1)
    showerVars = np.concatenate((displacement, dca, trackStubLength), axis=1)

    ###################################
    # PDG counts
    ################################### 
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
    
    return nEntries, startGridU_calo, startGridV_calo, startGridW_calo, endGridU_calo, endGridV_calo, endGridW_calo, startGridU_disp, startGridV_disp, startGridW_disp, endGridU_disp, endGridV_disp, endGridW_disp, trackVars, showerVars, y

#################################################################################################################
#################################################################################################################



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
    


    
    
    
    
    
    
    


