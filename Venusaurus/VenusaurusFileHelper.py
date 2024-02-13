import numpy as np
import uproot
import awkward as ak
import math

from tensorflow.keras.utils import to_categorical

def readTree(fileName, nClasses) :

    treeFile = uproot.open(fileName)
    tree = treeFile['venusaurus/venusaur']
    branches = tree.arrays()
    nEntries = len(branches['Longitudinal_track'])
    
    #######################################
    # First, get the energy profiles
    #######################################
    binWidth_l = 0.5   # Units cm - wire pitch?
    targetNBins_l = 50 # This equates to 25cm in length
    binWidth_t = 0.5   # Units cm - wire pitch?
    targetNBins_t = 20 # This equates to 10cm in length - moliere radius

    longitudinal_track = branches['Longitudinal_track'] # Longitudinal distance using sliding linear fit
    longitudinal_recoStart = branches['Longitudinal_recoStart'] # Longitudinal distance using initial direction
    
    transverse_track = branches['Transverse_track'] # Transverse distance using sliding linear fit
    transverse_recoStart = branches['Transverse_recoStart'] # Transverse distance using initial direction
    
    energy_track = branches['Energy_track'] 
    energy_recoStart = branches['Energy_recoStart'] 
    
    trackScore = branches['TrackScore'] 

    trackBools = trackScore > 0.5

    #######################################
    # Combine
    #######################################
    print('Merge track/shower fitting results')

    long_reweight = [(longitudinal_track[i] if (trackScore[i] > 0.5) else longitudinal_recoStart[i]) for i in range(nEntries)]
    trans_reweight = [(transverse_track[i] if (trackScore[i] > 0.5) else transverse_recoStart[i]) for i in range(nEntries)]
    energy_reweight = [(energy_track[i] if (trackScore[i] > 0.5) else energy_recoStart[i]) for i in range(nEntries)]

    # Make sure everything has an entry - later code relies on this
    energy_reweight = [ak.Array([0]) if len(energy_reweight[i]) == 0 else energy_reweight[i] for i in range(nEntries)]
    trans_reweight = [ak.Array([0]) if len(trans_reweight[i]) == 0 else trans_reweight[i] for i in range(nEntries)]   
    long_reweight = [ak.Array([0]) if len(long_reweight[i]) == 0 else long_reweight[i] for i in range(nEntries)]
    
    ####################################
    # Transverse
    ####################################
    print('Get the transverse profiles...')
    
    # Create histograms (padded and truncated!)
    transProfiles = [np.histogram(trans_reweight[i].to_numpy(), targetNBins_t, [0, (float(targetNBins_t) * binWidth_t)], weights=energy_reweight[i].to_numpy())[0] for i in range(nEntries)]

    # Turn into output numpy array
    transProfiles = np.array(transProfiles) 

    ####################################
    # Longitudinal
    ####################################
    print('Get the longitudinal profiles...')

    # Create histograms, length encapsulated by at least (targetNBins_l * binWidth_l)
    longProfiles = [np.histogram(long_reweight[i].to_numpy(), max(targetNBins_l, math.ceil(ak.max(long_reweight[i]) / binWidth_l)), [0, binWidth_l * float(max(targetNBins_l, math.ceil(ak.max(long_reweight[i]) / binWidth_l)))], weights=energy_reweight[i].to_numpy())[0] for i in range(nEntries)]

    # Get start vectors
    longProfiles_start = [longProfiles[i][0 : targetNBins_l] for i in range(nEntries)]

    # Get end vectors
    longProfiles_end = [longProfiles[i][(max(targetNBins_l, len(longProfiles[i])) - targetNBins_l) : ] for i in range(nEntries)]

    # Turn into output numpy array
    longProfiles_start = np.array(longProfiles_start)
    longProfiles_end = np.array(longProfiles_end)
        
    #################################
    # Track Vars
    #################################
    nTrackChildren = np.array(branches['NTrackChildren'])
    nShowerChildren = np.array(branches['NShowerChildren'])
    nGrandChildren = np.array(branches['NGrandChildren'])
    nChildHits = np.array(branches['NChildHits'])
    childEnergy = np.array(branches['ChildEnergy'])
    childTrackScore = np.array(branches['ChildTrackScore'])
    trackLength = np.array(branches['TrackLength'])
    trackWobble = np.array(branches['TrackWobble'])                
    trackScore = np.array(branches['TrackScore'])
    momComparison = np.array(branches['TrackMomComparison'])  
    
    #################################
    # Shower Vars
    #################################    
    displacement = np.array(branches['ShowerDisplacement'])
    dca = np.array(branches['ShowerDCA'])
    trackStubLength = np.array(branches['ShowerTrackStubLength'])
    nuVertexAvSeparation = np.array(branches['ShowerNuVertexAvSeparation'])
    nuVertexChargeAsymmetry = np.array(branches['ShowerNuVertexChargeAsymmetry'])    

    #################################
    # True PDG
    #################################         
    particlePDG = np.array(branches['TruePDG'])    
        
    ###################################
    # Reshape
    ###################################     
    transProfiles = transProfiles.reshape(nEntries, targetNBins_t, 1)
    longProfiles_start = longProfiles_start.reshape(nEntries, targetNBins_l, 1)
    longProfiles_end = longProfiles_end.reshape(nEntries, targetNBins_l, 1)   
    
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

    
    return longProfiles_start, longProfiles_end, transProfiles, trackVars, showerVars, y   
    
#####################################################################################################################    

def readTreeOld(fileName, nClasses) :

    treeFile = uproot.open(fileName)
    tree = treeFile['venusaurus/venusaur']
    branches = tree.arrays()
    nEntries = len(branches['Longitudinal_track'])
    
    #######################################
    # First, get the energy profiles
    #######################################
    binWidth_l = 0.5   # Units cm - wire pitch?
    targetNBins_l = 50 # This equates to 25cm in length
    binWidth_t = 0.5   # Units cm - wire pitch?
    targetNBins_t = 20 # This equates to 10cm in length - moliere radius

    longitudinal_track = branches['Longitudinal_track'] # Longitudinal distance using sliding linear fit
    longitudinal_recoStart = branches['Longitudinal_recoStart'] # Longitudinal distance using initial direction
    
    transverse_track = branches['Transverse_track'] # Transverse distance using sliding linear fit
    transverse_recoStart = branches['Transverse_recoStart'] # Transverse distance using initial direction
    
    energy_track = branches['Energy_track'] 
    energy_recoStart = branches['Energy_recoStart'] 
    
    trackScore = branches['TrackScore']


    #######################################    
    # Loop over tree - welp (because of jagged arrays...)
    #######################################
    # Define numpy arrays to fill 
    longitudinalProfile_start = np.empty((0, targetNBins_l))
    longitudinalBinIndicies_start = np.empty((0, targetNBins_l))

    longitudinalProfile_end = np.empty((0, targetNBins_l))
    longitudinalBinIndicies_end = np.empty((0, targetNBins_l))

    transverseProfile = np.empty((0, targetNBins_t))
    transverseBinIndicies = np.empty((0, targetNBins_t))

    # Loop the loop
    for entry in range(0, len(trackScore)) :

        #################################
        # TrackScore?
        #################################
        isTrack = trackScore > 0.5
    
        longitudinal = longitudinal_track if isTrack else longitudinal_recoStart
        transverse = transverse_track if isTrack else transverse_recoStart
        energy = energy_track if isTrack else energy_recoStart        
        
        #################################
        # Transverse - a bit overkill, but left for if i change in the future
        #################################
        # Work out how many bins the trajectory covers
        thisMax_t = ak.max(transverse)
        thisNBins_t = math.ceil(thisMax_t / binWidth_t)    
    
        # Turn into a histogram
        thisProfile_t, edges_t = np.histogram(transverse.to_numpy(), thisNBins_t, range=[0, thisMax_t], weights=energy.to_numpy())
        
        # Pad if needed
        if (thisNBins_t < targetNBins_t) :
            thisProfile_t = np.concatenate((thisProfile_t, np.zeros(targetNBins_t - thisNBins_t)), axis=0)        
        
        # Get position indexing vector
        thisBinIndexVector_t = np.arange(0, len(thisProfile_t))
        
         # Get truncated vector
        thisProfile_t_trunc = thisProfile_t[0 : targetNBins_t]
        thisBinIndicies_t_trunc = thisBinIndexVector_t[0 : targetNBins_t] # this will always be the same...
        
        # Concatenate
        transverseProfile = np.concatenate((transverseProfile, thisProfile_t_trunc.reshape(1, targetNBins_t)), axis=0)
        transverseBinIndicies = np.concatenate((transverseBinIndicies, thisBinIndicies_t_trunc.reshape(1, targetNBins_t)), axis=0)
    
        #################################
        # Longitudinal
        #################################
        # Work out how many bins the trajectory covers
        thisMax_l = ak.max(longitudinal)
        thisNBins_l = math.ceil(thisMax_l / binWidth_l)

        # Turn into a histogram
        thisProfile_l, edges = np.histogram(longitudinal.to_numpy(), thisNBins_l, range=[0, thisMax_l], weights=energy.to_numpy())

        # Pad if needed
        if (thisNBins_l < targetNBins_l) :
            thisProfile_l = np.concatenate((thisProfile_l, np.zeros(targetNBins_l - thisNBins_l)), axis=0)

        # Get position indexing vector
        thisBinIndexVector_l = np.arange(0, len(thisProfile_l))

        # Get start and end vectors
        thisProfile_l_start = thisProfile_l[0 : targetNBins_l]
        thisBinIndexVector_l_start = thisBinIndexVector_l[0 : targetNBins_l]

        thisProfile_l_end = thisProfile_l[(thisNBins_l - targetNBins_l) : ]
        thisBinIndexVector_l_end = thisBinIndexVector_l[(thisNBins_l - targetNBins_l) : ]
    
        # Concatenate
        longitudinalProfile_start = np.concatenate((longitudinalProfile_start, thisProfile_l_start.reshape(1, targetNBins_l)), axis=0)
        longitudinalBinIndicies_start = np.concatenate((longitudinalBinIndicies_start, thisBinIndexVector_l_start.reshape(1, targetNBins_l)), axis=0)
        
        longitudinalProfile_end = np.concatenate((longitudinalProfile_end, thisProfile_l_end.reshape(1, targetNBins_l)), axis=0)
        longitudinalBinIndicies_end = np.concatenate((longitudinalBinIndicies_end, thisBinIndexVector_l_end.reshape(1, targetNBins_l)), axis=0)
        
        
    #################################
    # Track Vars
    #################################
    nTrackChildren = np.array(branches['NTrackChildren'])
    nShowerChildren = np.array(branches['NShowerChildren'])
    nGrandChildren = np.array(branches['NGrandChildren'])
    nChildHits = np.array(branches['NChildHits'])
    childEnergy = np.array(branches['ChildEnergy'])
    childTrackScore = np.array(branches['ChildTrackScore'])
    trackLength = np.array(branches['TrackLength'])
    trackWobble = np.array(branches['TrackWobble'])                
    trackScore = np.array(branches['TrackScore'])
    momComparison = np.array(branches['TrackMomComparison'])  
    
    #################################
    # Shower Vars
    #################################    
    displacement = np.array(branches['ShowerDisplacement'])
    dca = np.array(branches['ShowerDCA'])
    trackStubLength = np.array(branches['ShowerTrackStubLength'])
    nuVertexAvSeparation = np.array(branches['ShowerNuVertexAvSeparation'])
    nuVertexChargeAsymmetry = np.array(branches['ShowerNuVertexChargeAsymmetry'])    

    #################################
    # True PDG
    #################################         
    particlePDG = np.array(branches['TruePDG'])    
        
    ###################################
    # Reshape
    ###################################      
    longitudinalProfile_start = longitudinalProfile_start.reshape((nEntries, targetNBins_l))
    longitudinalBinIndicies_start = longitudinalBinIndicies_start.reshape((nEntries, targetNBins_l))
    longitudinalProfile_end = longitudinalProfile_end.reshape((nEntries, targetNBins_l))
    longitudinalBinIndicies_end = longitudinalBinIndicies_end.reshape((nEntries, targetNBins_l))    
    transverseProfile = transverseProfile.reshape((nEntries, targetNBins_t))    
    transverseBinIndicies = transverseBinIndicies.reshape((nEntries, targetNBins_t))    
    
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

    return longitudinalProfile_start, longitudinalBinIndicies_start, longitudinalProfile_end, longitudinalBinIndicies_end, transverseProfile, transverseBinIndicies, trackVars, showerVars, y