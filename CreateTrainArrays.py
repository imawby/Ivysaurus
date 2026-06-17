print('111')

import numpy as np
import math
import glob
import sys
import FileHelper
import sklearn

print('555')

###########################################################

print('AAAAAAA')

dimensions = 24

###########################################################

# Here we'll get our information...
#fileNames = glob.glob('/storage/hpc/30/mawbyi1/Ivysaurus/files/grid24/*/' + sys.argv[1] + '.root')
fileNames = ['/Users/isobel/Desktop/DUNE/2024/Ivysaurus/files/revive/filtered_nu_0.root', '/Users/isobel/Desktop/DUNE/2024/Ivysaurus/files/revive/filtered_nue_0.root', '/Users/isobel/Desktop/DUNE/2024/Ivysaurus/files/revive/filtered_nutau_0.root']
#fileNames = ['/Users/isobel/Desktop/DUNE/2024/Ivysaurus/files/revive/filtered_nu_0.root']
trainVarFile = '/Users/isobel/Desktop/DUNE/2024/Ivysaurus/files/revive/filtered_0'
print(fileNames)

###########################################################

# Read tree
nEntries, startGridU, startGridU_valid, startGridV, startGridV_valid, startGridW, startGridW_valid, endGridU, endGridU_valid, endGridV, endGridV_valid, endGridW, endGridW_valid, pfpVars, trackVars, showerVars, y = FileHelper.readTree(fileNames, dimensions)

###########################################################

# This should shuffle things so that the indicies are still linked
startGridU, startGridU_valid, startGridV, startGridV_valid, startGridW, startGridW_valid, endGridU, endGridU_valid, endGridV, endGridV_valid, endGridW, endGridW_valid, pfpVars, trackVars, showerVars, y = sklearn.utils.shuffle(startGridU, startGridU_valid, startGridV, startGridV_valid, startGridW, startGridW_valid, endGridU, endGridU_valid, endGridV, endGridV_valid, endGridW, endGridW_valid, pfpVars, trackVars, showerVars, y)

###########################################################

# Work out contained and uncontained

dune_hd_fd_fv = {
    "MinX":(-360.0 + 50.0),
    "MaxX":(360.0 - 50.0),
    "MinY":(-600.0 + 50.0),
    "MaxY":(600.0 - 50.0),
    "MinZ":(0 + 50.0),
    "MaxZ":(1394.0 - 150.0)
}

is_in_fv_mask = (pfpVars[:,0] > dune_hd_fd_fv["MinX"]) & (pfpVars[:,0] < dune_hd_fd_fv["MaxX"]) &\
                (pfpVars[:,1] > dune_hd_fd_fv["MinY"]) & (pfpVars[:,1] < dune_hd_fd_fv["MaxY"]) &\
                (pfpVars[:,2] > dune_hd_fd_fv["MinZ"]) & (pfpVars[:,2] < dune_hd_fd_fv["MaxZ"])


print('is_in_fv_mask:', is_in_fv_mask.shape)

###########################################################

# Write files
for inFV in [True, False] :

    print('Contained' if inFV else 'Exiting')
    target_mask = is_in_fv_mask == inFV
    n_entries = np.sum(target_mask)
    print('n_entries:', n_entries)

    ntest = math.floor(n_entries * 0.1)
    ntrain = math.floor(n_entries * 0.9)
    print('ntest:', ntest)
    print('ntrain:', ntrain)

    startGridU_train = startGridU[target_mask][:ntrain]
    startGridV_train = startGridV[target_mask][:ntrain]
    startGridW_train = startGridW[target_mask][:ntrain]
    startGridU_test = startGridU[target_mask][ntrain:(ntrain + ntest)]
    startGridV_test = startGridV[target_mask][ntrain:(ntrain + ntest)]
    startGridW_test = startGridW[target_mask][ntrain:(ntrain + ntest)]
    startGridU_valid_train = startGridU_valid[target_mask][:ntrain]
    startGridV_valid_train = startGridV_valid[target_mask][:ntrain]
    startGridW_valid_train = startGridW_valid[target_mask][:ntrain]
    startGridU_valid_test = startGridU_valid[target_mask][ntrain:(ntrain + ntest)]
    startGridV_valid_test = startGridV_valid[target_mask][ntrain:(ntrain + ntest)]
    startGridW_valid_test = startGridW_valid[target_mask][ntrain:(ntrain + ntest)]

    endGridU_train = endGridU[target_mask][:ntrain]
    endGridV_train = endGridV[target_mask][:ntrain]
    endGridW_train = endGridW[target_mask][:ntrain]
    endGridU_test = endGridU[target_mask][ntrain:(ntrain + ntest)]
    endGridV_test = endGridV[target_mask][ntrain:(ntrain + ntest)]
    endGridW_test = endGridW[target_mask][ntrain:(ntrain + ntest)]
    endGridU_valid_train = endGridU_valid[target_mask][:ntrain]
    endGridV_valid_train = endGridV_valid[target_mask][:ntrain]
    endGridW_valid_train = endGridW_valid[target_mask][:ntrain]
    endGridU_valid_test = endGridU_valid[target_mask][ntrain:(ntrain + ntest)]
    endGridV_valid_test = endGridV_valid[target_mask][ntrain:(ntrain + ntest)]
    endGridW_valid_test = endGridW_valid[target_mask][ntrain:(ntrain + ntest)]

    pfpVars_train = pfpVars[:,0][target_mask][:ntrain]
    pfpVars_test = pfpVars[:,0][target_mask][ntrain:(ntrain + ntest)]
    
    trackVars_train = trackVars[target_mask][:ntrain]
    trackVars_test = trackVars[target_mask][ntrain:(ntrain + ntest)]

    showerVars_train = showerVars[target_mask][:ntrain]
    showerVars_test = showerVars[target_mask][ntrain:(ntrain + ntest)]
    
    y_train = y[target_mask][:ntrain]
    y_test = y[target_mask][ntrain:(ntrain + ntest)]

    output_file_name = f'{trainVarFile}_{"Contained" if inFV else "Exiting"}.npz'

    print('startGridU_train', startGridU_train.shape)
    print('startGridV_train', startGridV_train.shape)
    print('startGridW_train', startGridW_train.shape)
    print('startGridU_valid_train', startGridU_valid_train.shape)
    print('startGridV_valid_train', startGridV_valid_train.shape)
    print('startGridW_valid_train', startGridW_valid_train.shape)
    print('endGridU_train', endGridU_train.shape)
    print('endGridV_train', endGridV_train.shape)
    print('endGridW_train', endGridW_train.shape)
    print('endGridU_valid_train', endGridU_valid_train.shape)
    print('endGridV_valid_train', endGridV_valid_train.shape)
    print('endGridW_valid_train', endGridW_valid_train.shape)
    print('pfpVars_train', pfpVars_train.shape)
    print('trackVars_train', trackVars_train.shape)
    print('showerVars_train', showerVars_train.shape)
    print('y_train', y_train.shape)
    print('--------------------------------------')
    print('startGridU_test', startGridU_test.shape)
    print('startGridV_test', startGridV_test.shape)
    print('startGridW_test', startGridW_test.shape)
    print('startGridU_valid_test', startGridU_valid_test.shape)
    print('startGridV_valid_test', startGridV_valid_test.shape)
    print('startGridW_valid_test', startGridW_valid_test.shape)
    print('endGridU_test', endGridU_test.shape)
    print('endGridV_test', endGridV_test.shape)
    print('endGridW_test', endGridW_test.shape)
    print('endGridU_valid_test', endGridU_valid_test.shape)
    print('endGridV_valid_test', endGridV_valid_test.shape)
    print('endGridW_valid_test', endGridW_valid_test.shape)
    print('pfpVars_test', pfpVars_test.shape)
    print('trackVars_test', trackVars_test.shape)
    print('showerVars_test', showerVars_test.shape)
    print('y_test', y_test.shape)    

    np.savez(output_file_name,
         startGridU_train=startGridU_train, startGridU_valid_train=startGridU_valid_train,
         startGridV_train=startGridV_train, startGridV_valid_train=startGridV_valid_train,
         startGridW_train=startGridW_train, startGridW_valid_train=startGridW_valid_train,
         startGridU_test=startGridU_test, startGridU_valid_test=startGridU_valid_test,
         startGridV_test=startGridV_test, startGridV_valid_test=startGridV_valid_test,
         startGridW_test=startGridW_test, startGridW_valid_test=startGridW_valid_test,
         endGridU_train=endGridU_train, endGridU_valid_train=endGridU_valid_train,
         endGridV_train=endGridV_train, endGridV_valid_train=endGridV_valid_train,
         endGridW_train=endGridW_train, endGridW_valid_train=endGridW_valid_train,
         endGridU_test=endGridU_test, endGridU_valid_test=endGridU_valid_test,
         endGridV_test=endGridV_test, endGridV_valid_test=endGridV_valid_test,
         endGridW_test=endGridW_test, endGridW_valid_test=endGridW_valid_test,
         trackVars_train=trackVars_train,
         trackVars_test=trackVars_test,
         showerVars_test=showerVars_test,
         showerVars_train=showerVars_train,
         y_train=y_train,
         y_test=y_test)

    print('--------------------------------------')
    print('--------------------------------------')
    
###########################################################

print('BBBBBBB')

