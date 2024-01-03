print('111')

import numpy as np
import math
import glob
import sys
import FileHelper

print('555')

###########################################################

print('AAAAAAA')

dimensions = 24
nClasses = 5

###########################################################

# Here we'll get our information...

import sys
print("Let's read the arguments from command line")
print(sys.argv)
print("sys.argv[1]: ", sys.argv[1])

fileNames = glob.glob('/storage/hpc/30/mawbyi1/Ivysaurus/files/grid24/*/' + sys.argv[1] + '.root')
#fileNames.extend(glob.glob('/storage/hpc/30/mawbyi1/Ivysaurus/files/grid24/nue/*'))
#fileNames.extend(glob.glob('/storage/hpc/30/mawbyi1/Ivysaurus/files/grid24/nutau/*'))
trainVarFile = '/storage/hpc/30/mawbyi1/Ivysaurus/files/grid24/' + sys.argv[1] + '.npz'
print(fileNames)

###########################################################

# Read tree
nEntries, startGridU, startGridV, startGridW, endGridU, endGridV, endGridW, trackVars, showerVars, y = FileHelper.readTree(fileNames, dimensions, nClasses)

###########################################################

# This should shuffle things so that the indicies are still linked
startGridU, startGridV, startGridW, endGridU, endGridV, endGridW, trackVars, showerVars, y = sklearn.utils.shuffle(startGridU, startGridV, startGridW, endGridU, endGridV, endGridW, trackVars, showerVars, y)

###########################################################

# Write file

ntest = math.floor(nEntries * 0.9)
ntrain = math.floor(nEntries * 0.1)

print('ntest: ', ntest)
print('ntrain: ', ntrain)

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
    
showerVars_train = showerVars[:ntrain]
showerVars_test = showerVars[ntrain:(ntrain + ntest)]

y_train = y[:ntrain]
y_test = y[ntrain:(ntrain + ntest)]
    
np.savez(trainVarFile, startGridU_train=startGridU_train, startGridV_train=startGridV_train, startGridW_train=startGridW_train, startGridU_test=startGridU_test, startGridV_test=startGridV_test, startGridW_test=startGridW_test, endGridU_train=endGridU_train, endGridV_train=endGridV_train, endGridW_train=endGridW_train, trackVars_train=trackVars_train, endGridU_test=endGridU_test, endGridV_test=endGridV_test, endGridW_test=endGridW_test, trackVars_test=trackVars_test, showerVars_test=showerVars_test, showerVars_train=showerVars_train, y_train=y_train, y_test=y_test)   

print('BBBBBBB')

