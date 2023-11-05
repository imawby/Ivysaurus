def PDGToColor(pdg) :
    
    if (abs(pdg) == 13) :
        return 'b'
    elif (abs(pdg) == 22) :
        return 'tab:orange'
    elif (abs(pdg) == 11) :
        return 'r'
    elif (abs(pdg) == 211) :
        return 'k'
    elif (abs(pdg) == 2212) :
        return 'g'
    else :
        return 'tab:gray'
