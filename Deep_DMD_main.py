import pandas as pd
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
import GrowthCurveAnalysis as gra

# Constants
# OD_600_M9MIN = 0.038
# OD_600_R2A = # Yet to be determined

# INPUTS
filename = 'PFluor_M9min_Expt1'
with open('/Users/shara/Desktop/Vnat_GrowthCurveAnalysis/Data_'+filename +'.txt') as f:
    f.readline()
    Xp=[]
    Xf=[]
    Up=[]
    f.readline()
    for line in f:
        iteritem = re.split('\s', line)
        iteritem.remove(iteritem[-1])
        if iteritem[0]=='':
            break
        else:
            Xp.append([float(items) for items in iteritem])
    f.readline()
    for line in f:
        iteritem = re.split('\s', line)
        iteritem.remove(iteritem[-1])
        if iteritem[0]=='':
            break
        else:
            Xf.append([float(items) for items in iteritem])
    f.readline()
    for line in f:
        iteritem = re.split('\s', line)
        iteritem.remove(iteritem[-1])
        if iteritem[0]=='':
            break
        else:
            Up.append([float(items) for items in iteritem])
Xp = np.asmatrix(pd.DataFrame(Xp))
Xf = np.asmatrix(pd.DataFrame(Xf))
Up = np.asmatrix(pd.DataFrame(Up))


