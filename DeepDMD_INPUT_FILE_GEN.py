import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import GrowthCurveAnalysis as gra
# Constants
# OD_600_M9MIN = 0.038
# OD_600_R2A = # Yet to be determined

# INPUTS
filename = 'PFluor_M9min_Expt1'
OD_START = 0 #OD_600_M9MIN
del_key = []          # UNIQUE WELLS TO DELETE
del_col_key=[] # '1','2','3'] # ONLY NUMBERS 1---12
del_row_key=[] # 'E','F','G','H'] # ONLY LETTERS A---H

# Getting the Dataset
MP = pd.read_csv('/Users/shara/Desktop/Vnat_GrowthCurveAnalysis/'+filename+'.csv')
MP = MP.iloc[:,2:MP.shape[1]] - OD_START

# Getting the inputs [in mg]
INPUT_INTER = gra.TwoD_SerialDil()
dict_INPUT = {}
i = -1
# INPUT
for items in list(MP.columns):
    i = i + 1
    dict_INPUT[items] = {'Casein': INPUT_INTER.iloc[i, 0] * INPUT_INTER.iloc[i, 2] * 1e-3,
                         'Glucose': INPUT_INTER.iloc[i, 1] * INPUT_INTER.iloc[i, 2] * 1e-3}  # UNITS: [mg]
dict_MP = gra.plot_Growth(MP)

# DELETE THE DATA THAT IS NOT REQUIRED
del_key = set(del_key)  # To add only the unique entries
for items in del_row_key:
    for col in range(1, 13):
        del_key.add(items + str(col))
for items in del_col_key:
    row = 'A'
    for c in range(8):
        del_key.add(row + items)
        row = chr(ord(row) + 1)
for item in del_key:
    del dict_MP[item]
    del dict_INPUT[item]
# dict_MP = MP.to_dict('list')  # MP_dict - Dictionary of microplate reader data with keys

# Formulate Xp[OD600],Xf[OD600] and Up[mg] as matrices
Xp=[]
Xf=[]
U1p=[]
U2p=[]
for keys in dict_MP.keys():
    for i in range(len(dict_MP[keys])-1):
        Xp.append(dict_MP[keys][i])
        Xf.append(dict_MP[keys][i+1])
        U1p.append(dict_INPUT[keys]['Casein'])  # Creating a step input of Casein
        U2p.append(dict_INPUT[keys]['Glucose']) # Creating a step inout of Glucose
Xp = np.asmatrix(Xp)
Xf = np.asmatrix(Xf)
Up = np.asmatrix([U1p,U2p])
with open('/Users/shara/Desktop/Vnat_GrowthCurveAnalysis/Data_'+filename +'.txt','w') as f:
    f.write('\nXp\n')
    np.savetxt(f, Xp, fmt='%.4f')
    f.write('\nXf\n')
    np.savetxt(f, Xf, fmt='%.4f')
    f.write('\nUp\n')
    np.savetxt(f, Up, fmt='%.4f')