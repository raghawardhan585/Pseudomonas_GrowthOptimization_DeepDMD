import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import GrowthCurveAnalysis as gra



##################################################################
##### DATA STRUCTURING
##################################################################
DOWN_SAMPLE_FACTOR = 6
NO_SAMPLES = 70
###
# The file from the microplate reader is parsed using main.py and the useful data is converted into a .csv file
# .csv ---> pandas ---> dictionary file with keys
# MP = pd.read_csv('Expt4.csv')      # MP - Micro Plate
MP = pd.read_csv('/Users/shara/Desktop/Vnat_GrowthCurveAnalysis/PPutida_M9min_Expt1.csv')      # MP - Micro Plate
MP = MP.iloc[:,2:MP.shape[1]]
# gra.plot_Growth(MP)
# exit()
# MP = MP.drop(columns =['A1'])  # TO BE MODIFIED AFTER SEEING ALL THE CURVES
MP_dict = MP.to_dict('list')       # MP_dict - Dictionary of microplate reader data with keys
keys_all = list(MP_dict.keys())

Cas_mass_mg_base = np.zeros(8)
Glu_mass_mg_base = np.zeros(12)
Cas_mass_mg_base[0] = 9.375
Glu_mass_mg_base[0] = 9.375
for i in range(1,8):
    Cas_mass_mg_base[i]=Cas_mass_mg_base[i-1]/2
for i in range(1, 12):
    Glu_mass_mg_base[i] = Glu_mass_mg_base[i-1]/2

Cas_init={}
Glu_init={}
c = 0
g = 0
row_int='A'
for keys in MP_dict.keys():
    # CREATING THE INITIAL CONDITION DICTIONARY
    if row_int != keys[0]:
        row_int = chr(ord(row_int)+1)
        c = c+1
    Cas_init[keys] = Cas_mass_mg_base[c]
    Glu_init[keys] = Glu_mass_mg_base[g % 12]
    g = g+1
    # r = list(range(0,len(MP_dict[keys]),2))
    MP_dict[keys] = [np.mean(MP_dict[keys][i:i+DOWN_SAMPLE_FACTOR]) for i in range(0,len(MP_dict[keys]),DOWN_SAMPLE_FACTOR)] # DOWNSAMPLE
    MP_dict[keys] = MP_dict[keys][0:NO_SAMPLES]

##### Segregating Data as Training Set and Test Set
# del_key =['A1','A2','A3','A4','A5','B1','B12','C1','D1']
del_key=['A1','A2','A3','B1','C1','D1','D2','D11','D12','E1','E12']
del_key=['A1','A2','B1','C1','D1','D3','E1']
for item in del_key:
    del MP_dict[item]
    del Cas_init[item]
    del Glu_init[item]
    keys_all.remove(item)
pmax = 50
train_dict_key = keys_all[0:pmax:2]
test_dict_key = keys_all[1:pmax:4]
valid_dict_key = keys_all[3:pmax:4]
# train_dict_key = keys_all[1:18:2]
# test_dict_key = keys_all[2:18:2]
train_dict = {key: MP_dict[key] for key in train_dict_key}
test_dict = {key: MP_dict[key] for key in test_dict_key}
valid_dict = {key: MP_dict[key] for key in valid_dict_key}
X0_train ={key:[] for key in train_dict_key}
X0_test ={key:[] for key in test_dict_key}
X0_valid ={key:[] for key in valid_dict_key}
gra.plot_data(train_dict,test_dict,valid_dict)
train_Input = {key: [Cas_init[key],Glu_init[key]] for key in train_dict_key}
train_In = pd.DataFrame.from_dict(train_Input)
train_In.name = 'TRAINING INPUT'
test_Input = {key: [Cas_init[key],Glu_init[key]]  for key in test_dict_key}
test_In = pd.DataFrame.from_dict(test_Input)
test_In.name = 'TEST INPUT'
valid_Input = {key: [Cas_init[key],Glu_init[key]]  for key in valid_dict_key}
valid_In = pd.DataFrame.from_dict(valid_Input)
valid_In.name = 'VALIDATION INPUT'
# exit()
##################################################################
##### MODEL IDENTIFICTION x[k+1] = Ax[k]
##################################################################
# CJDMD Model identification
# n_LinReg_vec = np.arange(6,21,2)#np.array(list(range(9,14,2))) # Number of Linear-Regressors - monomials of order 1
# n_mon_vec = [2,3]    # Maximum order of the number of monomials
n_LinReg_vec = [14]
n_mon_vec = [2]
A,X1_train,X1_test,X1_valid = gra.CJ_DMD2(train_dict,test_dict,valid_dict,n_LinReg_vec,n_mon_vec)
train_Out = pd.DataFrame.from_dict(X1_train)
train_Out.name = 'TRAINING OUTPUT'
test_Out = pd.DataFrame.from_dict(X1_test)
test_Out.name = 'TEST OUTPUT'
valid_Out = pd.DataFrame.from_dict(X1_valid)
valid_Out.name = 'VALIDATION OUTPUT'

writer = pd.ExcelWriter('/Users/shara/Desktop/Vnat_GrowthCurveAnalysis/InputMap.xlsx')

train_In.to_excel(writer,sheet_name=train_In.name)
train_Out.to_excel(writer,sheet_name=train_Out.name)
test_In.to_excel(writer,sheet_name=test_In.name)
test_Out.to_excel(writer,sheet_name=test_Out.name)
valid_In.to_excel(writer,sheet_name=valid_In.name)
valid_Out.to_excel(writer,sheet_name=valid_Out.name)

writer.save()

# f=open('/Users/shara/Desktop/Vnat_GrowthCurveAnalysis/InputMap.txt','w+')
# f.write('TRAIN_DATA_INPUT')
# f.write(pd.DataFrame.from_dict(train_Input))
# f.write('TRAIN_DATA_OUTPUT')
# f.write(pd.DataFrame.from_dict(X1_train))
# f.write('TEST_DATA_INPUT')
# f.write(pd.DataFrame.from_dict(test_Input))
# f.write('TEST_DATA_OUTPUT')
# f.write(pd.DataFrame.from_dict(X1_test))
# f.write('VALIDATION_DATA_INPUT')
# f.write(pd.DataFrame.from_dict(valid_Input))
# f.write('VALIDATION_DATA_OUTPUT')
# f.write(pd.DataFrame.from_dict(X1_valid))
# f.close()
exit()
# Parameters to choose for the model
n_LinReg = 20
n_mon_max = 3    # Maximum order of the number of monomials
#
# # INITIALIZATION
# CHOICE = 'Disjoint' # 'Disjoint' or 'Overlap'
# # STEP 1 -
# dict_Xreg_train,dict_Xreg_test = gra.STEP_1_Init(train_dict,test_dict,n_LinReg,CHOICE)
# dict_Xmult_train = copy.deepcopy(dict_Xreg_train)
# dict_Xinter_train = copy.deepcopy(dict_Xreg_train)
# dict_Xmult_test = copy.deepcopy(dict_Xreg_test)
# dict_Xinter_test = copy.deepcopy(dict_Xreg_test)
# # STEP 2 - Formulate the Regressors - Iterative Fashion
# Yreg_train,Xreg_train = gra.STEP_2_Regressor_Form(dict_Xreg_train,n_LinReg,CHOICE)
# Yreg_test,Xreg_test = gra.STEP_2_Regressor_Form(dict_Xreg_test,n_LinReg,CHOICE)
# # STEP 3 - Solve using Ordinary Least Squares
# A,err2,nPC = gra.OLS_SVD_solve(Yreg_train,Xreg_train,Yreg_test,Xreg_test,1,False)
# print('|#monomials |       Error')
# print('|      1    | ',err2)
# for keys in dict_Xreg_train.keys():
#     X0_train[keys] = dict_Xreg_train[keys][0]
# for keys in dict_Xreg_test.keys():
#     X0_test[keys] = dict_Xreg_test[keys][0]
# gra.plot_fit(train_dict,test_dict,X0_train,X0_test,A,n_LinReg,1)
#
# # PROPAGATION
# for n_mon in range(2,n_mon_max+1):
#     # STEP 1 - Formulate the Regressors - Iterative Fashion
#     for key in dict_Xmult_train.keys():
#         for iter in dict_Xmult_train[key].keys():
#             # print(dict_Xinter_train[key][iter])
#             xtemp = gra.STEP_1_HigherOrderMonomials(dict_Xinter_train[key][iter],dict_Xmult_train[key][iter],n_LinReg)
#             [dict_Xreg_train[key][iter].append(items) for items in xtemp]
#             dict_Xinter_train[key][iter] = xtemp
#         X0_train[keys] = dict_Xreg_train[key][0]
#     for key in dict_Xmult_test.keys():
#         for iter in dict_Xmult_test[key].keys():
#             xtemp = gra.STEP_1_HigherOrderMonomials(dict_Xinter_test[key][iter], dict_Xmult_test[key][iter],n_LinReg)
#             [dict_Xreg_test[key][iter].append(items) for items in xtemp]
#             dict_Xinter_test[key][iter] = xtemp
#         X0_test[keys] = dict_Xreg_test[key][0]
#     # STEP 2 - Formulate the Regressors - Iterative Fashion
#     Yreg_train, Xreg_train = gra.STEP_2_Regressor_Form(dict_Xreg_train,n_LinReg,CHOICE)
#     Yreg_test, Xreg_test = gra.STEP_2_Regressor_Form(dict_Xreg_test,n_LinReg,CHOICE)
#     # STEP 3 - OLS solution - SOLVE FOR A in Y = AX
#     A,err2,nPC = gra.OLS_SVD_solve(Yreg_train, Xreg_train, Yreg_test, Xreg_test,n_mon,False)
#     print('|     ',n_mon,'   | ', err2)
#     gra.plot_fit(train_dict, test_dict, X0_train, X0_test, A, n_LinReg, n_mon)


A,X1_train,X1_test = gra.A_calc_CJDMD(train_dict, test_dict,n_LinReg,n_mon_max)
# X1_train = np.asmatrix(pd.DataFrame.from_dict((X1_train)))
# U0_train = {key: [Cas_init[key]] for key in train_dict_key}
# u0 = np.asmatrix(pd.DataFrame.from_dict(U0_train)).T
# b = X1_train*u0/(u0.T*u0)
# Input Estimation

# Procuring the Data
import math
X0_train = {keys:[] for keys in X1_train.keys()}
for keys in train_dict.keys():
    x0 = X1_train[keys][0]
    for n_mon in range(1,n_mon_max+1):
        nterms = math.factorial(n_LinReg+n_mon-1)/math.factorial(n_LinReg-1)/math.factorial(n_mon)
        [X0_train[keys].append(x0**n_mon) for i in range(int(nterms))]
U0_train = {keys: [Cas_init[keys]] for keys in X1_train.keys()}

# Formulating the Regressors
X0_mat =np.asmatrix(pd.DataFrame.from_dict(X0_train))
X1_mat =np.asmatrix(pd.DataFrame.from_dict(X1_train))
U0_mat =np.asmatrix(pd.DataFrame.from_dict(U0_train))
print(X1_mat.shape)
print(X0_mat.shape)

Yreg = X1_mat - A * X0_mat
Xreg = U0_mat
B = Yreg *Xreg.T/np.linalg.inv(Xreg*Xreg.T)
Yhat = B * Xreg
df_Yreg = pd.DataFrame.from_dict((Yreg))
df_Yhat = pd.DataFrame.from_dict((Yhat))
print(df_Yreg)
print(df_Yhat)

# X1_hat_mat = A * X0_mat + B * U0_mat
# print(pd.DataFrame.from_dict((X1_hat_mat)))
# print(pd.DataFrame.from_dict((X1_mat)))

# X0_test= np.asmatrix(pd.DataFrame.from_dict((X0_test)))
# U1_test = {key: [Cas_init[key]] for key in test_dict_key}
# u1_test = np.asmatrix(pd.DataFrame.from_dict(U1_test)).T
# plt.figure()
#
#
# print(X0_train.shape)
