## Importing the required packages
import pandas as pd
# import copy
import numpy as np
import matplotlib.pyplot as plt
import GrowthCurveAnalysis as gra
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error
from keras.layers import LeakyReLU
from mpl_toolkits import mplot3d

# ------------------------------------------------------------------------
## DATA STRUCTURING
# ------------------------------------------------------------------------
# ----- INPUT PARAMETERS -----
OD_600_M9MIN = 0.038
DOWN_SAMPLE_FACTOR = 1
NO_SAMPLES = 700
nLinReg = 12
n_mon = 2
filename = '/Users/shara/Desktop/Vnat_GrowthCurveAnalysis/PFluor_M9min_Expt1.csv'
del_key = []          # UNIQUE WELLS TO DELETE
del_col_key=['1','2','3'] # ONLY NUMBERS 1---12
del_row_key=['E','F','G','H'] # ONLY LETTERS A---H
# ----- Down sampling, Cropping and deleting irrelevant well data -----
dict_MP,dict_INPUT = gra.WELLData_Structuring(filename,OD_600_M9MIN,DOWN_SAMPLE_FACTOR,NO_SAMPLES,del_key,del_row_key,del_col_key)
# ----- Organizing the data -----
Cas_init = []
Glu_init = []
OD_init = []
X0 = []
for keys in dict_MP.keys():
    # Casein
    Cas_init.append(dict_INPUT[keys]['Casein'])
    # Glucose
    Glu_init.append(dict_INPUT[keys]['Glucose'])
    # Initial OD600
    OD_init.append(dict_MP[keys][0])
    # Initial State
    X0.append([dict_MP[keys][items] for items in range(nLinReg) ])
# Each column corresponds to one time instant
X0 = np.asmatrix(X0).T
Cas_init = np.array(Cas_init)[np.newaxis].T
Glu_init = np.array(Glu_init)[np.newaxis].T
OD_init = np.array(OD_init)[np.newaxis].T
keys_all = list(dict_MP.keys())
print('Number of entries: ',len(dict_MP))

## INITIAL FORWARD MAP [OD600(0),C(0),G(0)] ----> X(0)
# U0 = np.asmatrix(np.concatenate([Cas_init,Glu_init,OD_init],axis=1)) # Every row corresponds to one datapoint
# X0T = X0.T # Every row corresponds to one datapoint
# Blin = gra.OLS_SVD_solve3(X0T,U0)
# X0hat = U0*Blin
# f,ax = plt.subplots(8,12,sharex=True,sharey=True)
# for i in range(X0hat.shape[0]):
#     a = divmod(i,12)
#     ax[a[0],a[1]].plot(X0T[i].T)
#     ax[a[0], a[1]].plot(X0hat[i].T)
# f.show()

## NEURAL NET -
# from sklearn.preprocessing import *
# from sklearn.metrics import accuracy_score
# from keras.models import model_from_json
# X = np.asmatrix(U0)
# Y = np.asmatrix(X0T)
# # Curate the test set
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3) # train/test split is 90/10. Change accordingly
# # Build the neural net
# model = gra.BUILD_NN_Model_InitCond()
# # Define model type and compile
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) #https://keras.io/losses/
# # Train the model
# history = model.fit(X_train, Y_train, validation_split=0.5, epochs=200)
# # Evaluation and prediction
# # loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
# Ytrain_pred = model.predict(X_train)
# rmax = 7
# cmax = 6
# f,ax = plt.subplots(rmax,cmax,sharex=True,sharey=True)
# for i in range(np.min([Ytrain_pred.shape[0],rmax*cmax])):
#     a = divmod(i,cmax)
#     ax[a[0],a[1]].plot(Y_train[i].T)
#     ax[a[0],a[1]].plot(Ytrain_pred[i].T,'.')
# f.suptitle('Training Data Fit')
# f.show()
#
# #
# Ytest_pred = model.predict(X_test)
# rmax=4
# cmax =5
# f,ax = plt.subplots(rmax,cmax,sharex=True,sharey=True)
# for i in range(np.min([Ytest_pred.shape[0],rmax*cmax])):
#     a = divmod(i,cmax)
#     ax[a[0],a[1]].plot(Y_test[i].T)
#     ax[a[0],a[1]].plot(Ytest_pred[i].T,'.')
# f.suptitle('Test Data Fit')
# f.show()

# SAVING THE MODEL: model.save()
# ------------------------------------------------------------------------
# Training the Initial Reverse model
# Mapping from [C(0),G(0),OD(0)]
# ------------------------------------------------------------------------

## INPUTS
# mat_U0 = np.asmatrix(np.concatenate([Cas_init.T,Glu_init.T]))
mat_U0 = np.asmatrix(np.concatenate([Glu_init.T]))
dict_X0_i = {}
p = 0
test_size_val = 0.3
valid_split = 0.3
n_pts = 30
for key in dict_MP.keys():
    dict_X0_i[key] = gra.HigherOrderMonomials(dict_MP[key][p*nLinReg:(p+1)*nLinReg],1)#n_mon)  # n_mon kept as 1
df_X0_i = pd.DataFrame.from_dict(dict_X0_i)
mat_X0_i = np.asmatrix(df_X0_i)
mat_X0_i = mat_X0_i[:,0:mat_U0.shape[1]]

## Curate the test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_split_val) # train/test split is 90/10. Change accordingly
plt.figure()
plt.hist(Y_train,bins =4)
plt.hist(Y_test,bins =4)
plt.show()

## Neural Network Framework
NN_input_count = mat_X0_i.shape[0]
X = mat_X0_i.T
Y = mat_U0.T
LOSS_ITER=[]
test_split_val = 0.3
valid_split_val = 0
from keras.models import Sequential
from keras.layers import Dense
# Build the neural net
model = Sequential()
model.add(Dense(12, input_dim=NN_input_count, activation='relu'))
# model.add(LeakyReLU(alpha=0.1))
# model.add(Dense(40, activation='tanh'))
model.add(Dense(12, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(4, activation='tanh'))
# model.add(Dense(10, activation='selu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='exponential'))
# Define model type and compile
model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy']) #https://keras.io/losses/
##
for p in range(10):
    # Train the model
    history = model.fit(X_train, Y_train, batch_size = int(np.floor(X_train.shape[0]/1)),validation_split=valid_split_val, epochs=2000)
    [LOSS_ITER.append(items) for items in history.history['loss']]
    # Evaluation and prediction
    # loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    Ytrain_hat = model.predict(X_train)
    Ytest_hat = model.predict(X_test)
    if np.mod(p,1)==0:
        f,ax = plt.subplots(2,2,sharex=True)
        ax[0,0].plot(Y_train[:,0],'*')
        ax[0,0].plot(Ytrain_hat[:,0],'.')
        ax[0,0].set_title('Casein - Train')
        ax[1,0].plot(Y_test[:,0],'*')
        ax[1,0].plot(Ytest_hat[:,0],'.')
        ax[1,0].set_title('Casein - Test')
        # ax[0,1].plot(Y_train[:,1],'*')
        # ax[0,1].plot(Ytrain_hat[:,1],'.')
        # ax[0,1].set_title('Glucose - Train')
        # ax[1,1].plot(Y_test[:,1],'*')
        # ax[1,1].plot(Ytest_hat[:,1],'.')
        # ax[1,1].set_title('Glucose - Test')
        f.show()

f,ax = plt.subplots(2,1,sharex=True)
ax[0].plot(LOSS_ITER)
ax[1].plot(np.log(LOSS_ITER))
plt.show()

## Koopman Linear Model
# INPUTS
# mat_U0 = np.asmatrix(np.concatenate([Cas_init.T,Glu_init.T]))
# dict_X0_i = {}
# p = 1
# test_size_val = 0.3
# valid_split = 0.2
# n_pts = 30
# for key in dict_MP.keys():
#     dict_X0_i[key] = gra.HigherOrderMonomials(dict_MP[key][p*nLinReg:(p+1)*nLinReg],n_mon)
# df_X0_i = pd.DataFrame.from_dict(dict_X0_i)
# mat_X0_i = np.asmatrix(df_X0_i)
# mat_X0_i = mat_X0_i[:,0:mat_U0.shape[1]]
#
# # FUNCTION
# Yin = mat_U0[:,0:n_pts]
# Xin = mat_X0_i[:,0:n_pts]
# test_split = test_size_val
# PLOT_ON = True
# X_tr, Xreg_test, Y_tr, Yreg_test = gra.train_test_split_vertical_data(Xin, Yin,test_split)
# Xreg_train,Xreg_valid,Yreg_train,Yreg_valid = gra.train_test_split_vertical_data(X_tr, Y_tr, valid_split)
# # Taking logarithm of the variables
# U, S, Vh = np.linalg.svd(Xreg_train)
# U = np.asmatrix(U)
# V = np.asmatrix(Vh.T)
# Ntrain = np.prod(Yreg_train.shape)
# Ntest = np.prod(Yreg_valid.shape)
# err_train = np.zeros(len(S))
# err_valid = np.zeros(len(S))
# E_train = Yreg_train
# E_valid = Yreg_valid
# for iPC in range(0,len(S)): # iPC indicating the instantaneous principal component
#     E_train = E_train - Yreg_train * V[:, iPC] * U[:, iPC].T * Xreg_train / S[iPC]
#     err_train[iPC] = np.linalg.norm(E_train,'fro')**2/Ntrain
#     if valid_split:
#         E_valid = E_valid - Yreg_train * V[:, iPC] * U[:, iPC].T * Xreg_valid / S[iPC]
#         err_valid[iPC] = np.linalg.norm(E_valid,'fro')**2/Ntest
# J = err_train + err_valid
# if PLOT_ON:
#     plt.figure()
#     plt.xlabel('Number of Principal Components')
#     plt.ylabel('Mean Squared Error')
#     plt.plot(err_train)
#     plt.plot(err_valid)
#     plt.plot(J)
#     plt.title('Optimal Fit with both training and test data')
#     plt.show()
# nPC_opt = np.where(J == J.min())[0]
# print('Optimal Number of Principal Components: ',nPC_opt)
# Bopt = Yreg_train * V[:,0:nPC_opt[0]] * np.linalg.inv(np.diag(S[0:nPC_opt[0]])) * U[:,0:nPC_opt[0]].T
# if PLOT_ON:
#     f,ax = plt.subplots(2,2)
#     # Training Set
#     Y_tr_hat = Bopt * X_tr
#     # ax[0,0].plot(np.exp(Y_tr[0,:].T),'*')
#     # ax[0,0].plot(np.exp(Y_tr_hat[0,:].T),'.')
#     ax[0,0].plot(Y_tr[0,:].T,'*')
#     ax[0,0].plot(Y_tr_hat[0,:].T,'.')
#     # ax[0,0].title('Training - Casein')
#     ax[1,0].plot(Y_tr[1,:].T,'*')
#     ax[1,0].plot(Y_tr_hat[1,:].T,'.')
#     # ax[1,0].title('Training - Glucose')
#     if test_split !=0:
#         Y_te_hat = Bopt*Xreg_test
#         # Test Set
#         ax[0,1].plot(Yreg_test[0,:].T,'*')
#         ax[0,1].plot(Y_te_hat[0,:].T,'.')
#         # ax[0,1].title('Test - Casein')
#         ax[1,1].plot(Yreg_test[1,:].T,'*')
#         ax[1,1].plot(Y_te_hat[1,:].T,'.')
#         # ax[1,1].title('Test - Glucose')
#     f.show()
#





##
# # X_tr, X_te, Y_tr, Y_te = gra.train_test_split_vertical_data(np.log(mat_X0_i[:,0:n_pts]), np.log(mat_U0[:,0:n_pts]),test_size_val)
# # X_tr, X_te, Y_tr, Y_te = gra.train_test_split_vertical_data(mat_X0_i[:,0:n_pts], mat_U0[:,0:n_pts],test_size_val)
# Binit = gra.Init_Cond_OLS_SVD_solve(mat_U0[:,0:n_pts],mat_X0_i[:,0:n_pts],test_size_val,valid_split,True)
# # Binit = Y_tr*X_tr.T*np.linalg.inv(X_tr*X_tr.T)
# Y_tr_hat = Binit*X_tr
#
# f,ax = plt.subplots(2,2)
# # Training Set
# # ax[0,0].plot(np.exp(Y_tr[0,:].T),'*')
# # ax[0,0].plot(np.exp(Y_tr_hat[0,:].T),'.')
# ax[0,0].plot(Y_tr[0,:].T,'*')
# ax[0,0].plot(Y_tr_hat[0,:].T,'.')
# # ax[0,0].title('Training - Casein')
# ax[1,0].plot(Y_tr[1,:].T,'*')
# ax[1,0].plot(Y_tr_hat[1,:].T,'.')
# # ax[1,0].title('Training - Glucose')
# if test_size_val !=0:
#     Y_te_hat = Binit*X_te
#     # Test Set
#     ax[0,1].plot(Y_te[0,:].T,'*')
#     ax[0,1].plot(Y_te_hat[0,:].T,'.')
#     # ax[0,1].title('Test - Casein')
#     ax[1,1].plot(Y_te[1,:].T,'*')
#     ax[1,1].plot(Y_te_hat[1,:].T,'.')
#     # ax[1,1].title('Test - Glucose')
# f.show()

# #===========================================================================================================
# ##### Segregating Data as Training Set and Test Set
# # del_key =['A1','A2','A3','A4','A5','B1','B12','C1','D1']
# del_key=['A1','A2','A3','B1','C1','D1','D2','D11','D12','E1','E12']
# for item in del_key:
#     del dict_MP[item]
#     del Cas_init[item]
#     del Glu_init[item]
#     keys_all.remove(item)
# pmax = 49
# train_dict_key = keys_all[0:pmax:2]
# test_dict_key = keys_all[1:pmax:4]
# valid_dict_key = keys_all[3:pmax:4]
# # train_dict_key = keys_all[1:18:2]
# # test_dict_key = keys_all[2:18:2]
# train_dict = {key: dict_MP[key] for key in train_dict_key}
# test_dict = {key: dict_MP[key] for key in test_dict_key}
# valid_dict = {key: dict_MP[key] for key in valid_dict_key}
# X0_train ={key:[] for key in train_dict_key}
# X0_test ={key:[] for key in test_dict_key}
# X0_valid ={key:[] for key in valid_dict_key}
# gra.plot_data(train_dict,test_dict,valid_dict)
# train_Input = {key: [Cas_init[key],Glu_init[key]] for key in train_dict_key}
# train_In = pd.DataFrame.from_dict(train_Input)
# train_In.name = 'TRAINING INPUT'
# test_Input = {key: [Cas_init[key],Glu_init[key]]  for key in test_dict_key}
# test_In = pd.DataFrame.from_dict(test_Input)
# test_In.name = 'TEST INPUT'
# valid_Input = {key: [Cas_init[key],Glu_init[key]]  for key in valid_dict_key}
# valid_In = pd.DataFrame.from_dict(valid_Input)
# valid_In.name = 'VALIDATION INPUT'
# # exit()
# ##################################################################
# ##### MODEL IDENTIFICTION x[k+1] = Ax[k]
# ##################################################################
# # CJDMD Model identification
# n_LinReg_vec = [12]#np.arange(6,20,2)#np.array(list(range(9,14,2))) # Number of Linear-Regressors - monomials of order 1
# n_mon_vec = [2]    # Maximum order of the number of monomials
# A,X1_train,X1_test,X1_valid = gra.CJ_DMD2(train_dict,test_dict,valid_dict,n_LinReg_vec,n_mon_vec)
# train_Out = pd.DataFrame.from_dict(X1_train)
# train_Out.name = 'TRAINING OUTPUT'
# test_Out = pd.DataFrame.from_dict(X1_test)
# test_Out.name = 'TEST OUTPUT'
# valid_Out = pd.DataFrame.from_dict(X1_valid)
# valid_Out.name = 'VALIDATION OUTPUT'
#
# writer = pd.ExcelWriter('/Users/shara/Desktop/Vnat_GrowthCurveAnalysis/InputMap.xlsx')
#
# train_In.to_excel(writer,sheet_name=train_In.name)
# train_Out.to_excel(writer,sheet_name=train_Out.name)
# test_In.to_excel(writer,sheet_name=test_In.name)
# test_Out.to_excel(writer,sheet_name=test_Out.name)
# valid_In.to_excel(writer,sheet_name=valid_In.name)
# valid_Out.to_excel(writer,sheet_name=valid_Out.name)
#
# writer.save()
#
# # exit()
# # Parameters to choose for the model
# # n_LinReg = 20
# # n_mon_max = 3    # Maximum order of the number of monomials
# # A,X1_train,X1_test = gra.A_calc_CJDMD(train_dict, test_dict,n_LinReg,n_mon_max)
# # X1_train = np.asmatrix(pd.DataFrame.from_dict((X1_train)))
# # U0_train = {key: [Cas_init[key]] for key in train_dict_key}
# # u0 = np.asmatrix(pd.DataFrame.from_dict(U0_train)).T
# # b = X1_train*u0/(u0.T*u0)
# # Input Estimation
#
# # Procuring the Data
# # import math
# # X0_train = {keys:[] for keys in X1_train.keys()}
# # for keys in train_dict.keys():
# #     x0 = X1_train[keys][0]
# #     for n_mon in range(1,n_mon_max+1):
# #         nterms = math.factorial(n_LinReg+n_mon-1)/math.factorial(n_LinReg-1)/math.factorial(n_mon)
# #         [X0_train[keys].append(x0**n_mon) for i in range(int(nterms))]
# # U0_train = {keys: [Cas_init[keys]] for keys in X1_train.keys()}
#
# # Formulating the Regressors
# # X0_mat =np.asmatrix(pd.DataFrame.from_dict(X0_train))
# # X1_mat =np.asmatrix(pd.DataFrame.from_dict(X1_train))
# # U0_mat =np.asmatrix(pd.DataFrame.from_dict(U0_train))
# # print(X1_mat.shape)
# # print(X0_mat.shape)
# #
# # Yreg = X1_mat - A * X0_mat
# # Xreg = U0_mat
# # B = Yreg *Xreg.T/np.linalg.inv(Xreg*Xreg.T)
# # Yhat = B * Xreg
# # df_Yreg = pd.DataFrame.from_dict((Yreg))
# # df_Yhat = pd.DataFrame.from_dict((Yhat))
# # print(df_Yreg)
# # print(df_Yhat)
#
# # X1_hat_mat = A * X0_mat + B * U0_mat
# # print(pd.DataFrame.from_dict((X1_hat_mat)))
# # print(pd.DataFrame.from_dict((X1_mat)))
#
# # X0_test= np.asmatrix(pd.DataFrame.from_dict((X0_test)))
# # U1_test = {key: [Cas_init[key]] for key in test_dict_key}
# # u1_test = np.asmatrix(pd.DataFrame.from_dict(U1_test)).T
# # plt.figure()
# #
# #
# # print(X0_train.shape)



