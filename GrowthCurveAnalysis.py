import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from math import factorial as fctl
from itertools import combinations_with_replacement as cwb
from sklearn.model_selection import train_test_split

def OLS_SVD_solve(Yreg_train,Xreg_train,Yreg_test,Xreg_test,n_mon=1,PLOT_ON=False):
    # ASSUME ALL INPUTS ARE DATAFRAMES
    # Solves A in the equation Y = A*X
    # using the Frobenius Norm which is the matrix version of the Ordinary Least Squares
    # We iterate through the number of Principal Components
    Xreg_train = np.asmatrix(Xreg_train)
    Yreg_train = np.asmatrix(Yreg_train)
    Xreg_test = np.asmatrix(Xreg_test)
    Yreg_test = np.asmatrix(Yreg_test)
    U, S, Vh = np.linalg.svd(Xreg_train)
    U = np.asmatrix(U)
    V = np.asmatrix(Vh.T)
    Ntrain = np.prod(Yreg_train.shape)
    Ntest = np.prod(Yreg_test.shape)
    err_train = np.zeros(len(S))
    err_test = np.zeros(len(S))
    E_train = Yreg_train
    E_test = Yreg_test
    # Aest = np.asmatrix(np.zeros((Yreg_train.shape[0],Xreg_train.shape[0])))
    for iPC in range(0,len(S)): # iPC indicating the instantaneous principal component
        E_train = E_train - Yreg_train * V[:, iPC] * U[:, iPC].T * Xreg_train / S[iPC]
        E_test = E_test - Yreg_train * V[:, iPC] * U[:, iPC].T * Xreg_test/ S[iPC]
        err_train[iPC] = np.linalg.norm(E_train,'fro')**2/Ntrain
        err_test[iPC] = np.linalg.norm(E_test,'fro')**2/Ntest
    J = err_train + err_test
    if PLOT_ON:
        plt.figure()
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Error Metric')
        plt.plot(err_train)
        plt.plot(err_test)
        plt.plot(J)
        plt.title('MSE with highest monomial order ' + str(n_mon))
        plt.show()
    nPC_opt = np.where(J == J.min())[0]
    Aopt = Yreg_train * V[:,0:nPC_opt[0]] * np.linalg.inv(np.diag(S[0:nPC_opt[0]])) * U[:,0:nPC_opt[0]].T
    return Aopt,J.min(),nPC_opt

def Init_Cond_OLS_SVD_solve(Yin,Xin,test_split,valid_split,PLOT_ON=False):
    # INPUTS/OUTPUTS ARE MATRICES - Each column is a datapoint
    # Solves A in the equation Y = A*X
    # using the Frobenius Norm which is the matrix version of the Ordinary Least Squares
    # We iterate through the number of Principal Components
    # Split Data With Validation Data
    X_tr, Xreg_test, Y_tr, Yreg_test = train_test_split_vertical_data(Xin, Yin,test_split)
    Xreg_train,Xreg_valid,Yreg_train,Yreg_valid = train_test_split_vertical_data(X_tr, Y_tr, valid_split)
    U, S, Vh = np.linalg.svd(Xreg_train)
    U = np.asmatrix(U)
    V = np.asmatrix(Vh.T)
    Ntrain = np.prod(Yreg_train.shape)
    Ntest = np.prod(Yreg_valid.shape)
    err_train = np.zeros(len(S))
    err_test = np.zeros(len(S))
    E_train = Yreg_train
    E_test = Yreg_valid
    # Aest = np.asmatrix(np.zeros((Yreg_train.shape[0],Xreg_train.shape[0])))
    for iPC in range(0,len(S)): # iPC indicating the instantaneous principal component
        E_train = E_train - Yreg_train * V[:, iPC] * U[:, iPC].T * Xreg_train / S[iPC]
        E_test = E_test - Yreg_train * V[:, iPC] * U[:, iPC].T * Xreg_test/ S[iPC]
        err_train[iPC] = np.linalg.norm(E_train,'fro')**2/Ntrain
        err_test[iPC] = np.linalg.norm(E_test,'fro')**2/Ntest
    J = err_train + err_test
    if PLOT_ON:
        plt.figure()
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Mean Squared Error')
        plt.plot(err_train)
        plt.plot(err_test)
        plt.plot(J)
        plt.title('Optimal Fit with both training and test data')
        plt.show()
    nPC_opt = np.where(J == J.min())[0]
    print('Optimal Number of Principal Components: ',nPC_opt)
    Aopt = Yreg_train * V[:,0:nPC_opt[0]] * np.linalg.inv(np.diag(S[0:nPC_opt[0]])) * U[:,0:nPC_opt[0]].T
    return Aopt#, J.min(), nPC_opt

def OLS_SVD_solve3(Y,X,PLOT_ON=False):
    # We solve for the model of the form Y = X*A using
    # X,Y - input matrices
    # A - output matrix
    # using the Frobenius Norm which is the matrix version of the Ordinary Least Squares
    # We iterate through the number of Principal Components
    U, S, Vh = np.linalg.svd(X)
    U = np.asmatrix(U)
    V = np.asmatrix(Vh.T)
    N = np.prod(Y.shape)
    err = np.zeros(len(S))
    E_train = Y
    for iPC in range(0,len(S)): # iPC indicating the instantaneous principal component
        E_train = E_train - X * V[:, iPC] * U[:, iPC].T * Y / S[iPC]
        err[iPC] = np.linalg.norm(E_train,'fro')**2/N
    J = err
    if PLOT_ON:
        plt.figure()
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Mean Squared Error')
        plt.plot(err)
        plt.plot(J)
        plt.title('Training Data Fit')
        plt.show()
    nPC_opt = np.where(J == J.min())[0]
    Aopt = V[:,0:nPC_opt[0]] * np.linalg.inv(np.diag(S[0:nPC_opt[0]])) * U[:,0:nPC_opt[0]].T  * Y
    # Aopt =1
    return Aopt#,J.min(),nPC_opt

def STEP_2_Regressor_Form(dict_data,nReg):
    Yreg= pd.DataFrame()
    Xreg = pd.DataFrame()
    for key in dict_data.keys():
        Xdata = pd.DataFrame.from_dict(dict_data[key])
        for i in range(Xdata.shape[1] - 1):
            Xreg = pd.concat([Xreg,Xdata.iloc[0:nReg,i]],axis=1)
            Yreg = pd.concat([Yreg,Xdata.iloc[0:nReg,i+1]],axis=1)
    return Yreg,Xreg

def HigherOrderMonomials(ls_Xinput,n_order): # Takes a list as an input and outputs another list
    ls_x_out=[]
    for i in range(1,n_order+1):
        ls_high_order_tuple = list(cwb(ls_Xinput,i))
        [ls_x_out.append(np.prod(ls_high_order_tuple[j])) for j in range(len(ls_high_order_tuple))]
    return ls_x_out

def STEP_1_Init(dict_data,n):
    dict_Xreg = {key: {} for key in dict_data.keys()}  # Creating an empty dictionary with the same keys as that of the training set
    for key in dict_data.keys():
        i = 0
        iter = 0
        while ((i + n) <= len(dict_data[key])):
            dict_Xreg[key][iter] = dict_data[key][i:i + n]
            i = i + n
            iter = iter + 1
    return dict_Xreg

def CJ_DMD2(train_dict,test_dict,valid_dict,n_LinReg_vec,n_mon_vec):
    n_mon_max = np.max(n_mon_vec)
    LAMBDA = 0
    COST ={}
    ERR={}
    VALIDATION_ERR={}
    N_PARAM={}
    A_candidate={}
    for nLinReg in n_LinReg_vec:
        print(nLinReg)
        COST[nLinReg]={}
        ERR[nLinReg] = {}
        VALIDATION_ERR[nLinReg] = {}
        N_PARAM[nLinReg] = {}
        A_candidate[nLinReg] = {}
        # INITIALIZATION
        # STEP 1 - Construct the extended observables
        dict_Xreg_train = STEP_1_Init(train_dict, nLinReg)
        dict_Xreg_test = STEP_1_Init(test_dict, nLinReg)
        dict_Xreg_valid = STEP_1_Init(valid_dict, nLinReg)
        for key in dict_Xreg_train.keys():
            for iter in dict_Xreg_train[key].keys():
                dict_Xreg_train[key][iter] = HigherOrderMonomials(dict_Xreg_train[key][iter],n_mon_max)
        for key in dict_Xreg_test.keys():
            for iter in dict_Xreg_test[key].keys():
                dict_Xreg_test[key][iter] = HigherOrderMonomials(dict_Xreg_test[key][iter],n_mon_max)
        for key in dict_Xreg_valid.keys():
            for iter in dict_Xreg_valid[key].keys():
                dict_Xreg_valid[key][iter] = HigherOrderMonomials(dict_Xreg_valid[key][iter],n_mon_max)
        n_mon_terms = [0]
        for i in range(1,n_mon_max+1):
            n_mon_terms.append(int(fctl(nLinReg+i-1)/fctl(nLinReg-1)/fctl(i)))
            n_mon_terms[i] = n_mon_terms[i] + n_mon_terms[i - 1]
        for n_mon in n_mon_vec:
            # STEP 2 - Formulate the Regressors - Iterative Fashion
            Yreg_train, Xreg_train = STEP_2_Regressor_Form(dict_Xreg_train, n_mon_terms[n_mon])
            Yreg_test, Xreg_test = STEP_2_Regressor_Form(dict_Xreg_test,n_mon_terms[n_mon])
            # STEP 3 - Solve using Ordinary Least Squares
            A,err2,nPC = OLS_SVD_solve(Yreg_train,Xreg_train,Yreg_test,Xreg_test,1,False)
            X0_train = {key: dict_Xreg_train[key][0][0:n_mon_terms[n_mon]] for key in train_dict.keys()}
            X0_test = {key: dict_Xreg_test[key][0][0:n_mon_terms[n_mon]] for key in test_dict.keys()}
            X0_valid = {key: dict_Xreg_valid[key][0][0:n_mon_terms[n_mon]] for key in valid_dict.keys()}
            plt_title = 'Fit of Training Set with ' + str(nLinReg) + ' Initial Regressors and upto monomial order ' + str(n_mon)
            MSE_TRAIN = plot_fit(train_dict,  X0_train, A, nLinReg,plt_title,False)
            plt_title = 'Fit of Test Set with ' + str(nLinReg) + ' Initial Regressors and upto monomial order ' + str(n_mon)
            MSE_TEST = plot_fit(test_dict, X0_test, A, nLinReg, plt_title,False)
            plt_title = 'Fit of Validation Set with ' + str(nLinReg) + ' Initial Regressors and upto monomial order ' + str(n_mon)
            MSE_VALID = plot_fit(valid_dict, X0_valid, A, nLinReg, plt_title,False)
            ERR[nLinReg][n_mon] = MSE_TRAIN + MSE_TEST
            VALIDATION_ERR[nLinReg][n_mon] = MSE_VALID
            N_PARAM[nLinReg][n_mon] = np.prod(A.shape)
            COST[nLinReg][n_mon] = MSE_TRAIN + MSE_TEST + LAMBDA * np.prod(A.shape)
            A_candidate[nLinReg][n_mon] = A
            print(A.shape)
    print('===========================================================================================')
    print('             ********************  ESTIMATION STATISTICS  ********************             ')
    print('===========================================================================================')
    print('Number of Parameters')
    print(pd.DataFrame.from_dict(N_PARAM))
    print('===========================================================================================')
    print('Error')
    print(pd.DataFrame.from_dict(ERR))
    print('===========================================================================================')
    print('Validation Error')
    print(pd.DataFrame.from_dict(VALIDATION_ERR))
    print('===========================================================================================')

    # Finding the optimal function
    nLin_opt = list(ERR.keys())[0]
    nmon_opt = list(ERR[nLin_opt].keys())[0]
    err_min = ERR[nLin_opt][nmon_opt]
    for nL in ERR.keys():
        for n_mon in ERR[nL].keys():
            if err_min > ERR[nL][n_mon]:
                err_min = ERR[nL][n_mon]
                nLin_opt = nL
                nmon_opt = n_mon
    Aopt = A_candidate[nLin_opt][nmon_opt]
    print('|| Optimal Parameters ||')
    print('========================')
    print('Optimal Number of Linear Regressors = ',nLin_opt)
    print('Optimal Number of Highest Order Monomial = ',nmon_opt)
    print('===========================================================================================')
    dict_Xreg_train = STEP_1_Init(train_dict, nLin_opt)
    dict_Xreg_test = STEP_1_Init(test_dict, nLin_opt)
    dict_Xreg_valid = STEP_1_Init(valid_dict, nLin_opt)
    X0_train = {key: HigherOrderMonomials(dict_Xreg_train[key][0],nmon_opt) for key in train_dict.keys()}
    X0_test = {key: HigherOrderMonomials(dict_Xreg_test[key][0],nmon_opt) for key in test_dict.keys()}
    X0_valid = {key: HigherOrderMonomials(dict_Xreg_valid[key][0],nmon_opt) for key in valid_dict.keys()}
    plt_title = 'Optimal Fit of Training Set with ' + str(nLin_opt) + ' Initial Regressors and \n upto monomial order ' + str(nmon_opt)
    plot_fit(train_dict, X0_train, Aopt, nLin_opt, plt_title, True,8,'PPutida_Training_Fit')
    plt_title = 'Optimal Fit of Test Set with ' + str(nLin_opt) + ' Initial Regressors and \n upto monomial order ' + str(nmon_opt)
    plot_fit(test_dict, X0_test, Aopt, nLin_opt, plt_title, True,13,'PPutida_Test_Fit')
    plt_title = 'Optimal Fit of Validation Set with ' + str(nLin_opt) + ' Initial Regressors and \n upto monomial order ' + str(nmon_opt)
    plot_fit(valid_dict, X0_valid, Aopt, nLin_opt, plt_title, True,13,'Pputida_Validation_Fit')
    plt.show()
    return Aopt,X0_train,X0_test,X0_valid

def plot_data(train_data,test_data,valid_data):
    legend_entry = []
    plt.figure()
    for keys in train_data.keys():
        plt.plot(list(np.arange(0,len(train_data[keys]))*33/60),train_data[keys])
        legend_entry.append('Train ' + keys)
    for keys in test_data.keys():
        plt.plot(list(np.arange(0,len(test_data[keys]))*33/60),test_data[keys],Linewidth=2,Linestyle=':')
        legend_entry.append('*Test ' + keys)
    for keys in valid_data.keys():
        plt.plot(list(np.arange(0,len(valid_data[keys]))*33/60),valid_data[keys], Linewidth=2, Linestyle='--')
        legend_entry.append('**Validation ' + keys)
    # plt.xlabel('Time [Hrs]')
    # plt.ylabel('OD600')
    plt.legend(legend_entry,ncol=3,prop={'size': 6})
    plt.savefig('/Users/shara/Desktop/Vnat_GrowthCurveAnalysis/PPutida_Wells.svg')
    plt.show()
    return

def plot_fit(data,X0_data,A,nLinReg,title,PLOT_ON=False, ftsiz=10,filname=''):
    # Plotting training set fit
    Ntrain = len(data)
    nrows = int(np.ceil(Ntrain ** 0.5))
    ncols = int(np.floor(Ntrain/nrows))
    # nrows = 3
    # ncols = 3
    if PLOT_ON:
        f, ax = plt.subplots(nrows,ncols, sharex=True, sharey=True)
        plt.rcParams['ytick.labelsize'] = 7
        plt.rcParams['xtick.labelsize'] = 7
        # plt.suptitle(title)
    i = 0
    ERR_TRAIN = 0
    row_i = 0
    col_i = 0
    for keys in data.keys():
        i = i + 1
        Xdata = data[keys]
        X = np.asmatrix(X0_data[keys]).T
        Xpred = [X[i, 0] for i in range(nLinReg)]
        for j in range(np.floor_divide(len(Xdata), nLinReg)):
            X = A * X
            [Xpred.append(X[i, 0]) for i in range(nLinReg)]
        if PLOT_ON:
            ax[row_i,col_i].plot(list(np.arange(0,len(Xdata),4)*33/60),Xdata[0:len(Xdata):4],'.',Linewidth=2)
            ax[row_i,col_i].plot(list(np.arange(0,len(Xdata))*33/60),Xpred[0:len(Xdata)])
            ax[row_i, col_i].legend(title=keys,fontsize=7,loc='best')
            col_i = col_i + 1
            if col_i >= ncols:
                row_i = row_i + 1
                col_i = 0
            if row_i >= nrows:
                row_i = nrows-1
                col_i = ncols-1
                ax[row_i, col_i].cla()
                ax[row_i, col_i].plot([], '.', Linewidth=2)
                ax[row_i, col_i].plot([])
                ax[row_i, col_i].axis('off')
                ax[row_i, col_i].legend(['Observed', 'Predicted'], fontsize=ftsiz, loc='best')
                break
        err_train_current = np.array(Xdata) - np.array(Xpred[0:len(Xdata)])
        ERR_TRAIN = ERR_TRAIN + np.linalg.norm(err_train_current,2)**2
    MSE = ERR_TRAIN/len(data)/len(data[list(data.keys())[0]])
    if PLOT_ON:
        # ax[row_i, col_i].cla()
        # ax[row_i, col_i].plot([], '.', Linewidth=2)
        # ax[row_i, col_i].plot([])
        # ax[row_i, col_i].axis('off')
        # ax[row_i, col_i].legend(['Observed', 'Predicted'], fontsize=ftsiz, loc='best')
        print('Hi')
        plt.savefig('/Users/shara/Desktop/Vnat_GrowthCurveAnalysis/'+filname+'1.svg')
    print(MSE)
    return MSE

def print_all(MP_dict,disp_dict_key):
    plt.figure()
    for key in disp_dict_key:
        plt.plot(MP_dict[key])
    # plt.legend(lengend_entry)
    plt.xlabel('Time Index')
    plt.ylabel('OD600')
    plt.title('Growth Rate of Bacteria')
    plt.show()

def plot_Growth(MP):
    time_max_i=[]
    for item in list(MP.columns):
        max_pt = np.max(MP[item][1:640])
        for i in range(640):
            if MP[item][i] == max_pt:
                break
        time_max_i.append(i)
    f, ax = plt.subplots(8, 12, sharex=True, sharey=True)
    tmax = np.max(time_max_i)
    print(tmax*3)
    dict_MP={}
    for item in list(MP.columns):
        MP[item][tmax:] = np.nan
        # print(list(MP[item][1:tmax]))
        dict_MP[item] = list(MP[item][1:tmax])
    i = 0
    for row in range(8):
        for col in range(12):
            ax[row, col].plot(MP.iloc[:, i])
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            l = ax[row, col].legend([],fontsize=0.01,loc='best')
            # l.set_title(time_max_i[i],prop={'size':6})
            i = i + 1
    # plt.savefig('/Users/shara/Desktop/Vnat_GrowthCurveAnalysis/Growth_with_time.svg')
    plt.show()
    # pd_MP_out = pd.DataFrame.from_dict(dict_MP)
    return dict_MP

def TwoD_SerialDil():
    # INPUTS
    # 1 --- A matrix of solution volume
    # 2 --- A matrix of Glucose concentration before mixing
    # 3 --- A matrix of Casein concentration before mixing
    # 4 --- Volume transfer between wells
    # 5 ---

    # DEFAULT INPUTS
    mat_premix_Glu_conc = np.zeros([8, 12])
    mat_premix_Glu_conc[:, 0] = 125
    mat_premix_Cas_conc = np.zeros([8, 12])
    mat_premix_Cas_conc[0, :] = 125
    mat_premix_soln_vol = np.ones([8, 12]) * 75
    mat_premix_soln_vol[:, 0] = 150
    mat_premix_soln_vol[0, :] = 150
    mat_premix_soln_vol[0, 0] = 225
    VOLUME_TRANSFER = 75

    # Class Definition
    class MP_well:
        def __init__(self, sol_vol, C_conc, G_conc):
            self.soln_vol = sol_vol  # [mL]
            self.Cas_conc = C_conc  # [g/L]
            self.Glu_conc = G_conc  # [g/L]

        def mix_with_well(self, vol_transfer, x):
            # Some amount of volume from this well is being transferred to another well of the same class
            if (vol_transfer > self.soln_vol):
                print('Not Enough Volume to transfer!')
                exit()
            self.soln_vol = self.soln_vol - vol_transfer
            x.Cas_conc = (self.Cas_conc * vol_transfer + x.Cas_conc * x.soln_vol) / (vol_transfer + x.soln_vol)
            x.Glu_conc = (self.Glu_conc * vol_transfer + x.Glu_conc * x.soln_vol) / (vol_transfer + x.soln_vol)
            x.soln_vol = x.soln_vol + vol_transfer
            return x
        def remove_from_well(self, vol_remove):
            self.soln_vol = self.soln_vol - vol_remove
    # Function Definitions
    def mix_vert(well, VOLUME_TRANSFER):
        # MIX VERTICAL
        for row_i in range(7):
            for col_i in range(12):
                well[12 * (row_i + 1) + col_i] = well[12 * row_i + col_i].mix_with_well(VOLUME_TRANSFER,
                                                                                        well[12 * (row_i + 1) + col_i])
        # REMOVE FROM LAST ROW
        row_i = 7
        for col_i in range(12):
            well[12 * row_i + col_i].remove_from_well(VOLUME_TRANSFER)
        return well
    def mix_horz(well, VOLUME_TRANSFER):
        # MIX HORIZONTAL
        for row_i in range(8):
            for col_i in range(11):
                well[12 * row_i + col_i + 1] = well[12 * row_i + col_i].mix_with_well(VOLUME_TRANSFER,
                                                                                      well[12 * row_i + col_i + 1])
        # REMOVE FROM LAST COLUMN
        col_i = 11
        for row_i in range(8):
            well[12 * row_i + col_i].remove_from_well(VOLUME_TRANSFER)
        return well


    # VECTORIZE THE MATRICES
    vec_premix_Glu_conc = list(np.reshape(mat_premix_Glu_conc, (-1)))
    vec_premix_Cas_conc = list(np.reshape(mat_premix_Cas_conc, (-1)))
    vec_premix_soln_vol = list(np.reshape(mat_premix_soln_vol, (-1)))
    well = []
    for i in range(96):
        well.append(MP_well(vec_premix_soln_vol[i], vec_premix_Cas_conc[i], vec_premix_Glu_conc[i]))
    well = mix_vert(well, VOLUME_TRANSFER)
    well = mix_horz(well, VOLUME_TRANSFER)

    # OUTPUT
    # 1 --- Table listing out the  well number as columns and the rows indicating Casein conc, Glucose conc and Well Volume
    Cas_after = []
    Glu_after = []
    soln_vol_after = []
    for i in range(96):
        Cas_after.append(well[i].Cas_conc/2)
        Glu_after.append(well[i].Glu_conc/2)
        soln_vol_after.append(well[i].soln_vol*2)
    # Cas_after = np.array(Cas_after)[np.newaxis]
    # Cas_after = Cas_after.reshape((8, 12))
    # soln_vol_after = np.array(soln_vol_after)[np.newaxis]
    # soln_vol_after = soln_vol_after.reshape((8, 12))
    Input_Init = pd.DataFrame({'Casein':Cas_after,'Glucose':Glu_after,'Volume_solution':soln_vol_after})
    return Input_Init

def WELLData_Structuring(filename,OD600_MEDIA,DOWN_SAMPLE_FACTOR,NO_SAMPLES,del_key,del_row_key,del_col_key):
    # ------------------------------------------------------------------------
    # DATA STRUCTURING
    # ------------------------------------------------------------------------
    # The file from the microplate reader is parsed using main.py and the useful data is converted into a .csv file
    # .csv ---> pandas ---> dictionary file with keys
    MP = pd.read_csv(filename)  # MP - Micro Plate
    MP = MP.iloc[:, 2:MP.shape[1]] - OD600_MEDIA
    INPUT_INTER = TwoD_SerialDil()
    dict_INPUT = {}
    i = -1
    # INPUT
    for items in list(MP.columns):
        i = i + 1
        dict_INPUT[items] = {'Casein': INPUT_INTER.iloc[i, 0] * INPUT_INTER.iloc[i, 2] * 1e-3,
                             'Glucose': INPUT_INTER.iloc[i, 1] * INPUT_INTER.iloc[i, 2] * 1e-3}  # UNITS: [mg]
    pd_INPUT = pd.DataFrame.from_dict(dict_INPUT)
    # DELETE THE DATA THAT IS NOT REQUIRED
    del_key = set(del_key) # To add only the unique entries
    for items in del_row_key:
        for col in range(1,13):
            del_key.add(items+str(col))
    for items in del_col_key:
        row ='A'
        for c in range(8):
            del_key.add(row+items)
            row = chr(ord(row)+1)
    for item in del_key:
        del MP[item]
        del pd_INPUT[item]
    dict_MP = MP.to_dict('list')  # MP_dict - Dictionary of microplate reader data with keys
    # DOWNSAMPLING THE DATA
    for keys in dict_MP.keys():
        dict_MP[keys] = [np.mean(dict_MP[keys][i:i + DOWN_SAMPLE_FACTOR]) for i in
                         range(0, len(dict_MP[keys]), DOWN_SAMPLE_FACTOR)]  # DOWNSAMPLE
        dict_MP[keys] = dict_MP[keys][0:NO_SAMPLES]
    return dict_MP,dict_INPUT


def BUILD_NN_Model_InitCond():
    from keras.models import Sequential
    from keras.layers import Dense
    # Build the neural net
    model = Sequential()
    model.add(Dense(3, input_dim=3, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(12, activation='linear'))
    return model


def train_test_split_vertical_data(Xin,Yin,test_size_val):
    # Inputs and Outputs both have individual columns as a data points
    if test_size_val ==0:
        X_test=np.asmatrix([[]])
        Y_test=np.asmatrix([[]])
        X_train = Xin
        Y_train = Yin
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(Xin.T, Yin.T, test_size=test_size_val)
        X_train = X_train.T
        X_test = X_test.T
        Y_train = Y_train.T
        Y_test = Y_test.T
    return X_train,X_test,Y_train,Y_test

