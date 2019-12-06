import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GrowthCurveAnalysis as gra

def get_SamplingTime(T2,T1):
    from re import split
    t2_str = split(':', T2)
    t2 = int(t2_str[0]) * 3600 + int(t2_str[1]) * 60 + int(t2_str[2])
    t1_str = split(':', T1)
    t1 = int(t1_str[0]) * 3600 + int(t1_str[1]) * 60 + int(t1_str[2])
    Ts_sec = t2 - t1
    return Ts_sec


filename = 'PPutida_R2A_Expt1.csv'
# filename = 'PPutida_M9min_Expt1.csv'
# filename = 'PFluor_R2A_Expt1.csv'
# filename = 'PFluor_M9min_Expt1.csv'
df_DATA = pd.read_csv('PREPROCESSED_DATA/'+filename)
Ts_sec = get_SamplingTime(df_DATA.iloc[1,1],df_DATA.iloc[0,1])
df_DATA = df_DATA.iloc[:,2:]
LAG_THRESHOLD = 10 # In Percentage
STATIONARY_THRESHOLD = 70 # In Percentage

LAG_TIME = {}
GROWTH_TIME = {}
MAX_GROWTH = {}
for items in df_DATA.columns:
    ls_welldata = list(df_DATA[items])
    minval = ls_welldata[0]
    maxval = np.max(ls_welldata)
    lag_thres = LAG_THRESHOLD/100*(maxval-minval) + minval
    stationary_thres = STATIONARY_THRESHOLD / 100 * (maxval - minval) + minval
    MAX_GROWTH[items] = maxval
    for i in range(len(ls_welldata)):
        if ls_welldata[i] >= lag_thres:
            LAG_TIME[items] = Ts_sec*i
            break
    for i in range(len(ls_welldata)):
        if ls_welldata[i] >= stationary_thres:
            GROWTH_TIME[items] = Ts_sec * i
            break
df_AllWellStats = pd.DataFrame({'Lag Time':pd.Series(LAG_TIME),'Growth Time':pd.Series(GROWTH_TIME),'Maximum Growth':pd.Series(MAX_GROWTH)})

df_IMPULSE_INPUT_GMS = gra.TwoD_SerialDil()
df_IMPULSE_INPUT_GMS.index = list(df_DATA.columns)
# # Lets do a spline fit or a smoothing operation
# # Let's do a moving average filter
# MAF_WINDOW = 10
# dict_MAFilt_DATA = {}
# for items in  df_DATA.columns:
#     dict_MAFilt_DATA[items]={}
#     ls_welldata = list(df_DATA[items])
#     # for i in range(400): #range(len(df_DATA[items])-MAF_WINDOW):
#     #     dict_MAFilt_DATA[items][i] = np.mean(ls_welldata[i:i+MAF_WINDOW])
#     Ts_sec = Ts_sec*MAF_WINDOW
#     for i in range(int(400/MAF_WINDOW)): #range(len(df_DATA[items])-MAF_WINDOW):
#         dict_MAFilt_DATA[items][i] = np.mean(ls_welldata[i*MAF_WINDOW:(i+1)*MAF_WINDOW])
# df_MAFilt_DATA = pd.DataFrame.from_dict(dict_MAFilt_DATA)
#
#
# f,ax = plt.subplots(8,12,sharex=True,sharey=True)
# col_no = -1
# for i in range(8):
#     for j in range(12):
#         col_no +=1
#         ax[i,j].plot(df_DATA.iloc[:, col_no])
#         ax[i, j].plot(df_MAFilt_DATA.iloc[:, col_no])
# f.show()
# f.savefig('fig.svg')
#
# f,ax = plt.subplots(8,12,sharex=True,sharey=True)
# col_no = -1
# for i in range(8):
#     for j in range(12):
#         col_no +=1
#         ax[i,j].plot(np.array(df_MAFilt_DATA.iloc[1:,col_no]) - np.array(df_MAFilt_DATA.iloc[0:-1,col_no]))
# f.show()
#
# plt.figure()
# for c in range(len(df_MAFilt_DATA.columns)):
#     plt.plot(np.array(df_MAFilt_DATA.iloc[1:,c]) - np.array(df_MAFilt_DATA.iloc[0:-1,c]))
# plt.show()


# Downsampling by taking the mean
MEAN_WINDOW = 20
N_samples = 400
Ts_sec = Ts_sec * MEAN_WINDOW
dict_MAFilt_DATA = {}
for items in  df_DATA.columns:
    dict_MAFilt_DATA[items]={}
    ls_welldata = list(df_DATA[items])
    for i in range(int(N_samples/MEAN_WINDOW)): #range(len(df_DATA[items])-MAF_WINDOW):
        dict_MAFilt_DATA[items][i] = np.mean(ls_welldata[i*MEAN_WINDOW:(i+1)*MEAN_WINDOW])
x_index = np.arange(0,N_samples,MEAN_WINDOW)
x_index = x_index[0:len(dict_MAFilt_DATA[items])]
df_MAFilt_DATA = pd.DataFrame.from_dict(dict_MAFilt_DATA)


f,ax = plt.subplots(8,12,sharex=True,sharey=True)
col_no = -1
for i in range(8):
    for j in range(12):
        col_no +=1
        ax[i,j].plot(df_DATA.iloc[0:800, col_no])
        ax[i, j].plot(x_index,df_MAFilt_DATA.iloc[:, col_no])
f.show()

dict_DATA_DERIVATIVE ={}
plt.figure()
for keys in df_MAFilt_DATA.columns:
    dict_DATA_DERIVATIVE[keys] = list(np.diff(df_MAFilt_DATA[keys])/Ts_sec)
    plt.plot(dict_DATA_DERIVATIVE[keys])
plt.show()
df_DATA_DERIVATIVE = pd.DataFrame.from_dict(dict_DATA_DERIVATIVE)

f,ax = plt.subplots(8,12,sharex=True,sharey=True)
col_no = -1
for i in range(8):
    for j in range(12):
        col_no +=1
        ax[i,j].plot(df_DATA_DERIVATIVE.iloc[:,col_no])
f.show()

# Getting the maximal growth rate statistics
dict_MAX_GROWTH_RATE = {}
max_val = 0
for keys in df_DATA_DERIVATIVE.columns:
    # if (keys == 'D3') :#| (keys =='C6'):
    #     continue
    if np.max(df_DATA_DERIVATIVE[keys])>max_val:
        max_val = np.max(df_DATA_DERIVATIVE[keys])
        max_key = keys
        max_Casein = df_IMPULSE_INPUT_GMS.loc[keys,'Casein']
        max_Glucose = df_IMPULSE_INPUT_GMS.loc[keys,'Glucose']
        max_Volume = df_IMPULSE_INPUT_GMS.loc[keys, 'Volume_solution']
    dict_MAX_GROWTH_RATE[keys] = np.max(df_DATA_DERIVATIVE[keys])

print('==============================')
print('Maximal Growth Statstics')
print('==============================')
print('The well with the maximum growth rate is : ', max_key)
print('Maximal Growth Rate : ', max_val*3600, ' OD/hr')
print('Corresponding  Casein mass: ', max_Casein,' mgs')
print('Corresponding  Glucose mass: ', max_Glucose,' mgs')
print('Well Volume: ', max_Volume,' mL')
print('==============================')


# Getting the minimal growth rate statistics
dict_MIN_GROWTH_RATE = {}
min_val = 100000
for keys in df_DATA_DERIVATIVE.columns:
    if np.max(df_DATA_DERIVATIVE[keys])<min_val:
        min_val = np.max(df_DATA_DERIVATIVE[keys])
        min_key = keys
        min_Casein = df_IMPULSE_INPUT_GMS.loc[keys,'Casein']
        min_Glucose = df_IMPULSE_INPUT_GMS.loc[keys,'Glucose']
        min_Volume = df_IMPULSE_INPUT_GMS.loc[keys, 'Volume_solution']
    dict_MIN_GROWTH_RATE[keys] = np.max(df_DATA_DERIVATIVE[keys])

print('==============================')
print('Minimal Growth Statstics')
print('==============================')
print('The well with the maximum growth rate is : ', min_key)
print('Maximal Growth Rate : ', min_val*3600, ' OD/hr')
print('Corresponding  Casein mass: ', min_Casein,' mgs')
print('Corresponding  Glucose mass: ', min_Glucose,' mgs')
print('Well Volume: ', min_Volume,' mL')
print('==============================')
