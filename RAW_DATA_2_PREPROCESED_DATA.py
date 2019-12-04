import re
import pandas as pd

# filename = "SHARA_Vnat_Exp1_5thSept2019_1.txt"
# filename = "SHARA_Vnat_Exp3_10thSept2019_1.txt"
# filename = "SHARA_Vnat_Exp4_18thOct2019_1.txt"
# filename = "SHARA_strain_PPutida_medium_R2A_GrowthCurve1.txt"
filename = "SHARA_strain_PPutida_medium_M9min_GrowthCurve1.txt"

RawData = [];
with open(filename, "r") as f:  # 'r' is used to specify Read Mode
    reader = f.readlines()
    linecount = len(reader)
    # Skip everything upto the point where our useful data is available
    tim_ct = 0
    for i in range(linecount):
        iteritem = re.split(r'[\t\s]\s*', reader[i])  # re package is used just for this purpose
        if (iteritem[0] == "Time"):#if (iteritem[0] == "Time"):
            tim_ct = tim_ct + 1
            if tim_ct == 2:
                istop = i
                break
    iteritem.remove(iteritem[2])
    # iteritem.remove(iteritem[2])
    iteritem.remove(iteritem[-1])
    # Creating the pandas Dataframe
    RawData.append(iteritem)
    for i in range(istop + 1, linecount):
        iteritem = re.split(r'[\t\s]\s*', reader[i])  # re package is used just for this purpose
        iteritem.remove(iteritem[-1])
        if (iteritem[0] in ["","Results", "0:00:00"]):
            break;
        RawData.append(iteritem)
    Tb = pd.DataFrame(RawData[1:], columns=RawData[0])
f.close()
Tb = Tb.drop(columns='T∞',axis =1) # Dropping the T∞ column
Tb.to_csv('PPutida_M9min_Expt1.csv')

# Saving as a '.mat' file
# import scipy.io as sio
# sio.savemat('Expt1.mat',Tb)
