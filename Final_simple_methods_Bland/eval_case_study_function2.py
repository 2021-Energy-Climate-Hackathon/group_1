#
#
# functions to evaluate the case studies
#
#

import numpy as np
import matplotlib.pyplot as plt

def RMS_based_eval(timeseries_for_eval,field,case_no,method_str):

    ''''
    timeseries_for_eval [array] must be the appropriate length for case study evaluation 
    Case1 = 168 hours
    Case2 = 216 hours
    Case3 = 672 hours

    field [str] must either be 'T2m', 'speed10m' , or 'rsds'

    case no [str] must be either 1,2 or 3.

    method_str [str] what you'd like it referred to as.

    '''
    
    obs = np.loadtxt('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/case_studies/Case_' + case_no + '_' + field + '.dat')
    
    obs_date = np.load('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/case_studies/Case_' + case_no + '_date.npy')
    MAE = np.mean(np.abs(obs - timeseries_for_eval))
    RMS = np.sqrt(np.nanmean((obs - timeseries_for_eval)**2))
    
    R2 = 1 - ( (np.mean ( (obs - timeseries_for_eval)**2 )) / (np.mean ( (obs - np.mean(obs))**2 )) )
    
    P = np.corrcoef(obs, timeseries_for_eval)
    
    fig = plt.figure(figsize=(12,4))
    plt.plot(obs_date,obs,color='k',label='ERA5 obs')
    if field == 'T2m':
        plt.plot(obs_date,timeseries_for_eval,color='r',label=method_str)
        plt.ylabel('2m temperature ($^{o}$C)',fontsize=14)
        plt.title('Case study ' + case_no )
        plt.xlabel('MAE = ' + str(MAE) + ' , RMS = ' + str(RMS) + ' , R2 = ' + str(R2) + ' , PearsonC = ' + str(P))
    if field == 'speed10m':
        plt.plot(obs_date,timeseries_for_eval,color='b',label=method_str)
        plt.ylabel('10m wind speed (ms$^{-1}$)',fontsize=14)
        plt.title('Case study ' + case_no)
    if field == 'rsds':
        plt.plot(obs_date,timeseries_for_eval,color='gold',label=method_str)
        plt.ylabel('Surface shortwave radiation (Wm$^{-2}$)',fontsize=14)
        plt.title('Case study ' + case_no )
    plt.legend(frameon=False)
    plt.show()
    return([MAE, RMS, R2, P])
