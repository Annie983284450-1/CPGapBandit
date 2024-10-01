import numpy as np
import os
from os import listdir
from os.path import isfile,join
from os import walk
import matplotlib.pyplot as plt
import random
import pandas as pd 
import seaborn as sns
import matplotlib.ticker as ticker 

if __name__ == '__main__':
    start_test = 0
    start_nosepsis_train = 0
    start_sepsis_train = 0
    num_test_pat = 500
    num_train_sepsis_pat = 1000
    num_train_nosepsis_pat = 3000
    # experts = ['ridge', 'rf', 'xgb', 'lasso']
    # experts = ['ridge', 'rf', 'xgb']
    experts = ['ridge', 'rf']
    # experts = ['ridge' , 'xgb']
    itrial = 0
    method = 'Ensemble'
    final_result_path='../cpbanditsepsis_experiements'+'/static_refitting'+'/Results'+'('+f'test{num_test_pat},train{num_train_sepsis_pat}_{num_train_nosepsis_pat}'+str(experts)+')'
    test_set = np.load('../cpbanditsepsis_experiements//Data/test_set.npy')
    test_set =  test_set[start_test:start_test+ num_test_pat]

    f_name = ''
    for i, expert in enumerate(experts):
        if i==0:
            f_name = f_name+expert
        else:
            f_name = f_name+'_'+expert
    alpha_ls = np.linspace(0.05,0.25,5)
    for patient_id in test_set[0:1]:
        f_dat_path = os.path.join(final_result_path+'/dats',f_name)
        for alpha in alpha_ls:
            
            cp_dat_path = f_dat_path+'/itrial#'+str(itrial)+'/alpha='+str(alpha)+'/'+str(method)
            curr_pat_df = pd.read_csv(cp_dat_path+'/'+str(patient_id)+'.csv')
            winner_tmp = curr_pat_df['winner_tmp'] 
            regret_tmp = np.zeros(len(winner_tmp))
            i = 0
            for winner in winner_tmp:
                regret_tmp[i] = abs(curr_pat_df[f'{winner}_predictions'][i]-curr_pat_df['hours2sepsis'][i])/500
                i = i+1
            print(regret_tmp)
                # print(winner)
            print(len(curr_pat_df))
            # print(curr_pat_df.columns)
            # print(curr_pat_df.head())

                   
    