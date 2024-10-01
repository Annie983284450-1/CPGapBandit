import numpy as np, os, os.path, sys, warnings
import pandas as pd
from sklearn.metrics import mean_squared_error

def standardize_reward(hours2sepsis, Y_upper, Y_lower, rmse_max, rmse_min):
    # get the RMSE
    Y_rmse_upper = mean_squared_error(hours2sepsis, Y_upper, squared=False)
    Y_rmse_lower = mean_squared_error(hours2sepsis, Y_lower, squared=False)
    standardize_rmse_upper = (Y_rmse_upper-rmse_min)/(rmse_max)
    standardize_rmse_lower = (Y_rmse_lower-rmse_min)/(rmse_max)
    reward_upper = max(1-standardize_rmse_upper,1-standardize_rmse_lower)
    reward_lower = min(1-standardize_rmse_upper,1-standardize_rmse_lower)
    return reward_lower, reward_upper

#  get the reward  = 1- standardized_rmse during the whole ICU stay
def get_UCB_LCB(curr_pat_df, Y_upper, Y_lower, rmse_max, rmse_min ):
    hours2sepsis = curr_pat_df['hours2sepsis']
    LCB, UCB = standardize_reward(hours2sepsis,Y_upper, Y_lower, rmse_max, rmse_min)
    return UCB, LCB

def get_reward_bounds(curr_pat_df, Y_upper, Y_lower, rmse_max, rmse_min):
    bounds_df = pd.DataFrame(columns=['UCB_reward', 'LCB_reward'])
    hours2sepsis = curr_pat_df['hours2sepsis'] 
    hours2sepsis = hours2sepsis.reset_index(drop=True)
    prev_hours = 3
    after_hours = - 12
    if curr_pat_df['SepsisLabel'].sum() != 0:
        IsSeptic = False
    else:
        IsSeptic = True
    for i in range(len(Y_upper)):
        if not IsSeptic: # this means all hour2sepsis == max_hours, so hours2sepsis[i] is for sure > Y_upper[i]
            if abs(Y_upper[i] -hours2sepsis[i])<= prev_hours:
                UCB_reward = 1
            else:
                UCB_reward = 0
            if abs(Y_lower[i] -hours2sepsis[i])<= prev_hours:
                LCB_reward = 1
            else:
                LCB_reward = 0
        else: #septic 
            if Y_upper[i] - hours2sepsis[i] >  prev_hours:
                UCB_reward = -1
            elif prev_hours >= (Y_upper[i] - hours2sepsis[i])>=after_hours:
                UCB_reward = 1 - abs( Y_upper[i] - hours2sepsis[i])/(abs(after_hours))
            else: # Y_upper[i] - hours2sepsis[i] < after_hours
                # alert too early no reward
                UCB_reward = 0
            if Y_lower[i] - hours2sepsis[i] >  prev_hours:
                LCB_reward = -1
            elif prev_hours >= (Y_lower[i] - hours2sepsis[i])>=after_hours:
                LCB_reward = 1 - abs(Y_upper[i] - hours2sepsis[i])/(abs(after_hours))
            else: # Y_upper[i] - hours2sepsis[i] < after_hours
                LCB_reward = 0
        '''
        bounds_df['UCB_reward'].iloc[i] = max(LCB_reward, UCB_reward)
        bounds_df['LCB_reward'].iloc[i] = min(LCB_reward, UCB_reward)
        IndexError: iloc cannot enlarge its target object
        you are trying to assign a value to an index in a pandas DataFrame or Series 
        that does not exist using the .iloc indexer. Unlike Python lists, 
        pandas DataFrames and Series cannot dynamically grow in size 
        when you assign a value to a non-existent index.

        '''
        bounds_df.loc[i, 'UCB_reward'] = max(LCB_reward, UCB_reward)
        bounds_df.loc[i, 'LCB_reward'] = min(LCB_reward, UCB_reward)
    UCB_avg = bounds_df['UCB_reward'].mean()
    LCB_avg = bounds_df['LCB_reward'].mean()

    return UCB_avg, LCB_avg

def get_regrets(curr_pat_df, predictions):
    hours2sepsis = curr_pat_df['hours2sepsis']  # ground truth
    hours2sepsis = hours2sepsis.reset_index(drop=True)
    prev_hours = 3
    after_hours = - 12
    if curr_pat_df['SepsisLabel'].sum()!= 0:
        IsSeptic = False
    else:
        IsSeptic = True
    for i in range(len(hours2sepsis)):
        if not IsSeptic: # this means all hour2sepsis == 480, so hours2sepsis[i] is for sure > Y_upper[i]. [not necesarrily]
            if abs(hours2sepsis[i] - predictions[i])<= prev_hours:
                reward = 1
            else:
                reward = 0
        else: #septic 
            if predictions[i] - hours2sepsis[i] >  prev_hours:
                reward = -1
            elif prev_hours >= (predictions[i] - hours2sepsis[i])>=after_hours:
                reward = 1 - abs(predictions[i] - hours2sepsis[i])/(abs(after_hours))
            else: # Y_upper[i] - hours2sepsis[i] < after_hours
                # alert too early no reward
                reward = 0
    return 1-reward
 
def get_absolute_error_V2(curr_pat_df, Y_upper, Y_lower, rmse_max, rmse_min):
    hours2sepsis = curr_pat_df['hours2sepsis']          
    upper_array = np.array(Y_upper)
    lower_array = np.array(Y_lower)
    hours_array = np.array(hours2sepsis)
    upper_ae = abs(upper_array - hours_array)/(rmse_max-rmse_min)
    lower_ae = abs(lower_array - hours_array)/(rmse_max-rmse_min)
    for i in range(len(upper_ae)):
        if upper_ae[i] > 1:
            print(f'upper_ae[{i}] = {upper_ae[i]}')
            print(f'upper_array[{i}] = {upper_array[i]}')
        if lower_ae[i] > 1:
            print(f'lower_ae[{i}] = {lower_ae[i]}')
            print(f'lower_array[i] = {lower_array[i]}')
            print(f'rmse_max = {rmse_max}')
        if lower_ae[i] > 1 or upper_ae[i] > 1:
            print(f'@*%$$$$$    upper_ae: {upper_ae}')
            print(f'@*%$$$$$    lower_ae: {lower_ae}')
    return upper_ae, lower_ae

def get_absolute_error(curr_pat_df, Y_upper, Y_lower, max_hours):
        
    rmse_min = 0
    rmse_max = max_hours
                                                                
         
    hours2sepsis = curr_pat_df['hours2sepsis']          
    upper_array = np.array(Y_upper)
    lower_array = np.array(Y_lower)
    hours_array = np.array(hours2sepsis)
    upper_ae = abs(upper_array - hours_array)/(rmse_max-rmse_min)
    lower_ae = abs(lower_array - hours_array)/(rmse_max-rmse_min)
    for i in range(len(upper_ae)):
        if upper_ae[i] > 1:
            print(f'upper_ae[{i}] = {upper_ae[i]}')
            print(f'upper_array[{i}] = {upper_array[i]}')
        if lower_ae[i] > 1:
            print(f'lower_ae[{i}] = {lower_ae[i]}')
            print(f'lower_array[i] = {lower_array[i]}')
            print(f'rmse_max = {rmse_max}')
        if lower_ae[i] > 1 or upper_ae[i] > 1:
            print(f'@*%$$$$$    upper_ae: {upper_ae}')
            print(f'@*%$$$$$    lower_ae: {lower_ae}')
    return upper_ae, lower_ae
 

def get_UCB_LCB_avg(curr_pat_df, Y_upper, Y_lower, max_hours):
    rmse_min = 0
    rmse_max = max_hours
    bounds_df = pd.DataFrame(columns=['UCB_reward', 'LCB_reward'])
    hours2sepsis = curr_pat_df['hours2sepsis'] 
    hours2sepsis = hours2sepsis.reset_index(drop=True)
    prev_hours = 3
    after_hours = - 12
    if curr_pat_df['SepsisLabel'].sum() != 0:
        IsSeptic = False
    else:
        IsSeptic = True
    for i in range(len(Y_upper)):
        if not IsSeptic: # this means all hour2sepsis == max_hours, so hours2sepsis[i] is for sure > Y_upper[i]
            if abs(Y_upper[i] -hours2sepsis[i])<= prev_hours:
                UCB_reward = 1
            else:
                UCB_reward = 0
            if abs(Y_lower[i] -hours2sepsis[i])<= prev_hours:
                LCB_reward = 1
            else:
                LCB_reward = 0
        else: #septic 
            if Y_upper[i] - hours2sepsis[i] >  prev_hours:
                UCB_reward = -1
            elif prev_hours >= (Y_upper[i] - hours2sepsis[i])>=after_hours:
                UCB_reward = 1 - abs( Y_upper[i] - hours2sepsis[i])/(abs(after_hours))
            else: # Y_upper[i] - hours2sepsis[i] < after_hours
                # alert too early no reward
                UCB_reward = 0
            if Y_lower[i] - hours2sepsis[i] >  prev_hours:
                LCB_reward = -1
            elif prev_hours >= (Y_lower[i] - hours2sepsis[i])>=after_hours:
                LCB_reward = 1 - abs(Y_upper[i] - hours2sepsis[i])/(abs(after_hours))
            else: # Y_upper[i] - hours2sepsis[i] < after_hours
                LCB_reward = 0
        '''
        bounds_df['UCB_reward'].iloc[i] = max(LCB_reward, UCB_reward)
        bounds_df['LCB_reward'].iloc[i] = min(LCB_reward, UCB_reward)
        IndexError: iloc cannot enlarge its target object
        you are trying to assign a value to an index in a pandas DataFrame or Series 
        that does not exist using the .iloc indexer. Unlike Python lists, 
        pandas DataFrames and Series cannot dynamically grow in size 
        when you assign a value to a non-existent index.

        '''
        bounds_df.loc[i, 'UCB_reward'] = max(LCB_reward, UCB_reward)
        bounds_df.loc[i, 'LCB_reward'] = min(LCB_reward, UCB_reward)
    UCB_avg = bounds_df['UCB_reward'].mean()
    LCB_avg = bounds_df['LCB_reward'].mean()

    return UCB_avg, LCB_avg