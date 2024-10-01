# relabel hours2sepsis

import pandas as pd 
import time
import os
sepsis_full = pd.read_csv('fully_imputed.csv')
sepsis_full.drop(['hours2sepsis'], axis=1, inplace=True)

max_hours = 48

prev_hours = 12
after_hours = 36
start = time.time()
list_patients = []
file_path = 'fully_imputed_{prev_hours}_{after_hours}_win{max_hours}.csv'
if os.path.exists(file_path):
    # Delete the file
    os.remove(file_path)
    print(f"File '{file_path}' has been deleted.")
else:
    print(f"File '{file_path}' does not exist.")

with open(file_path, 'a') as f:

    for patient_id in sepsis_full['pat_id'].unique():
            # dding .copy() makes curr_pat_df an independent DataFrame, not a view of sepsis_full
        curr_pat_df_l = sepsis_full[sepsis_full['pat_id']==patient_id].copy()
        curr_pat_df_l.reset_index(drop=True, inplace=True)
        len_stay = len(curr_pat_df_l)
        
        if len_stay > max_hours:
            if curr_pat_df_l['SepsisLabel'].sum() != 0: # septic patient
                sepsis_onset_time_ori = curr_pat_df_l['SepsisLabel'].eq(1).idxmax()
                print(f'Patient {patient_id} has sepsis at #{sepsis_onset_time_ori} and len_stay == {len_stay} hours.')
                start_index = max(0, sepsis_onset_time_ori - after_hours)  # Ensure start_index is not less than 0
                end_index = min(len(curr_pat_df_l), sepsis_onset_time_ori + after_hours)  # Ensure end_index does not exceed DataFrame length
                if start_index  == 0:
                    curr_pat_df = sepsis_full[sepsis_full['pat_id']==patient_id].head(max_hours).copy()
                    # curr_pat_df.reset_index(drop=True, inplace=True)
                if end_index == len(curr_pat_df_l):
                    curr_pat_df = sepsis_full[sepsis_full['pat_id']==patient_id].tail(max_hours).copy()
                    # curr_pat_df.reset_index(drop=True, inplace=True)
                if start_index != 0 and end_index != len(curr_pat_df_l):
                    curr_pat_df = curr_pat_df_l.iloc[start_index:end_index]

                curr_pat_df.reset_index(drop=True, inplace=True)
                # Slice the DataFrame
                # curr_pat_df = curr_pat_df.iloc[start_index:end_index]
                if len(curr_pat_df) != max_hours:
                    print(f'len(curr_pat_df) ! = {max_hours}')
                    print(f'start_index: {start_index}')
                    print(f'end_index: {end_index}')
                    break
                if start_index>=end_index:
                    print(f'start index < end_index')
                    print(f'start_index: {start_index}')
                    print(f'end_index: {end_index}')
                    break
                sepsis_onset_time = curr_pat_df['SepsisLabel'].eq(1).idxmax()
                hours2sepsis_septic = list(reversed(range(sepsis_onset_time+1))) + [0] * (len(curr_pat_df)-sepsis_onset_time-1)
                if len(hours2sepsis_septic) != len(curr_pat_df):
                    print(f'len(hours2sepsis_septic) != len(curr_pat_df), len(hours2sepsis_septic) = {len(hours2sepsis_septic)}')
                    break
                curr_pat_df['hours2sepsis'] = hours2sepsis_septic
                
                # print(f'{len(curr_pat_df)}')
                    # break
            else: # non-septic patient
                curr_pat_df = sepsis_full[sepsis_full['pat_id']==patient_id].head(max_hours).copy()
                curr_pat_df.reset_index(drop=True, inplace=True)
                curr_pat_df['hours2sepsis'] = max_hours+1
            
        else: #len_stay < 48hrs
            curr_pat_df = sepsis_full[sepsis_full['pat_id']==patient_id].head(max_hours).copy()
            curr_pat_df.reset_index(drop=True, inplace=True)
            if curr_pat_df['SepsisLabel'].sum() == 0: # non-septic patient
                curr_pat_df['hours2sepsis'] = max_hours+1
            else: # septic patient
                sepsis_onset_time = curr_pat_df['SepsisLabel'].eq(1).idxmax()
                hours2sepsis_septic = list(reversed(range(sepsis_onset_time+1))) + [0] * (len(curr_pat_df)-sepsis_onset_time-1)
                if len(hours2sepsis_septic) != len(curr_pat_df):
                    print(f'len(hours2sepsis_septic) != len(curr_pat_df), len(hours2sepsis_septic) = {len(hours2sepsis_septic)}')
                    break
                curr_pat_df['hours2sepsis'] = hours2sepsis_septic
        # I am not sure what to do with the nonseptic patients, in order to have less bias as possible, we set the SepisLabel = 0 as max_hours+1, i.e., 49 hours herein
        curr_pat_df.to_csv(f, header=f.tell() == 0, index=False)
# concatenated_df = pd.concat(list_patients)
# concatenated_df.to_csv(f'fully_imputed_relabelled_{max_hours}.csv', index= False)
print(f'@*%$$$$$    time: {time.time()-start}')