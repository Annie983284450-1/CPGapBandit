    
import os
import numpy as np
import pandas as pd   
import sys 
# read sepsis dataset  
class get_params_prob():
    def __init__(self, experts, 
                 num_test_pat, 
                 num_train_sepsis_pat,  
                 start_test, 
                 start_nosepsis_train, 
                 start_sepsis_train, 
                 max_u_tp,
                 min_u_fn,
                 u_fp,
                 u_tn):
        self.experts = experts
        self.num_test_pat = num_test_pat
        self.num_train_sepsis_pat = num_train_sepsis_pat
        self.start_test = start_test
        self.start_nosepsis_train = start_nosepsis_train
        self.start_sepsis_train = start_sepsis_train
        self.max_u_tp = max_u_tp
        self.min_u_fn = min_u_fn
        self.u_fp = u_fp
        self.u_tn = u_tn

    def get_train_test_set(self):
        # num_test_pat = 250
        # num_train_sepsis_pat = 500
        # num_train_nosepsis_pat = num_train_sepsis_pat 

        # start_test = 0
        # start_nosepsis_train = 0
        # start_sepsis_train = 0
        num_test_pat =  self.num_test_pat
        num_train_sepsis_pat = self.num_train_sepsis_pat
        num_train_nosepsis_pat = self.num_train_sepsis_pat 

        start_test = self.start_test
        start_nosepsis_train =  self.start_nosepsis_train
        start_sepsis_train =    self.start_sepsis_train
        test_set_psv = np.load('./data/test_set.npy')
        test_set  = [filename.replace('.psv', '') for filename in test_set_psv]

        # print('test_set[0:5]:')
        # print(test_set[0:5])
        test_set =  test_set[start_test:start_test+num_test_pat]

        train_sepsis = np.load('./data/train_sepsis.npy')
        train_nosepsis = np.load('./data/train_nosepsis.npy')
        train_sepsis = train_sepsis[start_sepsis_train:start_sepsis_train+num_train_sepsis_pat]
        train_nosepsis = train_nosepsis[start_nosepsis_train:start_nosepsis_train+num_train_nosepsis_pat]
        train_set_psv = np.concatenate((train_sepsis, train_nosepsis), axis=0)

        train_set  = [filename.replace('.psv', '') for filename in train_set_psv]
        original_train_size_pat = len(train_set)

        # print('train_set[0:5]:')
        # print(train_set[0:5])

        sepsis_full = pd.read_csv('./data/fully_imputed.csv')
        sepsis_full.drop(['HospAdmTime'], axis=1, inplace=True)

        final_result_path = '../cpbandit_results_prob/' +   f'test{num_test_pat}_train_{num_train_sepsis_pat*2}'  +f'_max_u_tp={self.max_u_tp}_min_u_fn={self.min_u_fn}_u_fp={self.u_fp}_u_tn={self.u_tn}/'
        if not os.path.exists(final_result_path):
            os.makedirs(final_result_path)
        print(f'final_result_path: {final_result_path}')
        if start_test !=0:
            sys.exit('!!!!!!!!! start_test should be 0')
            # X_train = np.load(final_result_path +'./X_train_merged.npy', X_train_merged)
            # Y_train = np.load(final_result_path + '/Y_train_merged.npy', Y_train_merged)
        else:
            # train_sepis_df = sepsis_full[sepsis_full['pat_id'].isin(train_sepsis)]
            # train_nosepis_df = sepsis_full[sepsis_full['pat_id'].isin(train_nosepsis)]
            # test_set_df = sepsis_full[sepsis_full['pat_id'].isin(test_set)]
            train_set_df = sepsis_full[sepsis_full['pat_id'].isin(train_set)]

            # test_set_df = test_set_df.reset_index(drop=True)     
            train_set_df_x = train_set_df.drop(columns = ['pat_id','hours2sepsis', 'SepsisLabel'])
            # train_set_df_y = train_set_df['hours2sepsis']
            train_set_df_y = train_set_df['SepsisLabel']
            # do not use inplace =True, otherwise train_set_df_x/y and test_set_df_x/y will become nonetype
            # then we cannot use to_numpy()
            # the original training dataset before experts selection
            X_train = train_set_df_x.to_numpy(dtype='float', na_value=np.nan)
            Y_train = train_set_df_y.to_numpy(dtype='float', na_value=np.nan)
        
        print(f'X_train shape: {X_train.shape}')
        # print(f'Y_train shape: {Y_train.shape}')

        return X_train, Y_train, test_set, final_result_path, sepsis_full, original_train_size_pat
