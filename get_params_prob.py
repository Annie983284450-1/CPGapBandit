    
import os
import numpy as np
import pandas as pd   
import sys 
import math
# read sepsis dataset  
class get_params_prob():
    def __init__(self, experts, 
                 num_test_sepsis_pat, 
                 num_train_sepsis_pat,  
                 start_test, 
                 start_nosepsis_train, 
                 start_sepsis_train,
                 B
                #  max_u_tp,
                #  min_u_fn,
                #  u_fp,
                #  u_tn
                 ):
        self.experts = experts
        self.num_test_sepsis_pat = num_test_sepsis_pat
        self.num_train_sepsis_pat = num_train_sepsis_pat
        self.num_train_nosepsis_pat =  num_train_sepsis_pat 
        self.start_test = start_test
        self.start_nosepsis_train = start_nosepsis_train
        self.start_sepsis_train = start_sepsis_train
        self.B = B
        # self.max_u_tp = max_u_tp
        # self.min_u_fn = min_u_fn
        # self.u_fp = u_fp
        # self.u_tn = u_tn

    def get_train_test_set(self):
        # num_test_pat = 250
        # num_train_sepsis_pat = 500
        

        # start_test = 0
        # start_nosepsis_train = 0
        # start_sepsis_train = 0

        # num_test_pat =  self.num_test_pat
        # num_train_sepsis_pat = self.num_train_sepsis_pat
        # num_train_nosepsis_pat = self.num_train_sepsis_pat 

        # start_test = self.start_test
        # start_nosepsis_train =  self.start_nosepsis_train
        # start_sepsis_train =    self.start_sepsis_train


        # a total of 24,532 populations was divided into septic/no-septic (1606/2,2926) patients
        # development dateset (34,285 patients, 2,492 septic & 31,793 non-septic)
        # validation dataset (6,051 patients, 440 septic & 5,611 non-septic)
        #~~~~~~~~~~~~~~Testing set~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ratio = math.floor(22926/1606)
        test_sepsis = np.load('./data/test_sepsis.npy')
        test_nosepsis = np.load('./data/test_nosepsis.npy')
        test_sepsis = test_sepsis[self.start_test: min(self.start_test+self.num_test_sepsis_pat, len(test_sepsis))]
        test_nosepsis = test_nosepsis[self.start_test:min(self.start_test+self.num_test_sepsis_pat*ratio, len(test_nosepsis))]
        print(f'test_sepsis: {len(test_sepsis)}')
        print(f'test_nosepsis: {len(test_nosepsis)}')

        test_set_psv = np.concatenate((test_sepsis, test_nosepsis), axis=0) 
        test_set = [filename.replace('.psv', '') for filename in test_set_psv]
        #~~~~~~~~~~~~~~Training set~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        train_sepsis = np.load('./data/train_sepsis.npy')
        train_nosepsis = np.load('./data/train_nosepsis.npy')
        train_sepsis = train_sepsis[self.start_sepsis_train:min(self.start_sepsis_train+self.num_train_sepsis_pat, len(train_sepsis))]
        train_nosepsis = train_nosepsis[self.start_nosepsis_train:min(self.start_nosepsis_train+self.num_train_nosepsis_pat,len(train_nosepsis))]
        print(f'train_sepsis: {len(train_sepsis)}')
        print(f'train_nosepsis: {len(train_nosepsis)}')
    
        train_set_psv = np.concatenate((train_sepsis, train_nosepsis), axis=0)
        train_set  = [filename.replace('.psv', '') for filename in train_set_psv]
        original_train_size_pat = len(train_set) # original training set size before refitting
 

        sepsis_full = pd.read_csv('./data/fully_imputed.csv')
        sepsis_full.drop(['HospAdmTime'], axis=1, inplace=True)

        final_result_path = '../cpbandit_results_prob/'+f'testSeptic{self.num_test_sepsis_pat}_trainSeptic{self.num_train_sepsis_pat}_B{self.B}'
        if not os.path.exists(final_result_path):
            os.makedirs(final_result_path)
        print(f'final_result_path: {final_result_path}')
        if self.start_test !=0:
            sys.exit('!!!!!!!!! start_test should be 0')
        else:
            train_set_df = sepsis_full[sepsis_full['pat_id'].isin(train_set)]
            # test_set_df = test_set_df.reset_index(drop=True)     
            train_set_df_x = train_set_df.drop(columns = ['pat_id','hours2sepsis', 'SepsisLabel'])
            train_set_df_y = train_set_df['SepsisLabel']
            # do not use inplace =True, otherwise train_set_df_x/y and test_set_df_x/y will become nonetype
            # then we cannot use to_numpy()
            # the original training dataset before experts selection
            X_train = train_set_df_x.to_numpy(dtype='float', na_value=np.nan)
            Y_train = train_set_df_y.to_numpy(dtype='float', na_value=np.nan)
        
        print(f'X_train shape: {X_train.shape}')
        return X_train, Y_train, test_set, final_result_path, sepsis_full, original_train_size_pat
