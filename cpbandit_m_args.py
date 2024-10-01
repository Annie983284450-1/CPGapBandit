import time
import math
import pickle
# import utils_ECAD_journal as utils_ECAD
from scipy.stats import skew
import seaborn as sns
# import PI_class_EnbPI_journal as EnbPI
import PI_Sepsysolcp as EnbPI
import matplotlib
# matplotlib.use('TkAgg',force=True)
# matplotlib.use('Agg')
import os
if os.environ.get('DISPLAY','') == '':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN   # kNN detector
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
# from sklearn import svm
import utils_Sepsysolcp as util
from matplotlib.lines import Line2D  # For legend handles
# import calendar
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
 
from sklearn.preprocessing import MinMaxScaler
import itertools
import importlib
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from reward import get_UCB_LCB
from reward import get_absolute_error
from gap_b import gap_bandit 
import sys
# this will suppress all the error messages
# be cautious
# stderr = sys.stderr
# sys.stderr = open('logfile.log','w')
import tensorflow as tf
# sys.stderr = stderr
# tf.get_logger().setLevel('ERROR')
# # warnings.filterwarnings("ignore")
# # importlib.reload(sys.modules['PI_class_EnbPI_journal'])

import multiprocessing
import dill

multiprocessing.get_context().Process().Pickle = dill
# =============Read data and initialize parameters
class CPBandit:
    def __init__(self, experts, num_test_pat_win, num_train_sepsis_pat_win, num_train_nosepsis_pat_win):
        self.experts = experts
        self.k = len(experts)
        # maintain a list of upper bound value and a lower bound value for each arm
        # only include the reward when the arm is pulled
        self.UCB_avg = [0]*self.k
        self.LCB_avg = [0]*self.k
        # the upper and lower bound value for the current patient
        # with the current patient data only
        # we keep track of hourly UCB and LCB values for each expert
        self.UCB_hourly = [0]*self.k
        self.LCB_hourly = [0]*self.k   
        # maintain a list of number of times each arm is pulled
        self.Nk = [0]*self.k
        # Reward is the utilty function, i.e., 1-RMSE
        # self.rewards = []
        self.num_test_pat_win = num_test_pat_win
        self.num_train_sepsis_pat_win = num_train_sepsis_pat_win
        self.num_train_nosepsis_pat_win = num_train_nosepsis_pat_win
    def _start_game(self):
        '''
        num_test_pat = 10  
        num_train_sepsis_pat = 50 
        num_train_nosepsis_pat = 150 
        Total excution time ---  5 seconds ---
I
        num_test_pat = 100  
        num_train_sepsis_pat = 500 
        num_train_nosepsis_pat = 1500 
        Total excution time --- 2472.87833571434 seconds ---
        
        the time complexity is too high. So we need to do the refitting less frequently
        '''
        # read sepsis dataset
        # num_test_pat = 20
        # num_train_sepsis_pat = 20
        # num_train_nosepsis_pat = 40
        start_time = time.time()

        start_test = 0
        start_nosepsis_train = 0
        start_sepsis_train = 0

        # test_set = np.load('../cpbanditsepsis_experiements//Data/test_set.npy')
        # test_set =  test_set[start_test:start_test+self.num_test_pat]
        test_nosepsis = np.load("../cpbanditsepsis_experiements//Data/test_nosepsis.npy")
        test_sepsis = np.load("../cpbanditsepsis_experiements//Data/test_sepsis.npy")
        train_sepsis = np.load('../cpbanditsepsis_experiements//Data/train_sepsis.npy')
        train_nosepsis = np.load('../cpbanditsepsis_experiements//Data/train_nosepsis.npy')
        # train_sepsis = train_sepsis[start_sepsis_train:start_sepsis_train+self.num_train_sepsis_pat]
        # train_nosepsis = train_nosepsis[start_nosepsis_train:start_nosepsis_train+self.num_train_nosepsis_pat]
        # print(f'test_set:{test_set}')
        # print(f'train_sepsis:{train_sepsis}')
        # print(f'train_nosepsis:{train_nosepsis}')

        win_size = 8
        sepsis_full = pd.read_csv(f'../cpbanditsepsis_experiements/Data/fully_imputed_8windowed_max48_updated.csv')
    
        sepsis_train_wins = []
        nosepsis_train_wins = []
        test_wins = []
        test_septic_wins = []
        test_noseptic_wins = []
        num_test_pat_noseptic_win = math.floor(self.num_test_pat_win*12)
        
        for patient_winid in sepsis_full['pat_id'].unique():
            if patient_winid.split('_')[0]+'.psv' in train_sepsis:
                # print(patient_winid)
                sepsis_train_wins.append(patient_winid)
            elif patient_winid.split('_')[0]+'.psv' in train_nosepsis:
                nosepsis_train_wins.append(patient_winid)
            elif patient_winid.split('_')[0]+'.psv' in test_nosepsis:
                test_noseptic_wins.append(patient_winid)
            elif patient_winid.split('_')[0]+'.psv' in test_sepsis:
                test_septic_wins.append(patient_winid)
        sepsis_train_wins = sepsis_train_wins[start_sepsis_train:start_sepsis_train+self.num_train_sepsis_pat_win]
        nosepsis_train_wins = nosepsis_train_wins[start_nosepsis_train:start_nosepsis_train+self.num_train_nosepsis_pat_win]
        test_septic_wins =  test_septic_wins[start_test:start_test + self.num_test_pat_win]
        test_noseptic_wins =  test_noseptic_wins[start_test:start_test + num_test_pat_noseptic_win]
        test_wins = test_septic_wins + test_noseptic_wins
        
        # print(f'sepsis_train_wins:{sepsis_train_wins}')
        # print(f'nosepsis_train_wins:{nosepsis_train_wins}')
        # print(f'test_wins:{test_wins}')
        
      

        sepsis_full.drop(['HospAdmTime'], axis=1, inplace=True)
        final_result_path = '../cpbanditsepsis_experiements'+ f'/no_refitting_balanced_win{win_size}_updated'+'/Results'+'('+f'test{self.num_test_pat_win},train{self.num_train_sepsis_pat_win}_{self.num_train_nosepsis_pat_win}'+str(self.experts)+')'

        if not os.path.exists(final_result_path):
            os.makedirs(final_result_path)
        if not os.path.exists(final_result_path+'/dats'):
            os.makedirs(final_result_path+'/dats')
        if not os.path.exists(final_result_path+'/imgs'):
            os.makedirs(final_result_path+'/imgs')



        if start_test !=0:
            # X_train = np.load(final_result_path +'./X_train_merged.npy', X_train_merged)
            # Y_train = np.load(final_result_path + '/Y_train_merged.npy', Y_train_merged)
            pass

        else:
            # train_sepis_df = sepsis_full[sepsis_full['pat_id'].isin(train_sepsis)]
            # train_nosepis_df = sepsis_full[sepsis_full['pat_id'].isin(train_nosepsis)]
            train_sepis_df = sepsis_full[sepsis_full['pat_id'].isin(sepsis_train_wins)]
            train_nosepis_df = sepsis_full[sepsis_full['pat_id'].isin(nosepsis_train_wins)]
            # test_set_df = sepsis_full[sepsis_full['pat_id'].isin(test_set)]
            # test_set_df = sepsis_full[sepsis_full['pat_id'].isin(test_wins)]
            train_set_df = pd.concat([train_sepis_df, train_nosepis_df], ignore_index=True)

            print(f'train_set_df: {train_set_df.head()}')
            # test_set_df = test_set_df.reset_index(drop=True)     
            # print(f'test_set_df: {test_set_df.head()}')
            train_set_df_x = train_set_df.drop(columns = ['pat_id','hours2sepsis'])
            train_set_df_y = train_set_df['hours2sepsis']
            # do not use inplace =True, otherwise train_set_df_x/y and test_set_df_x/y will become nonetype
            # then we cannot use to_numpy()
            # the original training dataset before experts selection
            X_train = train_set_df_x.to_numpy(dtype='float', na_value=np.nan)
            Y_train = train_set_df_y.to_numpy(dtype='float', na_value=np.nan)


        # # ==================Getting the conformal intervals......===================
        # initialze parameters
        data_name = 'physionet_sepsis'
        stride = 1
        miss_test_idx=[]
        tot_trial = 1        # usually B=30 can make sure every sample have LOO residual
        B = 25
        K = len(self.experts)
        alpha=0.1

        alpha_ls = np.linspace(0.05,0.25,5)
        # alpha_ls = [0.1]
        min_alpha = 0.0001
        max_alpha = 10

        if not os.path.exists(final_result_path+'/dats'):
            os.makedirs(final_result_path+'/dats')
        if not os.path.exists(final_result_path+'/imgs'):
            os.makedirs(final_result_path+'/imgs')
        f_name = ''
        for i, expert in enumerate(self.experts):
            if i==0:
                f_name = f_name+expert
            else:
                f_name = f_name+'_'+expert
        f_dat_path = os.path.join(final_result_path+'/dats',f_name)
        f_img_path = os.path.join(final_result_path+'/imgs', f_name)
        if not os.path.exists(f_dat_path):
            os.makedirs(f_dat_path)
        if not os.path.exists(f_img_path):
            os.makedirs(f_img_path)

        
        expert_idx = list(range(K))
        expert_dict = dict(zip(self.experts, expert_idx))
        print(f'expert dict: {expert_dict}')
     
        interval_namelist = [x+'_interval' for x in self.experts]
        coverage_namelist = [x+'_coverage' for x in self.experts]
        regret_namelist = [x+'_regret' for x in self.experts]
        UCB_namelist = ['UCB_avg_'+x for x in self.experts]
        LCB_namelist = ['LCB_avg_'+x for x in self.experts]
        final_columns = ['patient_id', 'alpha', 'itrial', 'method']
        final_columns.extend(interval_namelist)
        final_columns.extend(coverage_namelist)
        final_columns.extend(regret_namelist)
        final_columns.extend(UCB_namelist)
        final_columns.extend(LCB_namelist)
        final_columns.append('winner_avg')
        print(f'final columns: {final_columns}')

        # final_all_results_avg = pd.DataFrame(columns=final_columns)
        
                        
        predictions_namelist = [x+'_predictions' for x in self.experts]
        predictions_col = predictions_namelist

        UCB_namelist = ['UCB_hourly_'+x for x in self.experts]
        LCB_namelist = ['LCB_hourly_'+x  for x in self.experts]
        upper_namelist = ['upper_'+x for x in self.experts]
        lower_namelist = ['lower_'+x for x in self.experts]

        cp_col = ['patient_id', 'alpha', 'itrial', 'method']
        cp_col.extend(UCB_namelist)
        cp_col.extend(LCB_namelist)
        cp_col.extend(predictions_namelist)
        # cp_col.extend('hours2sepsis') no!!
        cp_col.append('hours2sepsis')
        cp_col.extend(upper_namelist)
        cp_col.extend(lower_namelist)
        cp_col.append('absolute_error')
        cp_col.append('winner_tmp')
        cp_col.append('winner_avg')
        cp_col.append('pulled_idx_tmp')

        # print(f'cp_col: {cp_col}')

        if 'ridge' in self.experts:
            ridge_f = RidgeCV(alphas=np.linspace(min_alpha, max_alpha, 10))
        if 'lasso' in self.experts:
            lasso_f = LassoCV(alphas=np.linspace(min_alpha, max_alpha, 10))
        if 'rf' in self.experts:
            rf_hyperparams = {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 10}
            # rf_hyperparams = {'n_estimators':10, 'criterion':'mse','bootstrap': False, 'max_depth': 2}

            rf_f = RandomForestRegressor(n_estimators=100, min_samples_split = 2, min_samples_leaf = 4, criterion='mse',  max_features = 'sqrt' , max_depth=10, n_jobs=-1)
        # if 'svr' in self.experts:
        #     svr_f = SVR()
        if 'xgb' in self.experts:
            xgb_hyperparams = {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
            xgb_f = XGBRegressor(subsample=xgb_hyperparams['subsample'],
                                                n_estimators=xgb_hyperparams['n_estimators'],
                                                max_depth=xgb_hyperparams['max_depth'],
                                                learning_rate=xgb_hyperparams['learning_rate'])
        if 'nn' in self.experts:
            # nn_f = util.keras_mod(seed=12345)
            # if self contains unpicklable objects, this can still cause issues. When you use instance methods with multiprocessing, the entire instance (self) gets pickled implicitly.
            # we can only parse a string or sth that is pickable if we want to multiprocessing instead of parsing the neural network model directly in self
            nn_f = 'nnet'

        methods  = ['Ensemble'] # conformal prediction methods
        
        
 
        num_pat_win_tested = 0
        num_pat_tested =0
        num_fitting = 0
        refit_step = 100
        # X_size = {}
        
        rmse_min = 0
        max_hours = 48       
        rmse_max = max_hours + 1         
        no_refit = True
        num_septic_windows = 0
        num_noseptic_windows = 0
        itrial = 0
        # for psv in test_set:
        # prev_win = test_wins[0]
        if num_pat_win_tested == 0:
            # Open in write mode to truncate the file
            with open(final_result_path+'/final_all_results_avg.csv', 'w') as f:
                pass  # Just opening in 'w' mode truncates the file

            
        for i in range(len(test_wins)):
            print(f'num_pat_win_tested:{num_pat_win_tested}')
        # for patient_id in test_wins:
            patient_id = test_wins[i]

            start_curr_pat_time =  time.time()
            print('\n\n')
            print(f'=======Processing patient {num_pat_tested}th patient: {patient_id}  ====================')
            if no_refit:
                if num_pat_win_tested == 0:
                    Isrefit = True
                else:
                    Isrefit = False      
            else:         
                if num_pat_tested % refit_step==0:
                    Isrefit = True
                    if num_pat_win_tested!=0:
                        # X_size['Old_X_Size'] = X_train.shape[0]
                        # X_train = X_train_merged
                        # Y_train = Y_train_merged
                        # X_size['New_X_Size'] = X_train.shape[0]
                        # print(f'Training Dataset Updated!!!!!!!!!!!!!!!!!!!!!')
                        # print(f'{X_size}')
                        

                        ## update the training dataset for refitting

                        # .... needs to be modified. How to balance the dynamically expanding dataset?
                        new_train_wins = sepsis_train_wins+ nosepsis_train_wins + test_wins[0:i-1]
                        train_sepis_df = sepsis_full[sepsis_full['pat_id'].isin(new_train_wins)].copy()
                        train_set_df_x = train_set_df.drop(columns = ['pat_id','hours2sepsis'])
                        train_set_df_y = train_set_df['hours2sepsis']
                        # do not use inplace =True, otherwise train_set_df_x/y and test_set_df_x/y will become nonetype
                        # then we cannot use to_numpy()
                        X_train = train_set_df_x.to_numpy(dtype='float', na_value=np.nan)
                        Y_train = train_set_df_y.to_numpy(dtype='float', na_value=np.nan)
                    if start_test!=0:
                        # Isrefit = False
                        pass  
                else:
                    Isrefit = False            

            curr_pat_df = sepsis_full[sepsis_full['pat_id']==patient_id].copy()
            if patient_id.split('_')[2] == 'septic':
                IsSeptic = True
            else:
                IsSeptic = False
            print(f'len(curr_pat_df): {len(curr_pat_df)}')
            if len(curr_pat_df) != win_size:
                print("WIndow size!=8!")
                sys.exit()  # Skip to the next iteration of the loop and abandon the window less than 8 hrs
                
                
            X_predict = curr_pat_df.drop(columns=['pat_id','hours2sepsis'])
            Y_predict = curr_pat_df['hours2sepsis']    

                # try different alphas every time
            curr_pat_predictions = pd.DataFrame(columns=predictions_col)
            np.random.seed(12345+ itrial)
            standardize_rmse_dict = {}
            cp_EnbPI_dict = {}
                    
            for expert in self.experts:
                cp_EnbPI = EnbPI.prediction_interval(locals()[f'{expert}_f'], X_train, X_predict, Y_train, Y_predict, final_result_path)
                cp_EnbPI.fit_bootstrap_models_online_multi(B, miss_test_idx, Isrefit, model_name = expert, max_hours = max_hours)
                predictions = cp_EnbPI.Ensemble_pred_interval_centers
                curr_pat_predictions[f'{expert}_predictions'] = predictions 
                rmse = mean_squared_error(Y_predict, predictions, squared=False)
                standardize_rmse_dict[f'{expert}'] = (rmse-rmse_min)/(rmse_max-rmse_min)
                cp_EnbPI_dict[f'{expert}'] = cp_EnbPI
                        
            print('======================**************============================')
            print(f'%%%%%%%%%%%  Calculating residuals and conformal prediction intervals for patient {patient_id} .......................')
                    # for alpha in alpha_ls:
            for alpha in alpha_ls:
                print(f'~~~~~~~~~~At trial # {itrial} and alpha={alpha}~~~~~~~~~~~~~~')
                for method in methods:
                    cp_dat_path = f_dat_path+'/itrial#'+str(itrial)+'/alpha='+str(alpha)+'/'+str(method)
                    if not os.path.exists(cp_dat_path):
                        os.makedirs(cp_dat_path)
    
                    curr_pat_cp = pd.DataFrame(columns=cp_col)
                    LCB_curr_dict = {}
                    UCB_curr_dict = {}
                    
                    coverage_dict = {}
                    regret_dict = {}
                    upper_ae_dict = {}
                    lower_ae_dict = {}
                            # new_row_all_avg is a dict storing the results summarizing all the result of each patient at the end of admission
                            # it is a row of the final_all_results_avg dataframe
                    new_row_all_avg = {'patient_id': patient_id, 'alpha': alpha, 'itrial': itrial, 'method': method}
                    if method == 'Ensemble':
                        for expert in self.experts:
                            curr_pat_cp[f'{expert}_predictions'] = curr_pat_predictions[f'{expert}_predictions']
                            PIs_df, results = cp_EnbPI_dict[f'{expert}'].run_experiments(alpha, stride, data_name, itrial,
                                                        true_Y_predict=[], get_plots=False, none_CP=False, methods=methods, max_hours = max_hours)
                            print(results)
                            coverage_dict[expert] = results.mean_coverage.values[0]
                            Y_upper = PIs_df['upper']
                            Y_lower = PIs_df['lower']

                                # if we cut the predictions to [0, 49] the problem is the coverage might be 0% for some patients, especially nonsepstic patients

                            Y_upper = [rmse_max if y>rmse_max else y for y in Y_upper]
                            Y_upper = [rmse_min if y<rmse_min else y for y in Y_upper]
                            Y_lower = [rmse_max if y>rmse_max else y for y in Y_lower]
                            Y_lower = [rmse_min if y<rmse_min else y for y in Y_lower] 
                            curr_pat_cp[f'upper_{expert}'] = Y_upper
                            curr_pat_cp[f'lower_{expert}'] = Y_lower
                                # upper_ae, lower_ae = get_absolute_error(curr_pat_df, Y_upper, Y_lower, rmse_max, rmse_min)
                            upper_ae, lower_ae = get_absolute_error(curr_pat_df, Y_upper, Y_lower, max_hours+1, rmse_min)
                            upper_ae_dict[f'{expert}'] = upper_ae
                            lower_ae_dict[f'{expert}'] = lower_ae
                                    # get the reward  = 1- standardized_rmse during the whole ICU stay
                                    # intermediate UCB and LCB value of current patient within 48 hours
                                    # self.UCB_curr, self.LCB_curr are two values, not list
                                # UCB_curr, LCB_curr = get_UCB_LCB(curr_pat_df, Y_upper, Y_lower, rmse_max, rmse_min)
                            
                            UCB_curr, LCB_curr = get_UCB_LCB(curr_pat_df, Y_upper, Y_lower, max_hours+1, rmse_min)

                                    # UCB_avg, LCB_avg = get_UCB_LCB(curr_pat_df, Y_upper, Y_lower)
                            UCB_curr_dict[f'{expert}'] = UCB_curr
                            LCB_curr_dict[f'{expert}'] = LCB_curr

                            if UCB_curr >1 or UCB_curr <0 or LCB_curr >1 or LCB_curr <0 or UCB_curr < LCB_curr:
                                print(f'@*%$$$$$    UCB/LCB_curr value errors!!!!')
                                print(f'@*%$$$$$      UCB_curr: { UCB_curr}')
                                print(f'@*%$$$$$     LCB_curr: {LCB_curr}')
                                break

                            print(f'[LCB_{expert}_curr, UCB_{expert}_curr]: [{LCB_curr}, {UCB_curr}]')
                            k_idx = expert_dict[f'{expert}']
                                    # update the empirical mean of UCB_avg and LCB_avg for each expert based on all historical data and the current patient data
                                    # UCB_avg_k = self.UCB_avg[k_idx]
                                    # LCB_avg_k = self.LCB_avg[k_idx]
                            self.UCB_avg[k_idx] = ( self.UCB_avg[k_idx] * num_pat_win_tested + UCB_curr) / (num_pat_win_tested+1)
                            self.LCB_avg[k_idx] = ( self.LCB_avg[k_idx]* num_pat_win_tested + LCB_curr) /(num_pat_win_tested+1)

                            if self.UCB_avg[k_idx]>1 or self.LCB_avg[k_idx] <0 or LCB_curr >1 or self.LCB_avg[k_idx]<0 or self.UCB_avg[k_idx]<self.LCB_avg[k_idx]:
                                print(f'@*%$$$$$    self.UCB/LCB_avg value errors!!!!')
                                print(f'@*%$$$$$      self.UCB_avg[k_idx]: {self.UCB_avg[k_idx]}')
                                print(f'@*%$$$$$     self.LCB_avg[k_idx]: {self.LCB_avg[k_idx]}')
                                break

                            curr_pat_cp[f'{expert}_predictions'] = curr_pat_predictions[f'{expert}_predictions']
                        
                            # pull the arm based on all the historial data in this arm
                            # since each expert can know the ground truth when the patient is discharged,
                            # so we can update the UCB_avg and LCB_avg for each expert every hour (self.U/LCB_curr) and every patient (self.U/LCB_avg)
                    print(f'Selecting the best expert on average for {patient_id} @@@@~~~')
                    pulled_arm_idx_avg = gap_bandit(self.UCB_avg, self.LCB_avg, self.k).pull_arm()
                    for expert in self.experts:
                                #the larger the standardized_rmse, the larger the regret
                                # regret_dict[f'{expert}'] = 1 - standardize_rmse_dict[f'{expert}']
                                # the regret is got after the patient is discharged
                                # it is an average value within 48 hours
                        regret_dict[f'{expert}'] = standardize_rmse_dict[f'{expert}']
                        new_row_all_avg[f'{expert}_interval'] = (LCB_curr_dict[expert], UCB_curr_dict[expert])
                        new_row_all_avg[f'{expert}_coverage'] = coverage_dict[expert]
                        new_row_all_avg[f'{expert}_regret'] = regret_dict[expert]
                        k_idx = expert_dict[f'{expert}']
                        new_row_all_avg[f'UCB_avg_{expert}'] = self.UCB_avg[k_idx] 
                        new_row_all_avg[f'LCB_avg_{expert}'] = self.LCB_avg[k_idx] 

                    new_row_all_avg['winner_avg'] = list(expert_dict.keys())[pulled_arm_idx_avg]
                    if not isinstance(new_row_all_avg, pd.DataFrame):
                        new_row_all_avg = pd.DataFrame([new_row_all_avg])
                        # Append to CSV file
                            
                    # if os.path.exists(final_result_path+'/final_all_results_avg.csv'):
                    #     if num_pat_win_tested == 0:
                    #         # Delete the file
                    #         os.remove(final_result_path+'/final_all_results_avg.csv')
                    #         print(f"File '{final_result_path+'/final_all_results_avg.csv'}' has been deleted.")
                    # else:
                    #     print(f"File '{final_result_path+'/final_all_results_avg.csv'}' does not exist.")

                    with open(final_result_path+'/final_all_results_avg.csv', 'a') as f:
                        # if num_pat_win_tested == 0:
                        #     f.truncate()
                        new_row_all_avg.to_csv(f, header=f.tell()==0, index=False)

                    print(f'Selecting the best expert on hourly basis for {patient_id} @@@@@@@~~~')
                    UCB_tmp = [0]*self.k
                    LCB_tmp = [0]*self.k
                    pulled_arms_tmp = []
                    selected_experts = []     
                    ae = [] 
                            # store temporary UCB and LCB values at each hour for  
                    for m in range(len(Y_predict)):
                        for expert in self.experts:
                            k_idx = expert_dict[expert]
                            upper_ae_k = upper_ae_dict[f'{expert}'] 
                            lower_ae_k = lower_ae_dict[f'{expert}']
                            UCB_tmp[k_idx] = max(1-upper_ae_k[m],1-lower_ae_k[m])
                            LCB_tmp[k_idx] = min(1-upper_ae_k[m],1-lower_ae_k[m])
                                # print(f'@*%$$$$$     UCB_tmp[k_idx]: { UCB_tmp[k_idx]}')
                                # print(f'@*%$$$$$     LCB_tmp[k_idx]: { LCB_tmp[k_idx]}')
                            self.UCB_hourly[k_idx] = (self.UCB_hourly[k_idx]*num_pat_win_tested  + UCB_tmp[k_idx] ) / (num_pat_win_tested+1)
                            self.LCB_hourly[k_idx] = (self.LCB_hourly[k_idx]*num_pat_win_tested  + LCB_tmp[k_idx] ) / (num_pat_win_tested+1)

                            if self.UCB_hourly[k_idx] >1 or self.UCB_hourly[k_idx] <0 or self.LCB_hourly[k_idx] >1 or self.LCB_hourly[k_idx] <0 or self.UCB_hourly[k_idx] < self.LCB_hourly[k_idx]:
                                print(f'@*%$$$$$    UCB/LCB_hourly value errors!!!!')
                                print(f'@*%$$$$$     UCB_hourly[k_idx]: {self.UCB_hourly[k_idx]}')
                                print(f'@*%$$$$$     LCB_hourly[k_idx]: {self.LCB_hourly[k_idx]}')
                                break

                                    # curr_pat_cp.loc[m, f'UCB_{expert}'] = UCB_tmp[k_idx]
                                    # curr_pat_cp.loc[m, f'LCB_{expert}'] = LCB_tmp[k_idx]
                            curr_pat_cp.loc[m, f'UCB_hourly_{expert}'] = self.UCB_hourly[k_idx] 
                            curr_pat_cp.loc[m, f'LCB_hourly_{expert}'] = self.LCB_hourly[k_idx]
                                # pull one arm on an hourly basis
                                # pulled_arm_idx_tmp = gap_bandit(UCB_tmp, LCB_tmp, self.k).pull_arm() 
                        pulled_arm_idx_tmp = gap_bandit(self.UCB_hourly, self.LCB_hourly, self.k).pull_arm() 
                            
                        selected_expert_tmp = list(expert_dict.keys())[pulled_arm_idx_tmp]
                            # absolute error for the selected expert at the current hour
                        ae_tmp = abs(curr_pat_cp[f'{selected_expert_tmp}_predictions'].iloc[m] - Y_predict.iloc[m])/500
                        ae.append(ae_tmp)
                        pulled_arms_tmp.append(pulled_arm_idx_tmp)
                        selected_experts.append(selected_expert_tmp)
                    curr_pat_cp['absolute_error'] = ae
                    curr_pat_cp['winner_tmp'] = selected_experts
                    curr_pat_cp['pulled_idx_tmp'] = pulled_arms_tmp
                        # the whole column is the same for the following variabels
                    curr_pat_cp['winner_avg'] = list(expert_dict.keys())[pulled_arm_idx_avg]
                    curr_pat_cp['patient_id'] = patient_id
                    curr_pat_cp['itrial'] = itrial
                    curr_pat_cp['method'] = method
                    curr_pat_cp['alpha'] = alpha
                        # do not forget .values so that the index is ignored. Y_predict's indexes do not start from 0. It is cut from the full dataframe containing all the patients
                    curr_pat_cp['hours2sepsis'] = Y_predict.values
                    curr_pat_cp.to_csv(f'{cp_dat_path}/{patient_id}.csv')
                # win_id = win_id + 1
                # maintain a balanceed training dataset
            if IsSeptic:
                num_septic_windows = num_septic_windows + 1
            else:
                num_noseptic_windows = num_noseptic_windows + 1
            print(f'num_septic_windows: {num_septic_windows}')
            print(f'num_noseptic_windows: {num_noseptic_windows}')

            # if not no_refit:
            #         # print(f'Updating the training dataset ..........')
            #     if num_noseptic_windows >= num_septic_windows:
            #         if IsSeptic:
            #             X_train_new_df = curr_pat_df.drop(columns = ['pat_id','hours2sepsis'])
            #             Y_train_new_df = curr_pat_df['hours2sepsis']
            #             X_train_new = X_train_new_df.to_numpy(dtype='float', na_value=np.nan)
            #             Y_train_new = Y_train_new_df.to_numpy(dtype='float', na_value=np.nan)
            #             if num_pat_tested ==0:
            #                 X_train_merged = np.append(X_train,X_train_new,axis=0)
            #                 Y_train_merged = np.append(Y_train,Y_train_new,axis=0)
            #             else:
            #                 X_train_merged = np.append(X_train_merged,X_train_new,axis=0)
            #                 Y_train_merged = np.append(Y_train_merged,Y_train_new,axis=0)
            #         else:
            #             pass
            #     else: # num_noseptic_windows < num_septic_windows
            #         if not IsSeptic:
            #             X_train_new_df = curr_pat_df.drop(columns = ['pat_id','hours2sepsis'])
            #             Y_train_new_df = curr_pat_df['hours2sepsis']
            #             X_train_new = X_train_new_df.to_numpy(dtype='float', na_value=np.nan)
            #             Y_train_new = Y_train_new_df.to_numpy(dtype='float', na_value=np.nan)
            #             if num_pat_tested ==0:
            #                 X_train_merged = np.append(X_train,X_train_new,axis=0)
            #                 Y_train_merged = np.append(Y_train,Y_train_new,axis=0)
            #             else:
            #                 X_train_merged = np.append(X_train_merged,X_train_new,axis=0)
            #                 Y_train_merged = np.append(Y_train_merged,Y_train_new,axis=0)
            #         else:
            #             pass
            # else:
            #     pass
                
            num_pat_win_tested = num_pat_win_tested + 1


            if i >=1:
                if test_wins[i-1].split('_')[0]!=patient_id.split('_')[0]:
                    num_pat_tested = num_pat_tested + 1
            
            if Isrefit:
                num_fitting = num_fitting+1            
            # updating dataset
                
            print(f'# {num_pat_tested} patients already tested! ......')
            print(f'#{num_septic_windows} septic windows and #{num_noseptic_windows} nonseptic windows have been processed!')
            print(f'\n')
            print(f'-------------------------------------------')
            print(f'~~~~~~Excution time for # {patient_id}: {time.time()-start_curr_pat_time} seconds~~~~~~')
            print('\n\n')
            print('========================================================')
        
        if not no_refit:
            pass
            # np.save(final_result_path+'/X_train_merged.npy', X_train_merged)
            # np.save(final_result_path+'/Y_train_merged.npy', Y_train_merged)
         
        print('========================================================')
        print('========================================================')
        # print(f'Test size: {len(test_set)}')
      
        print(f'Total excution time: {(time.time() - start_time)} seconds~~~~~~' )
        machine = 'ece-kl2313-01.ece.gatech.edu'
        with open(final_result_path+'/execution_info.txt', 'w') as file:
            file.write(f'Total excution time: {(time.time() - start_time)} seconds\n')
            file.write(f'No. of bootstraps: {B}\n')
            file.write(f'num_test_pat_win = {self.num_test_pat_win}\n')
            file.write(f'num_train_sepsis_pat_win = {self.num_train_sepsis_pat_win}\n')
            file.write(f'num_train_nosepsis_pat_win = {self.num_train_nosepsis_pat_win}\n')
            file.write(f'win_size = {win_size}hrs\n')
            file.write(f'num_septic_windows = {num_septic_windows} added\n')
            file.write(f'num_noseptic_windows = {num_noseptic_windows} added\n')
            file.write(f'tot_trial = {tot_trial}\n')
            file.write(f'refit_step = {refit_step}\n') 
            file.write(f'No of experts = {self.k}\n')
            file.write(f'Experts = {list(expert_dict.keys())}\n')
            file.write(f'The models have been fitted for {num_fitting} times.\n')
            file.write(f'Machine: {machine}\n')
            file.write(f'Multiprocessor: True\n')
            # file.write(f'start_test = {start_test}\n')
            # file.write(f'start_nosepsis_train = {start_nosepsis_train}\n')
            # file.write(f'start_sepsis_train = {start_sepsis_train}\n') 
            if 'xgb' in self.experts:
                file.write(f'xgb hyperparameters: {xgb_hyperparams}\n')
            if 'rf' in self.experts:
                file.write(f'rf hyperparameters: {rf_hyperparams}\n')
             
        print(f'The models have been fitted for {num_fitting} times.')
        print('========================================================')

import argparse

def main():

     
    parser = argparse.ArgumentParser(description='Run CPBandit with a list of experts.')
    parser.add_argument('experts_list', nargs='+', help='List of expert names')

    
    parser.add_argument('--num_test_pat_win', type=int, help='Number of testing set')
    parser.add_argument('--num_train_sepsis_pat_win', type=int,  help='Number of septic training set')
    parser.add_argument('--num_train_nosepsis_pat_win', type=int, help='Number of nonseptic training set')

    args = parser.parse_args()
    cpbanit_player = CPBandit(experts=args.experts_list, num_test_pat_win=args.num_test_pat_win, num_train_sepsis_pat_win=args.num_train_sepsis_pat_win, num_train_nosepsis_pat_win = args.num_train_nosepsis_pat_win)
    # cpbanit_player = CPBandit(experts=['rf','ridge'], num_test_pat=10, num_train_sepsis_pat=10, num_train_nosepsis_pat = 30)

    cpbanit_player._start_game()

if __name__=='__main__':
    main()
 