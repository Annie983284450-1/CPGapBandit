import time
import argparse
# from scipy.stats import skew
import seaborn as sns
import math
import sys
import PI_Sepsysolcp_prob as EnbPI
import matplotlib
# from sklearn.linear_model import ElasticNet
# from sklearn.linear_model import ElasticNetCV
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
# from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier

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
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.svm import SVR
# from sklearn.model_selection import RandomizedSearchCV


# from sklearn import neighbors
# from sklearn.neural_network import MLPClassifier
from sklearn import svm
import utils_Sepsysolcp as util
from matplotlib.lines import Line2D  # For legend handles
import calendar
import warnings
import matplotlib.pyplot as plt
# from sklearn.linear_model import RidgeCV, LassoCV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import MinMaxScaler
import time
import pandas as pd
import numpy as np
# from sklearn.metrics import mean_squared_error
# from reward import get_UCB_LCB_avg
# from reward import get_absolute_error
from gap_b import gap_bandit 
# this will suppress all the error messages
# be cautious
# stderr = sys.stderr
# sys.stderr = open('logfile.log','w')
# import tensorflow as tf
# sys.stderr = stderr
# tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")
# importlib.reload(sys.modules['PI_class_EnbPI_journal'])

import multiprocessing
import dill
# from get_params import get_params
from get_params_prob import get_params_prob
# from get_UCB_LCB import UCB_LCB
# from evaluate_sepsis_score import compute_prediction_utility


from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
# from sklearn.linear_model import ElasticNetCV

multiprocessing.get_context().Process().Pickle = dill
# =============Read data and initialize parameters
class CPBandit:
    def __init__(self, experts, num_test_sepsis_pat_args, num_train_sepsis_pat_args, refit_step_args, B_args,numprocessors_args):
        self.experts = experts
        self.k = len(experts)
        self.UCB = [0]*self.k
        self.LCB = [0]*self.k
        self.UCB1 = [0]*self.k
        self.LCB1 = [0]*self.k
        # self.UCB_prob = [0]*self.k
        # self.LCB_prob = [0]*self.k
        self.rewards =[0]*self.k
        self.hat_mu_list_upper = np.zeros(self.k)   
        self.hat_mu_list_lower = np.zeros(self.k)   
        self.hat_mu_list = np.zeros(self.k)  

        self.hat_mu_list_upper1 = np.zeros(self.k)   
        self.hat_mu_list_lower1 = np.zeros(self.k)   
        self.hat_mu_list1 = np.zeros(self.k) 

        self.N = [0]*self.k  # N_i(t) (i=1,2,3,...k), cumulated # times arm i got pulled
        self.N1 = [0]*self.k  # N_i(t) (i=1,2,3,...k), cumulated # times arm i got pulled
        self.t = 0 # round t, each round is correlated to one test patient
        # self.rewards = list()    #[0]*self.T
        self.mu_best =1
        self.num_test_sepsis_pat = num_test_sepsis_pat_args
        self.num_train_sepsis_pat = num_train_sepsis_pat_args
        self.refit_step = refit_step_args
        self.B = B_args
        self.numprocessors = numprocessors_args

    def upperBound2(self, N_it, alpha):
        return  math.sqrt(alpha * math.log(self.t+1) / N_it)
    def _start_game(self):
        '''
        num_test_pat = 10  
        num_train_sepsis_pat = 50 
        num_train_nosepsis_pat = 150 
        Total excution time ---  5 seconds ---
        num_test_pat = 100  
        num_train_sepsis_pat = 500 
        num_train_nosepsis_pat = 1500 
        Total excution time --- 2472.87833571434 seconds ---
        the time complexity is too high. So we need to do the refitting less frequently
        '''

        num_test_sepsis_pat = self.num_test_sepsis_pat
        num_train_sepsis_pat = self.num_train_sepsis_pat
        refit_step = self.refit_step

        # num_test_sepsis_pat = 100
        # num_train_sepsis_pat = 1500
        # refit_step = 100000 # just do not refit to save time
        num_train_nosepsis_pat = num_train_sepsis_pat 
        

        start_test = 0
        start_nosepsis_train = 0
        start_sepsis_train = 0
        # debugging passed for get_params() - 20240925
        
        data_name = 'physionet_sepsis'
        stride = 1
        miss_test_idx=[]
        tot_trial = 1
        # usually B=30 can make sure every sample have LOO residual
        B = self.B
        K = len(self.experts)
    
        alpha_ls = np.linspace(0.05,0.25,5)
        min_alpha = 0.0001
        max_alpha = 10
        bandit_alpha = 2       


        # dt_early=-12
        # dt_optimal=-6
        # dt_late=3.0
        max_u_tp=2
        min_u_fn=-2
        u_fp=-0.05
        u_tn=0.05

        
        
        params = get_params_prob(self.experts, num_test_sepsis_pat, num_train_sepsis_pat, start_test, start_nosepsis_train, start_sepsis_train,B)
        X_train, Y_train, test_set, final_result_path, sepsis_full, train_size_pat, test_sepsis, test_nosepsis = params.get_train_test_set()
      


        f_name = ''
        for i, expert in enumerate(self.experts):
            if i==0:
                f_name = f_name+expert
            else:
                f_name = f_name+'_'+expert
        f_dat_path = os.path.join(final_result_path,f_name)
        os.makedirs(f_dat_path, exist_ok=True)

        # the original num of pats in the training dataset
        train_folder_name = train_size_pat
        original_stdout = sys.stdout

        with open(f_dat_path+'/'+'_'.join(self.experts)+'.log', 'w') as f:
            sys.stdout = f  
            print(f'=================~~~~~~~~~~~        expert list: {self.experts}. =================~~~~~~~~~~~ ')
            methods  = ['Ensemble'] 
            start_time = time.time()
            num_pat_tested = 0
            num_pat_tested0 = 0
            num_pat_tested1 = 0
            num_fitting = 0
            
            X_size = {}
            expert_idx = list(range(K))
            expert_dict = dict(zip(self.experts, expert_idx))
            max_hours = 480  
            predictions_namelist = [x+'_predictions' for x in self.experts]
            predictions_col = ['pat_id','itrial', 'SepsisLabel']
            predictions_col.extend(predictions_namelist)
            # Supports predict_proba:
            if 'lasso' in self.experts:
                lasso_f = LogisticRegressionCV(Cs=np.linspace(min_alpha, max_alpha, 10), penalty='l1', solver='liblinear')
            # Supports predict_proba:
            if 'ridge' in self.experts:
                ridge_f = LogisticRegressionCV(Cs=np.linspace(min_alpha, max_alpha, 10), penalty='l2')

            # Supports predict_proba: Yes, since Logistic Regression is a classification model, it can provide probability estimates for each class.
            # classifier
            if 'lr' in self.experts:
                lr_f = LogisticRegressionCV(Cs=np.linspace(min_alpha, max_alpha, 10), penalty='elasticnet', solver='saga', l1_ratios=[0.5])


            # Supports predict_proba: Yes, Random Forest is a classification algorithm, and it supports the predict_proba method for estimating class probabilities.
            if 'rf' in self.experts:
                rf_f = RandomForestClassifier(n_estimators=10, criterion='gini',
                                            bootstrap=False, max_depth=2, n_jobs=-1)
            # Supports predict_proba: Yes, but only if you set probability=True, 
            if 'svr' in self.experts:
                svr_f = SVC(probability=True)  # Use SVC for classification with probability estimates
                
            if 'xgb' in self.experts:
                xgb_f = XGBClassifier(
                    tree_method='hist',           # Use 'hist' for faster tree method
                    # tree_method='gpu_hist',         # Set to 'gpu_hist' for GPU acceleration
                    objective='binary:logistic',  # Use 'binary:logistic' for binary classification
                    n_estimators=100,             # Number of trees (boosting rounds)
                    learning_rate=0.1,            # Step size shrinkage
                    max_depth=3,                  # Maximum depth of a tree
                    subsample=1,                  # Subsample ratio of the training instances
                    colsample_bytree=1            # Subsample ratio of columns when constructing each tree
                )

            # Supports predict_proba: Yes, DecisionTreeClassifier supports predict_proba for estimating class probabilities.
            if 'dct' in self.experts:
                dct_f = DecisionTreeClassifier(random_state=0)
            # Supports predict_proba: No. This is a regression model for linear regression, so it does not provide class probabilities. It uses continuous outputs instead.
            # if 'enet' in self.experts:
            #     enet_f = ElasticNetCV(alpha=1.0, l1_ratio=0.5)
            if 'lgb' in self.experts:
                lgb_f = LGBMClassifier(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=-1
                    # device='gpu'
                    )  # Set device to 'gpu' for GPU acceleration
            if 'cat' in self.experts:
                cat_f = CatBoostClassifier(
                    iterations=100, 
                    learning_rate=0.1,
                    depth=3, 
                    verbose=0
                    # task_type='GPU'
                    )  # Set task type to 'GPU'
            if 'nnet' in self.experts:
                nnet_f = 'nnet_f'

            
            for patient_id in test_set:
                print('\n\n')
                print(f'=======         Processing patient {num_pat_tested}th patient: {patient_id}====================')
                start_curr_pat_time =  time.time()
                curr_pat_df = sepsis_full[sepsis_full['pat_id']==patient_id] 
                # Don;t forget to reset the index, pandas assign data by index. Otherwise, we will get NaN values
                curr_pat_df = curr_pat_df.reset_index(drop=True)
                if curr_pat_df['SepsisLabel'].notna().any():  
                    curr_pat_SepsisLabel = curr_pat_df['SepsisLabel']
                    Y_predict = curr_pat_df['SepsisLabel']
                    print(f'Hours of ICU stay: {len(Y_predict)}')
                else:
                    print("Patient {patient_id}: Y_predict or SepsisLabel contains NaN values. Please check the data.")
                    continue

                X_predict = curr_pat_df.drop(columns=['pat_id','hours2sepsis','SepsisLabel'])

                # update the training dataset when the number of tested patients is a multiple of refit_step
                # this must be done before the model fitting 
                if num_pat_tested % refit_step==0 and num_pat_tested>refit_step:
                    Isrefit = True
                    if num_pat_tested !=0:
                        # update the datasize 
                        train_folder_name = train_size_pat
                    
                    if num_pat_tested!=0:
                        X_size['Old_X_Size'] = X_train.shape[0]
                        X_train = X_train_merged
                        Y_train = Y_train_merged
                        X_size['New_X_Size'] = X_train.shape[0]        
                        print(f'Training Dataset Updated!!!!!!!!!!!!!!!!!!!!!')
                        print(X_size)
                    if start_test!=0:
                        Isrefit = False

                else:
                    Isrefit = False    
                    if num_pat_tested == 0:
                        Isrefit = True

                # train_size = X_train.shape[0]        

                for itrial in range(tot_trial):
                    curr_pat_predictions = pd.DataFrame(columns=predictions_col)
                    np.random.seed(99999+itrial)
                    cp_EnbPI_dict = {}
                    for expert in self.experts:
                        # print('\n\n')
                        # print(f' ========= &&&&&& Starting fitting the model and make predictions for {expert}.... ========= &&&&&&')
                        cp_EnbPI = EnbPI.prediction_interval(locals()[f'{expert}_f'], X_train, X_predict, Y_train, Y_predict, \
                                                             final_result_path, \
                                                             self.experts, train_folder_name, refit_step, self.numprocessors)
                        cp_EnbPI.fit_bootstrap_models_online_multi(B, miss_test_idx, Isrefit, model_name = expert)
                        # print(f' ========= &&&&&& Finish fitting model and predictions got for {expert}.... ========= &&&&&&')
                        print('\n\n')
                        curr_pat_predictions[f'{expert}_predictions'] = cp_EnbPI.Ensemble_pred_interval_centers
                        cp_EnbPI_dict[f'{expert}'] = cp_EnbPI


                    curr_pat_predictions['itrial'] = itrial
                    curr_pat_predictions['pat_id'] = patient_id
                  
                    curr_pat_predictions['SepsisLabel'] = curr_pat_SepsisLabel

                    histories_dat_path = f_dat_path + '/all_histories/' + 'itrial#'+str(itrial) 
                    if not os.path.exists(histories_dat_path):
                        os.makedirs(histories_dat_path)

                    # curr_pat_predictions: predictions_col =['pat_id','itrial', 'ridge_predictions', 'rf_predictions', 'nn_predictions']    
                    curr_pat_predictions.to_csv(f'{histories_dat_path}/predictions_{patient_id}.csv')

                    for alpha in alpha_ls:

                        print(f'~~~~~~~~~~At trial # {itrial} and alpha={alpha}~~~~~~~~~~~~~~')
                        for method in methods:
                            # cp_dat_path = histories_dat_path + '/' + method
                            # if not os.path.exists(cp_dat_path):
                            #     os.makedirs(cp_dat_path)
                            coverage_dict = {}
                            avg_width_dict = {}

                            if method == 'Ensemble':
                                for expert in self.experts:
                                    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~run_experiments() for [{expert}] executing~~~~~~~~~~~~~~~~~~~~')
                                    # print(f'~~~~~~~~~~At trial # {itrial} and alpha={alpha} and method={method} and expert={expert}~~~~~~~~~~~~~~')
                                    PIs_df, results = cp_EnbPI_dict[f'{expert}'].run_experiments(alpha, stride, data_name, itrial,
                                                        true_Y_predict=[], get_plots=False, none_CP=False, methods=methods,max_hours=max_hours)
                                    print(results)
                                    coverage_dict[expert] = results.mean_coverage.values[0]
                                    avg_width_dict[expert] = results.avg_width.values[0]

                                    # Y_upper = PIs_df['upper']
                                    # Y_lower = PIs_df['lower']
                                    k_idx = expert_dict[f'{expert}']
                               
                                    if len(curr_pat_predictions) == len(PIs_df):
                                        # Calculate the absolute error
                                        errors_upper = (curr_pat_predictions['SepsisLabel'] - PIs_df['upper']).abs()
                                        errors_lower = (curr_pat_predictions['SepsisLabel'] - PIs_df['lower']).abs()
                                        errors = (curr_pat_predictions['SepsisLabel'] - curr_pat_predictions[f'{expert}_predictions']).abs()
                                        # Calculate the mean error
                                        mean_error_upper = errors_upper.mean()
                                        mean_error_lower = errors_lower.mean()
                                        mean_error = errors.mean()
                                        # print(f"Mean Error: {mean_error_upper}")
                                    else:
                                        print("Warning: The lengths of curr_pat_predictions and PIs_df do not match.")
                                        sys.exit('Please check the data.')
                                    observed_utilities =    1 - mean_error
                                    upper_observed_utilities = 1- mean_error_upper
                                    lower_observed_utilities = 1- mean_error_lower                     
                                    self.rewards[k_idx] = observed_utilities


                                    if patient_id in test_nosepsis:

                                        self.hat_mu_list[k_idx] = (self.rewards[k_idx] +   self.hat_mu_list[k_idx]*num_pat_tested0)/(num_pat_tested0+1)
                                        # self.hat_mu_list[k_idx] =  observed_utilities 
                                        # print(f'============== observed_utilities of {expert}: {self.hat_mu_list[k_idx]} ==============')

                                        self.hat_mu_list_upper[k_idx] = (upper_observed_utilities + self.hat_mu_list_upper[k_idx]*num_pat_tested0)/(num_pat_tested0+1)
                                        self.hat_mu_list_lower[k_idx] = (lower_observed_utilities + self.hat_mu_list_lower[k_idx]*num_pat_tested0)/(num_pat_tested0+1)
                                    

                                        if self.N[k_idx] == 0:
                                            self.UCB[k_idx] = sys.float_info.max  #  encourage exploration
                                            self.LCB[k_idx] = -sys.float_info.max  # Also encourage exploration
                                        else:
                                            self.UCB[k_idx]= self.hat_mu_list_upper[k_idx]+self.upperBound2(N_it = self.N[k_idx], alpha = bandit_alpha)
                                            self.LCB[k_idx]= self.hat_mu_list_lower[k_idx]+self.upperBound2(N_it = self.N[k_idx], alpha = bandit_alpha)
                                    if patient_id in test_sepsis:
                                        self.hat_mu_list1[k_idx] = (self.rewards[k_idx] + self.hat_mu_list1[k_idx]*num_pat_tested1)/(num_pat_tested1+1)
                                        # self.hat_mu_list[k_idx] =  observed_utilities 
                                        # print(f'============== observed_utilities of {expert}: {self.hat_mu_list[k_idx]} ==============')

                                        self.hat_mu_list_upper1[k_idx] = (upper_observed_utilities +   self.hat_mu_list_upper1[k_idx]*num_pat_tested1)/(num_pat_tested1+1)
                                        self.hat_mu_list_lower1[k_idx] = (lower_observed_utilities +   self.hat_mu_list_lower1[k_idx]*num_pat_tested1)/(num_pat_tested1+1)
                                    

                                        if self.N1[k_idx] == 0:
                                            self.UCB1[k_idx] = sys.float_info.max  #  encourage exploration
                                            self.LCB1[k_idx] = -sys.float_info.max  # Also encourage exploration
                                        else:
                                            self.UCB1[k_idx]= self.hat_mu_list_upper1[k_idx]+self.upperBound2(N_it = self.N1[k_idx], alpha = bandit_alpha)
                                            self.LCB1[k_idx]= self.hat_mu_list_lower1[k_idx]+self.upperBound2(N_it = self.N1[k_idx], alpha = bandit_alpha)
    
 
                                  

                            print(f'        @@@@~~~@@@@~~~@@@@~~~Selecting the best expert on average @@@@~~~')

                            if patient_id in test_nosepsis:
                                pulled_arm_idx_curr_pat = gap_bandit(self.UCB, self.LCB, self.k).pull_arm()
                                self.N[pulled_arm_idx_curr_pat] += 1
                            if patient_id in test_sepsis:
                                pulled_arm_idx_curr_pat = gap_bandit(self.UCB1, self.LCB1, self.k).pull_arm()
                                self.N1[pulled_arm_idx_curr_pat] += 1
                
                            
   
                            new_row_all_avg = {'patient_id': patient_id, 'alpha': alpha, 'itrial': itrial, 'method': method}

                            for expert in self.experts:
                                new_row_all_avg[f'{expert}_coverage'] = coverage_dict[expert]
                                new_row_all_avg[f'{expert}_avg_width'] = avg_width_dict[expert]
                                # k_idx = expert_dict[f'{expert}']
                            new_row_all_avg['winner'] = list(expert_dict.keys())[pulled_arm_idx_curr_pat]
                            new_row_all_avg['regret'] = 1 - self.rewards[pulled_arm_idx_curr_pat]
                            new_row_all_avg['observed_utility'] =self.rewards[pulled_arm_idx_curr_pat]
                            

                            if patient_id in test_nosepsis:
                                new_row_all_avg['class'] = 0
                                new_row_all_avg['hat_mu_optimal'] = self.hat_mu_list[pulled_arm_idx_curr_pat]
                            if patient_id in test_sepsis:
                                new_row_all_avg['class'] = 1
                                new_row_all_avg['hat_mu_optimal'] = self.hat_mu_list1[pulled_arm_idx_curr_pat]


                            if not isinstance(new_row_all_avg, pd.DataFrame):
                                new_row_all_avg = pd.DataFrame([new_row_all_avg])
                    
                            if num_pat_tested == 0:
                                os.makedirs(f_dat_path+f'/final_all_results_avg', exist_ok=True)
                                with open(f_dat_path+f'/final_all_results_avg'+f'/final_all_results_avg(alpha={alpha}).csv', 'w') as f:
                                    pass  # Just opening in 'w' mode truncates the file
                            # Append to CSV file
                            with open(f_dat_path+f'/final_all_results_avg'+f'/final_all_results_avg(alpha={alpha}).csv', 'a') as f:
                                new_row_all_avg.to_csv(f, header=f.tell()==0, index=False)     



                # updating dataset
                print(f'\n')
                print(f'-------------------------------------------')
                print(f'Updating the training dataset ..........')
                X_train_new_df = curr_pat_df.drop(columns = ['pat_id','hours2sepsis', 'SepsisLabel'])
                Y_train_new_df = curr_pat_df['SepsisLabel'] 
                X_train_new = X_train_new_df.to_numpy(dtype='float', na_value=np.nan)
                Y_train_new = Y_train_new_df.to_numpy(dtype='float', na_value=np.nan)
                if num_pat_tested ==0:
                    X_train_merged = np.append(X_train,X_train_new,axis=0)
                    Y_train_merged = np.append(Y_train,Y_train_new,axis=0)
                else:
                    X_train_merged = np.append(X_train_merged,X_train_new,axis=0)
                    Y_train_merged = np.append(Y_train_merged,Y_train_new,axis=0)

                train_size_pat = train_size_pat+1
                if Isrefit:
                    np.save(final_result_path+'/'+'_'.join(self.experts)+f'/X_train_{len(X_train_merged)}.npy', X_train_merged)
                    np.save(final_result_path+'/'+'_'.join(self.experts)+f'/Y_train_{len(X_train_merged)}.npy', Y_train_merged)
                    num_fitting = num_fitting+1 

                print(f'~~~~~~Excution time for # {patient_id}: {time.time()-start_curr_pat_time} seconds~~~~~~')
                print('\n\n')
                print('========================================================')
                print('========================================================')
                num_pat_tested = num_pat_tested + 1
                self.t += 1
                if patient_id in test_nosepsis:
                    num_pat_tested0+=1
                if patient_id in test_sepsis:
                    num_pat_tested1+=1

                if num_pat_tested1+num_pat_tested0!=num_pat_tested:
                    sys.exit('!!!!!!!!! Error: num_pat_tested1+num_pat_tested0!=num_pat_tested')
                else:
                    print(f'# {num_pat_tested} patients already tested! ......')
                    print(f'# {num_pat_tested0} nonseptic patients already tested! ......')
                    print(f'# {num_pat_tested1} septic patients already tested! ......')
                if self.t != num_pat_tested:
                    sys.exit('!!!!!!!!! Error: self.t != num_pat_tested ')

            print('========================================================')
            print('========================================================')
            print(f'Test size: {len(test_set)}')
        
        
            print(f'Total excution time: {(time.time() - start_time)} seconds~~~~~~' )
            machine = 'login-phoenix-rh9.pace.gatech.edu'
            with open(final_result_path+'/'+'_'.join(self.experts)+'/execution_info.txt', 'w') as file:
                file.write(f'Total excution time: {(time.time() - start_time)} seconds\n')
                file.write(f'num_test_sepsis_pat = {num_test_sepsis_pat}\n')
                file.write(f'num_test_nosepsis_pat = {len(test_nosepsis)}\n')
                file.write(f'num_train_sepsis_pat = {num_train_sepsis_pat}\n')
                file.write(f'num_train_nosepsis_pat = {num_train_nosepsis_pat}\n')
                file.write(f'tot_trial = {tot_trial}\n')
                file.write(f'refit_step = {refit_step}\n') 
                file.write(f'No of experts = {self.k}\n')
                file.write(f'Experts = {list(expert_dict.keys())}\n')
                file.write(f'The models have been fitted for {num_fitting} times.\n')
                file.write(f'Machine: {machine}\n')
                file.write(f'Multiprocessing: True \n')
            print(f'The models have been fitted for {num_fitting} time(s).')
            print('========================================================')

        # Restore stdout to its original value (console output)

        sys.stdout = original_stdout

    


def main():
    parser = argparse.ArgumentParser(description='CPBandit')
    parser.add_argument('--num_test_sepsis_pat', type=int, default=2)
    parser.add_argument('--num_train_sepsis_pat', type=int, default=5)
    parser.add_argument('--refit_step', type=int, default=1000000) # do not refit by default
    parser.add_argument('--B', type=int, default=25)
    parser.add_argument('--np', type=int, default=12)

    parser.add_argument('--combo', type=str, default=None)

    args = parser.parse_args() 

    if args.combo =='lasso':
        experts_lists = [
                     ['lasso','xgb','rf','dct','lr'],
                     ['lasso','rf','dct','lr'],
                     ['lasso','rf','lr'],
                     ['lasso', 'rf'],
                     ['lasso']
        ]
    if args.combo == 'nnet':
        experts_lists = [
                    ['nnet', 'rf','xgb', 'ridge','dct'],
                    ['nnet', 'rf','xgb', 'ridge'],
                    ['nnet', 'rf','xgb'],
                    ['nnet', 'rf'],
                    ['nnet']             
        ]
    if args.combo == 'rf':
        experts_lists = [
                    # ['rf','xgb', 'ridge','dct','lr'], # already run  
                    ['rf','xgb', 'ridge','lr'],
                    ['rf','xgb', 'lr'],
                    ['rf','xgb'],
                    ['rf'] 
                               
        ]


    if args.combo == 'xgb':
        experts_lists = [
                        ['xgb','cat','rf','dct','lr'],     # already run  
                        ['xgb','rf','dct','lr'],
                        ['xgb','rf','lr'],
                        ['xgb', 'rf'],
                        ['xgb']
                        ]
        
        
    if args.combo == 'dct':
        experts_lists = [
                        # ['dct','cat','rf','ridge','lr'],      # already run   
                        ['dct','rf','ridge','cat'],
                        ['dct','rf','cat'],
                        ['dct', 'cat'],
                        ['dct']
                        ]
    if args.combo == 'ridge':
        experts_lists = [
                        ['ridge','cat','rf','dct','lr'],     # already run  
                        ['ridge','rf','dct','lr'],
                        ['ridge','rf','xgb'],
                        ['ridge', 'xgb'],
                        ['ridge']
                        ]

    if args.combo == None:
        sys.exit('Input combo!!!')


    for experts_list in experts_lists:
        print('\n\n')
        print('\n\n')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'@#$%$&^%*&^ @@@@@@@@@@@ experts_list: @#$%$&^%*&^ @@@@@@@@@@@ ')
        print(experts_list)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('\n\n')
        cpbanit_player = CPBandit(experts=experts_list, num_test_sepsis_pat_args=args.num_test_sepsis_pat, \
                                  num_train_sepsis_pat_args=args.num_train_sepsis_pat, refit_step_args=args.refit_step, B_args = args.B, numprocessors_args = args.np)
        cpbanit_player._start_game()
if __name__=='__main__':
    main()
 