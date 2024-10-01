import time
from scipy.stats import skew
import seaborn as sns
import math
import sys
import PI_Sepsysolcp as EnbPI
import matplotlib
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
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


from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import utils_Sepsysolcp as util
from matplotlib.lines import Line2D  # For legend handles
import calendar
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import MinMaxScaler
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
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
from get_params import get_params
from get_UCB_LCB import UCB_LCB
from evaluate_sepsis_score import compute_prediction_utility


from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import ElasticNetCV
from xgboost import XGBClassifier

multiprocessing.get_context().Process().Pickle = dill
# =============Read data and initialize parameters
class CPBandit:
    def __init__(self, experts):
        self.experts = experts
        self.k = len(experts)
        self.UCB = [0]*self.k
        self.LCB = [0]*self.k
        self.rewards =[0]*self.k
        self.hat_mu_list_upper = np.zeros(self.k)   
        self.hat_mu_list_lower = np.zeros(self.k)   
        self.hat_mu_list = np.zeros(self.k)   
        self.N = [0]*self.k  # N_i(t) (i=1,2,3,...k), cumulated # times arm i got pulled
        self.t = 0 # round t, each round is correlated to one test patient
        # self.rewards = list()    #[0]*self.T
        self.mu_best =1
 

    def upperBound2(self, N_it, alpha):
        return  math.sqrt(alpha * math.log(self.t+1) / N_it)

    # def get_average_regret(self,t):

    #     regret = ((t+1)*self.mu_best - np.sum(self.rewards))/float(t+1)
    #     if self.mu_best<self.rewards[-1]:
    #         print("the reward of round ", t , " exceeds 4, which is ", self.rewards[-1], "!!!!!!")
    #     return regret

    # def get_cu_regret(self):

    #     regret = (self.t+1)*self.mu_best - np.sum(self.rewards)
    #     if self.mu_best<self.rewards[-1]:
    #         print("the reward of round ", t , " exceeds 4, which is ", self.rewards[-1], "!!!!!!")
    #     return regret


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
        num_test_pat = 300
        num_train_sepsis_pat = 500
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
        B = 25
        K = len(self.experts)
        alpha=0.1
        alpha_ls = np.linspace(0.05,0.25,5)
        min_alpha = 0.0001
        max_alpha = 10
        bandit_alpha = 2       


        dt_early=-12
        dt_optimal=-6
        dt_late=3.0
        max_u_tp=2
        min_u_fn=-2
        u_fp=-0.05
        u_tn=0.05
        
        params = get_params(self.experts, num_test_pat, num_train_sepsis_pat, start_test, start_nosepsis_train, start_sepsis_train, max_u_tp=max_u_tp,min_u_fn=min_u_fn,u_fp=u_fp,u_tn=u_tn)
        X_train, Y_train, test_set, final_result_path, sepsis_full = params.get_train_test_set()
        # sys.exit('testing get_train_test_set()')


        f_name = ''
        for i, expert in enumerate(self.experts):
            if i==0:
                f_name = f_name+expert
            else:
                f_name = f_name+'_'+expert
        f_dat_path = os.path.join(final_result_path,f_name)
        os.makedirs(f_dat_path, exist_ok=True)


        original_stdout = sys.stdout

        with open(f_dat_path+'/log.txt', 'w') as f:
            sys.stdout = f  # Redirect stdout to the file

            # Your main code goes here
            # print("This will be written to the output_log.out file.")
            print(f'=================~~~~~~~~~~~        expert list: {self.experts}. =================~~~~~~~~~~~ ')
            print(f'len(self.experts) == {len(self.experts)}')
            # sys.exit()




            methods  = ['Ensemble'] # conformal prediction methods
            
            start_time = time.time()
    
            num_pat_tested = 0
            num_fitting = 0
            refit_step = 100
            X_size = {}
            
            expert_idx = list(range(K))
            expert_dict = dict(zip(self.experts, expert_idx))
        
    
            

            max_hours = 480  
            predictions_namelist = [x+'_predictions' for x in self.experts]
            predictions_col = ['pat_id','itrial', 'SepsisLabel']
            predictions_col.extend(predictions_namelist)
    
            # if 'lasso' in self.experts:
            #     lasso_f = LassoCV(alphas=np.linspace(min_alpha, max_alpha, 10))

            # if 'ridge' in self.experts:
            #     ridge_f = RidgeCV(alphas=np.linspace(min_alpha, max_alpha, 10))
            # if 'rf' in self.experts:
            #     rf_f = RandomForestRegressor(n_estimators=10, criterion='mse',
            #                                 bootstrap=False, max_depth=2, n_jobs=-1)
            # if 'svr' in self.experts:
            #     # param_distributions = {
            #     #     'C': [0.1, 1, 10, 100],
            #     #     'gamma': ['scale', 'auto'],
            #     #     'kernel': ['linear', 'rbf', 'poly']
            #     # }
            #     svr_f = SVR()
            #     # random_search_svr = RandomizedSearchCV(svr, param_distributions, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
            # if 'xgb' in self.experts:
            #     xgb_hyperparams = {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
            #     xgb_f = XGBRegressor(subsample=xgb_hyperparams['subsample'],
            #                                         n_estimators=xgb_hyperparams['n_estimators'],
            #                                         max_depth=xgb_hyperparams['max_depth'],
            #                                         learning_rate=xgb_hyperparams['learning_rate'])
            # if 'enet' in self.experts:
            #     enet_f = ElasticNet(random_state=0)
            # if 'dct' in self.experts:
            #     dct_f = DecisionTreeRegressor(random_state=0)

            if 'lasso' in self.experts:
                lasso_f = LogisticRegressionCV(Cs=np.linspace(min_alpha, max_alpha, 10), penalty='l1', solver='liblinear')

            if 'ridge' in self.experts:
                ridge_f = LogisticRegressionCV(Cs=np.linspace(min_alpha, max_alpha, 10), penalty='l2')

            if 'rf' in self.experts:
                rf_f = RandomForestClassifier(n_estimators=10, criterion='gini',
                                            bootstrap=False, max_depth=2, n_jobs=-1)

            if 'svr' in self.experts:
                svr_f = SVC(probability=True)  # Use SVC for classification with probability estimates

            if 'xgb' in self.experts:
                xgb_hyperparams = {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
                xgb_f = XGBClassifier(subsample=xgb_hyperparams['subsample'],
                                    n_estimators=xgb_hyperparams['n_estimators'],
                                    max_depth=xgb_hyperparams['max_depth'],
                                    learning_rate=xgb_hyperparams['learning_rate'])

            if 'lr' in self.experts:
                lr_f = LogisticRegressionCV(Cs=np.linspace(min_alpha, max_alpha, 10), penalty='elasticnet', solver='saga', l1_ratios=[0.5])

            if 'dct' in self.experts:
                dct_f = DecisionTreeClassifier(random_state=0)


            for patient_id in test_set:
                start_curr_pat_time =  time.time()
                # print('\n\n')
                
                # Note: If there are fewer than 48 rows with the desired value, the result will contain all rows with that value.
                # only take the data from the first 48 hours
                # curr_pat_df = sepsis_full[sepsis_full['pat_id']==patient_id].head(48)
                curr_pat_df = sepsis_full[sepsis_full['pat_id']==patient_id] 
                # Don;t forget to reset the index, pandas assign data by index. Otherwise, we will get NaN values
                curr_pat_df = curr_pat_df.reset_index(drop=True)
                
                # curr_pat_SepsisLabel = curr_pat_df['SepsisLabel']
                # curr_pat_df is not changed
                
                # Y_predict = curr_pat_df['hours2sepsis']
                if curr_pat_df['hours2sepsis'].notna().any() and curr_pat_df['SepsisLabel'].notna().any():
                    # Y_predict = curr_pat_df['hours2sepsis']
                    curr_pat_SepsisLabel = curr_pat_df['SepsisLabel']
                    # Y_predict = curr_pat_df['hours2sepsis']
                    Y_predict = curr_pat_df['SepsisLabel']
                    print(f'Hours of ICU stay: {len(Y_predict)}')
                else:
                    print("Patient {patient_id}: Y_predict or SepsisLabel contains NaN values. Please check the data.")
                    continue

                X_predict = curr_pat_df.drop(columns=['pat_id','hours2sepsis','SepsisLabel'])
                # print(f'Y_predict: {Y_predict}')


                # if pd.isnull(Y_predict).any():
                #     print(f'Patient {patient_id} has missing sepsis labels!!!!!!! Skip this patient!')
                #     continue


                print(f'=======         Processing patient {num_pat_tested}th patient: {patient_id}====================')

                # curr_pat_histories['hours2sepsis'] = Y_predict
                        
                if num_pat_tested % refit_step==0:
                    Isrefit = True
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

                for itrial in range(tot_trial):
                    curr_pat_predictions = pd.DataFrame(columns=predictions_col)
                    np.random.seed(99999+itrial)
                    cp_EnbPI_dict = {}
                    for expert in self.experts:

                        print(f' ========= &&&&&& Starting fitting the model and make predictions for {expert}.... ========= &&&&&&')
                        cp_EnbPI = EnbPI.prediction_interval(locals()[f'{expert}_f'], X_train, X_predict, Y_train, Y_predict,final_result_path)
                        cp_EnbPI.fit_bootstrap_models_online_multi(B, miss_test_idx, Isrefit, model_name = expert, max_hours=max_hours)
                        # cp_EnbPI.fit_bootstrap_models_online(B, miss_test_idx, Isrefit, model_name = expert, max_hours = max_hours)
                     
                        
                        
                        print(f' ========= &&&&&& Finish fitting model and predictions got for {expert}.... ========= &&&&&&')
                        curr_pat_predictions[f'{expert}_predictions'] = cp_EnbPI.Ensemble_pred_interval_centers
                        cp_EnbPI_dict[f'{expert}'] = cp_EnbPI


                    curr_pat_predictions['itrial'] = itrial
                    curr_pat_predictions['pat_id'] = patient_id
                    # curr_pat_predictions['hours2sepsis'] = Y_predict
                    curr_pat_predictions['SepsisLabel'] = curr_pat_SepsisLabel

                    # print(f'curr_pat_predictions: {curr_pat_predictions.head()}')
                    # sys.exit('testing curr_pat_predictions')


                    # final_result_path = './Results('test{num_test_pat},train_sepsis{num_train_sepsis_pat}self.experts))'
                    # f_dat_path final_result_path+'/dat'
                    histories_dat_path = f_dat_path + '/all_histories/' + 'itrial#'+str(itrial) 
                    if not os.path.exists(histories_dat_path):
                        os.makedirs(histories_dat_path)

                    # curr_pat_predictions: predictions_col =['pat_id','itrial', 'ridge_predictions', 'rf_predictions', 'nn_predictions',  'hours2sepsis']    
                    curr_pat_predictions.to_csv(f'{histories_dat_path}/predictions_{patient_id}.csv')

                    for alpha in alpha_ls:

                        print(f'~~~~~~~~~~At trial # {itrial} and alpha={alpha}~~~~~~~~~~~~~~')
                        for method in methods:
                            # cp_dat_path = histories_dat_path + '/' + method
                            # if not os.path.exists(cp_dat_path):
                            #     os.makedirs(cp_dat_path)
                            coverage_dict = {}
                            if method == 'Ensemble':
                                for expert in self.experts:
                                    print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~run_experiments() executing~~~~~~~~~~~~~~~~~~~~')
                                    print(f'~~~~~~~~~~At trial # {itrial} and alpha={alpha} and method={method} and expert={expert}~~~~~~~~~~~~~~')
                                    PIs_df, results = cp_EnbPI_dict[f'{expert}'].run_experiments(alpha, stride, data_name, itrial,
                                                        true_Y_predict=[], get_plots=False, none_CP=False, methods=methods,max_hours=max_hours)
                                    print(results)
                                    coverage_dict[expert] = results.mean_coverage.values[0]

                                    Y_upper = PIs_df['upper']
                                    Y_lower = PIs_df['lower']
                                    k_idx = expert_dict[f'{expert}']
                                    print(f'k_idx == {k_idx}')
                               
                                    
                                    
                                    observed_utilities = compute_prediction_utility(labels= curr_pat_SepsisLabel, predictions = curr_pat_predictions[f'{expert}_predictions'] , 
                                                                                    dt_early=dt_early, dt_optimal=dt_optimal, dt_late=dt_late, max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn,
                                                                                    max_hours= max_hours, check_errors = True)    
                                    upper_observed_utilities = compute_prediction_utility(labels= curr_pat_SepsisLabel, predictions = Y_upper, 
                                                                                    dt_early=dt_early, dt_optimal=dt_optimal, dt_late=dt_late, max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn,
                                                                                    max_hours= max_hours, check_errors = True)          
                                    lower_observed_utilities = compute_prediction_utility(labels= curr_pat_SepsisLabel, predictions = Y_lower, 
                                                                                    dt_early=dt_early, dt_optimal=dt_optimal, dt_late=dt_late, max_u_tp=max_u_tp, min_u_fn=min_u_fn, u_fp=u_fp, u_tn=u_tn,
                                                                                    max_hours= max_hours, check_errors = True) 
                                     
                                    self.rewards[k_idx] = observed_utilities

                                    self.hat_mu_list[k_idx] = (self.rewards[k_idx] +   self.hat_mu_list[k_idx]*num_pat_tested)/(num_pat_tested+1)
                                    # self.hat_mu_list[k_idx] =  observed_utilities 
                                    print(f'============== observed_utilities of {expert}: {self.hat_mu_list[k_idx]} ==============')

                                    self.hat_mu_list_upper[k_idx] = (upper_observed_utilities +   self.hat_mu_list_upper[k_idx]*num_pat_tested)/(num_pat_tested+1)
                                    self.hat_mu_list_lower[k_idx] = (lower_observed_utilities +   self.hat_mu_list_lower[k_idx]*num_pat_tested)/(num_pat_tested+1)
                                
                                    
                                    if self.N[k_idx] == 0:
                                        self.UCB[k_idx] = sys.float_info.max  #  encourage exploration
                                        self.LCB[k_idx] = -sys.float_info.max  # Also encourage exploration
                                    else:
                                        self.UCB[k_idx]= self.hat_mu_list_upper[k_idx]+self.upperBound2(N_it = self.N[k_idx], alpha = bandit_alpha)
                                        self.LCB[k_idx]= self.hat_mu_list_lower[k_idx]+self.upperBound2(N_it = self.N[k_idx], alpha = bandit_alpha)
                                    self.t += 1

                            print(f'@@@@~~~@@@@~~~@@@@~~~Selecting the best expert on average @@@@~~~')
                            pulled_arm_idx_curr_pat = gap_bandit(self.UCB, self.LCB, self.k).pull_arm()
                            # reward_curr_pat =  self.rewards[pulled_arm_idx_curr_pat]
                            # self.rewards.append(reward_curr_pat)
                            self.N[pulled_arm_idx_curr_pat] += 1

                            new_row_all_avg = {'patient_id': patient_id, 'alpha': alpha, 'itrial': itrial, 'method': method}

                            for expert in self.experts:
                                new_row_all_avg[f'{expert}_coverage'] = coverage_dict[expert]
                                # k_idx = expert_dict[f'{expert}']
                            new_row_all_avg['winner'] = list(expert_dict.keys())[pulled_arm_idx_curr_pat]
                            new_row_all_avg['regret'] = 1 - self.rewards[pulled_arm_idx_curr_pat]
                            new_row_all_avg['observed_utility'] =self.rewards[pulled_arm_idx_curr_pat]
                            new_row_all_avg['hat_mu_optimal'] = self.hat_mu_list[pulled_arm_idx_curr_pat]

                            if not isinstance(new_row_all_avg, pd.DataFrame):
                                new_row_all_avg = pd.DataFrame([new_row_all_avg])
                    
                            if num_pat_tested == 0:
                                os.makedirs(f_dat_path+f'/final_all_results_avg', exist_ok=True)
                                with open(f_dat_path+f'/final_all_results_avg'+f'/final_all_results_avg(alpha={alpha}).csv', 'w') as f:
                                    pass  # Just opening in 'w' mode truncates the file
                            # Append to CSV file
                            with open(f_dat_path+f'/final_all_results_avg'+f'/final_all_results_avg(alpha={alpha}).csv', 'a') as f:
                                new_row_all_avg.to_csv(f, header=f.tell()==0, index=False)    
    


    

                if Isrefit:
                    num_fitting = num_fitting+1            
                # updating dataset
                
                print(f'\n')
                print(f'-------------------------------------------')
                print(f'Updating the training dataset ..........')
                X_train_new_df = curr_pat_df.drop(columns = ['pat_id','hours2sepsis', 'SepsisLabel'])
                # train_set_df_x.append(X_train_new_df, ignore_index = True)
                Y_train_new_df = curr_pat_df['hours2sepsis']
                X_train_new = X_train_new_df.to_numpy(dtype='float', na_value=np.nan)
                Y_train_new = Y_train_new_df.to_numpy(dtype='float', na_value=np.nan)
                if num_pat_tested ==0:
                    X_train_merged = np.append(X_train,X_train_new,axis=0)
                    Y_train_merged = np.append(Y_train,Y_train_new,axis=0)
                else:
                    X_train_merged = np.append(X_train_merged,X_train_new,axis=0)
                    Y_train_merged = np.append(Y_train_merged,Y_train_new,axis=0)

                # dataset updated
                print(f'~~~~~~Excution time for # {patient_id}: {time.time()-start_curr_pat_time} seconds~~~~~~')
                print('\n\n')
                print('========================================================')
                print('========================================================')
                num_pat_tested = num_pat_tested + 1
                self.t += 1
                print(f'# {num_pat_tested} patients already tested! ......')

            
            np.save(final_result_path+'/X_train_merged.npy', X_train_merged)
            np.save(final_result_path+'/Y_train_merged.npy', Y_train_merged)

            # final_all_results_avg.to_csv(final_result_path+'/final_all_results_avg.csv')
            print('========================================================')
            print('========================================================')
            print(f'Test size: {len(test_set)}')
        
        
            print(f'Total excution time: {(time.time() - start_time)} seconds~~~~~~' )
            machine = 'ece-kl2313-01.ece.gatech.edu'
            with open(final_result_path+'/execution_info.txt', 'w') as file:
                file.write(f'Total excution time: {(time.time() - start_time)} seconds\n')
                file.write(f'num_test_pat = {num_test_pat}\n')
                file.write(f'num_train_sepsis_pat = {num_train_sepsis_pat}\n')
                file.write(f'num_train_nosepsis_pat = {num_train_nosepsis_pat}\n')
                # file.write(f'start_test = {start_test}\n')
                # file.write(f'start_nosepsis_train = {start_nosepsis_train}\n')
                # file.write(f'start_sepsis_train = {start_sepsis_train}\n')
                file.write(f'tot_trial = {tot_trial}\n')
                file.write(f'refit_step = {refit_step}\n') 
                file.write(f'No of experts = {self.k}\n')
                file.write(f'Experts = {list(expert_dict.keys())}\n')
                file.write(f'The models have been fitted for {num_fitting} times.\n')
                file.write(f'Machine: {machine}\n')
                file.write(f'Multiprocessor: True\n')  
                file.write(f'dt_early=-12\n')
                file.write(f'dt_optimal=-6\n')
                file.write(f'dt_late=3.0\n')
                file.write(f'max_u_tp=2\n')
                file.write(f'min_u_fn=-2\n')
                file.write(f'u_fp=-0.05\n')
                file.write(f'u_tn=0.05\n')
            print(f'The models have been fitted for {num_fitting} times.')
            print('========================================================')

        # Restore stdout to its original value (console output)

        sys.stdout = original_stdout

    


def main():
  
    # experts_lists = [['ridge','rf','lasso','xgb','dct','lr'],
    #                  ['ridge','rf','lasso','dct','lr'],
    #                  ['ridge','rf','dct','lr'],
    #                  ['ridge','rf','lr'],
    #                  ['ridge', 'rf'],
    #                  ['ridge']]  
    experts_lists = [['ridge','rf','dct','lr'],
                     ['ridge','rf','lr'],
                     ['ridge', 'rf'],
                     ['ridge']]  
    for experts_list in experts_lists:
        
        cpbanit_player = CPBandit(experts=experts_list)
        cpbanit_player._start_game()
if __name__=='__main__':
    main()
 