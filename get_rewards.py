import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from os import listdir
import csv
import shutil, os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import plot_roc_curve
#from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import joblib
import sys
import lime
import lime.lime_tabular
import math
#  ../../
path = os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir,os.path.pardir))
# add ../../
sys.path.append(path)

import evaluate_sepsis_score

'''
write all the patient data in the train_patient.csv file
append the patient_id tro identify each patient
add a time index for the data of each row (i.e., the data measured in an hour)
'''
 
## save the predicted labels
# def save_challenge_predictions(file, scores, labels):
#     with open(file, 'w+') as f:
#         f.write('PredictedProbability|PredictedLabel\n')
#         for (s, l) in zip(scores, labels):
#             f.write('%g|%d\n' % (s, l))
# ## save the true labels
# def save_challenge_testlabel(file, labels):
#     with open(file, 'w+') as f:
#         f.write('SepsisLabel\n')
#         for l in labels:
#             f.write('%d\n' % l)

# def load_model_predict(X_test, k_fold, path):
#     "ensemble the five XGBoost models by averaging their output probabilities"
#     test_pred = np.zeros((X_test.shape[0], k_fold))
#     X_test = xgb.DMatrix(X_test)
#     for k in range(k_fold):
#         model_path_name = path + 'model{}.mdl'.format(k+1)
#         xgb_model = xgb.Booster(model_file = model_path_name)
#         y_test_pred = xgb_model.predict(X_test)
#         test_pred[:, k] = y_test_predT1
#     test_pred = pd.DataFrame(test_pred)
#     result_pro = test_pred.mean(axis=1)

    # return result_pro



# def predict(psv,
#             save_prediction_dir,
#             save_label_dir,
#             X_testi,
#             Y_testi,
#             risk_threshold,
#             saved_model
#             ):
#     ## load trained model (if we run the script in the root folder)

#     fitted_model =  joblib.load(saved_model)

#     predicted_pro  = fitted_model.predict_proba(X_testi)
#     # we only need to know the pro of developing sepsis
#     PredictedProbability = np.array(predicted_pro[:,1])
#     print("Predicted Probability:", PredictedProbability)
#     PredictedLabel = [0 if i <= risk_threshold else 1 for i in predicted_pro[:,1]]


#     labels = Y_testi
 
#     save_prediction_name = save_prediction_dir + '/'+ psv
#     save_challenge_predictions(save_prediction_name, PredictedProbability, PredictedLabel)
#     save_testlabel_name = save_label_dir +  '/'+ psv
#     save_challenge_testlabel(save_testlabel_name, labels)

def predict( 
            predicted_hours2sepsis,
            Y_testi,     
            max_hours,
            len_stay
            ):
 
    # ground truth
    labels = [1 if i < max_hours else 0 for i in Y_testi]
    Predicted_hours2sepsis = predicted_hours2sepsis
    if len(labels) != len(Predicted_hours2sepsis):
        sys.exit("!!! Error: The length of the predicted hours2sepsis is not equal to the length of the labels!!!")
    
    PredictedLabel = [1 if i < min(len_stay, max_hours) else 0 for i in Predicted_hours2sepsis]


        

 

def get_reward_for_curr_pat(X_testi,Y_testi, patient_i,model_name):
    pass

def get_reward_for_current_patient_rf(X_testi,Y_testi, patient_i, experts_name_list, test_set, model_name):
    # test_set = np.load('./Candidate_xgb/data/test_set.npy')
    # test_set = np.sort(test_set,axis = None)
 
    test_seti = test_set[patient_i]
 
    f_name = ''
    for i, expert_name in enumerate( experts_name_list ):
        if i==0:
            f_name = f_name + expert_name
        else:
            f_name = f_name +'_'+ expert_name
        prediction_directory = os.path.join( f'./prediction_{model_name}/',f_name)
        label_directory = os.path.join(f'./labels_rf/',f_name)
 

    if not os.path.exists(prediction_directory):
        os.makedirs(prediction_directory)
    if not os.path.exists(label_directory):
        os.makedirs(label_directory)

    #prediction_directory = './Candidate_rf_svc_lr/prediction_rf/'
    #label_directory = './Candidate_rf_svc_lr/label/'
    saved_model= "./Candidate_rf_svc_lr/saved_model/random_forest.sav"
    print("random forest prediction: ")
    predict(test_seti, prediction_directory, label_directory, X_testi,Y_testi ,0.525, saved_model)


    utility = evaluate_sepsis_score(label_directory, prediction_directory,patient_i)

    print("Calculating utility score.............\n")
    print("The utility of Random Forest): ", patient_i)
    print(utility)
    print("\n")

    return float(utility)



def get_reward_for_current_patient_SVC(X_testi,Y_testi, patient_i, experts_name_list ):
    test_set = np.load('./Candidate_xgb/data/test_set.npy')
    ## sort all the psv files based on the number included in the psv filename (in ascending order)
    test_set = np.sort(test_set,axis = None)
    # Just want to get the psv file name herein
    test_seti = test_set[patient_i]

    f_name = ''
    for i, expert_name in enumerate( experts_name_list ):
        if i==0:
            f_name = f_name + expert_name
        else:
            f_name = f_name +'_'+ expert_name
        prediction_directory = os.path.join( './prediction_svc/',f_name)
        label_directory = os.path.join('./labels_svc/',f_name)

    if not os.path.isdir(prediction_directory):
        os.makedirs(prediction_directory)
    if not os.path.isdir(label_directory):
        os.makedirs(label_directory)
    # # if we run from the root directory ...
    saved_model= "./Candidate_rf_svc_lr/saved_model/svc.sav"
    print("SVC prediction: ")
    predict(test_seti, prediction_directory, label_directory, X_testi,Y_testi ,0.525, saved_model)

    utility = evaluate_sepsis_score(label_directory, prediction_directory,patient_i)
    print("Calculating utility score.............\n")
    print("The utility of SVC: ", patient_i)
    print(utility)
    print("\n")
    return float(utility)
## build the LIME explainer for random forest classifier if we get an extremely high or low utility

def get_reward_for_current_patient_LR(X_testi,Y_testi, patient_i, experts_name_list):
    test_set = np.load('./Candidate_xgb/data/test_set.npy')
    ## sort all the psv files based on the number included in the psv filename (in ascending order)
    test_set = np.sort(test_set,axis = None)
    # Just want to get the psv file name herein
    test_seti = test_set[patient_i]

    f_name = ''
    for i, expert_name in enumerate( experts_name_list ):
        if i==0:
            f_name = f_name + expert_name
        else:
            f_name = f_name +'_'+ expert_name
        prediction_directory = os.path.join( './prediction_lr/',f_name)
        label_directory = os.path.join('./labels_lr/',f_name)

    if not os.path.isdir(prediction_directory):
        os.makedirs(prediction_directory)
    if not os.path.isdir(label_directory):
        os.makedirs(label_directory)

    saved_model= "./Candidate_rf_svc_lr/saved_model/LR.sav"
 
    predict(test_seti, prediction_directory, label_directory, X_testi,Y_testi ,0.525, saved_model)

    utility = evaluate_sepsis_score(label_directory, prediction_directory,patient_i)
    print("Calculating utility score.............\n")
    print("The utility of LR: ", patient_i)
    print(utility)
    print("\n")
    return float(utility)

def lime_explainer(expert_name,  X_testi, patient_i):
    test_set = np.load('./Candidate_xgb/data/test_set.npy')
    ## sort all the psv files based on the number included in the psv filename (in ascending order)
    test_set = np.sort(test_set,axis = None)
    # Just want to get the psv file name herein
    test_seti = test_set[patient_i]

    if not os.path.exists('Lime_explaination'):
        os.makedirs('Lime_explaination')

    if expert_name == 'RandomForest':
        saved_model =  "./Candidate_rf_svc_lr/saved_model/random_forest.sav"
    elif expert_name == "SVC":
        saved_model =  "./Candidate_rf_svc_lr/saved_model/svc.sav"
    elif expert_name == "LR":
        saved_model =  "./Candidate_rf_svc_lr/saved_model/LR.sav"
    fitted_model = joblib.load(saved_model)
    predict_fn = lambda x: fitted_model.predict_proba(x).astype(float)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_testi.values, mode='classification',feature_names = X_testi.columns, class_names=['Non-Sepsis','Sepsis'] )
    choosen_instance = X_testi.values[0]
    exp = explainer.explain_instance(choosen_instance, predict_fn, num_features=np.shape(X_testi)[1])

    ## save the html
    file_path = './Lime_explaination/'+test_seti.strip('.psv') + '_'+ expert_name+'.html'
    exp.save_to_file(file_path, labels=None, predict_proba=True, show_predicted_value=True )

    # save as txt
    list_of_tuples = exp.as_list()
    txt_filename = './Lime_explaination/'+test_seti.strip('.psv') + '_'+expert_name+'.txt'
    f = open(txt_filename, 'w')
    for t in list_of_tuples:
        line = ','.join(str(x) for x in t)
        f.write(line + '\n')
    f.close()
 

 