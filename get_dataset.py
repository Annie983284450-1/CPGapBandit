# import sys
# import os 

# if os.name == 'nt':
#     # windows
#     neruallcb_path = r'E:\phd thesis ideas\NeuralLCB_C'
# elif os.name == 'posix':
#     if sys.platform == 'darwin':
#         # neruallcb_path = r'C:\Users\98328\Desktop\phd thesis ideas\offline_neural_bandits-main'
#         # macOS
#         neruallcb_path = '/Users/anniezhou/Desktop/NeuralLCB_C'

# if neruallcb_path not in sys.path:
#     sys.path.append(neruallcb_path)

import numpy as np 
import pandas as pd 
from utils import sample_offline_policy
import math


# import tensorflow as tf

def one_hot(df, cols):
    """Returns one-hot encoding of DataFrame df including columns in cols."""
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df
## analyze this one as example


def safe_std(values):
    """
    Remove zero std values for ones.
    using a standard deviation of zero in normalization or scaling operations 
    can lead to division by zero errors or undefined behavior. 
    """
    values = np.array(values)
    values[values==0]=1.0
    return values
    # return np.array([val if val != 0.0 else 1.0 for val in values])
    # return np.array([val if val != 0 else 1.0 for val in values])



def classification_to_bandit_problem_sepsis(contexts, labels, df,num_actions=None):
    """Normalize contexts and encode deterministic rewards."""
    # is not actions is specified, the action is the prediction class

    # Normalize only the continuous features

    # contexts = df.copy()
    
    # Separate categorical and continuous features
    categorical_cols = ['Gender', 'Unit1', 'Unit2', 'ICULOS']  # columns to exclude from normalization, ICULOS specially included
    continuous_cols = [col for col in df.columns if col not in categorical_cols]
    # contexts[continuous_cols] = (df[continuous_cols] - df[continuous_cols].mean()) / df[continuous_cols].std()
    
    if num_actions is None:
        num_actions = np.max(labels) + 1
    num_contexts = contexts.shape[0]

    # Due to random subsampling in small problems, some features may be constant
    # if the features are constant, the std = 0, replace them with 1
    # sstd = safe_std(np.std(contexts, axis=0, keepdims=True)[0, :])

    # sstd = safe_std(np.std(contexts[continuous_cols].to_numpy(), axis=0, keepdims=True)[0, :])
    # Normalize features to have zero mean and unit variance

    #sstd = np.std(contexts[continuous_cols], axis=0).reshape(1, -1)  # Ensuring the result is 2D
    #sstd = safe_std(sstd[0, :])  # Passing it to safe_std as before

    contexts_continuous_np = df[continuous_cols].to_numpy()  # Convert to NumPy array for calculations
    contexts_categorical_np = df[categorical_cols].to_numpy()
    sstd = safe_std(np.std(contexts_continuous_np, axis=0, keepdims=True)[0, :])

    # Normalize features to have zero mean and unit variance
    contexts_continuous_np = ((contexts_continuous_np - np.mean(contexts_continuous_np, axis=0, keepdims=True)) / sstd)
    final_contexts_np = np.concatenate([contexts_continuous_np, contexts_categorical_np], axis=1)


    # One hot encode labels as rewards
    '''
    each row corresponds to a context, and each columnx corresponds to an action
    initialize the rewards to zeros
    selecting the correct action (class) yields reward 1, otherwise 0
    '''
    rewards = np.zeros((num_contexts, num_actions))

    # reward is already been one_hot encoded herein
    rewards[np.arange(num_contexts), labels.ravel().astype(int)] = 1.0

    # return contexts, rewards #, (np.ones(num_contexts), labels)
    # print(f'final_contexts_np:{final_contexts_np}')

    return final_contexts_np, rewards

  

def remove_underrepresented_classes(features, labels, thresh=0.0005):
    """Removes classes when number of datapoints fraction is below a threshold."""

    # Threshold doesn't seem to agree with https://arxiv.org/pdf/1706.04687.pdf
    # Example: for Covertype, they report 4 classes after filtering, we get 7?
    total_count = labels.shape[0]
    unique, counts = np.unique(labels, return_counts=True)
    ratios = counts.astype('float') / total_count
    vals_and_ratios = dict(zip(unique, ratios))
    print('Unique classes and their ratio of total: %s' % vals_and_ratios)
    keep = [vals_and_ratios[v] >= thresh for v in labels]
    return features[keep], labels[np.array(keep)]

import numpy as np
import pandas as pd
'''
If rewards are deterministic (clear cause-effect relationship): You may not need any noise, and you can set noise_std to 0. 
If rewards are stochastic (uncertainty in rewards): 
In cases where the outcome (reward) has some inherent randomness or uncertainty 
(e.g., real-world situations with incomplete information), adding noise can better simulate the real-world conditions. 
In such cases, you could choose a small positive value for noise_std.
start from 0.01 to 0.1 for reward is either 0 or 1 (small values)
'''
class SepsisData(object):
    def __init__(self,  
                is_window,
                num_train_sepsis_pat_win,
                num_test_pat_septic_win,
                num_contexts, 
                num_test_contexts,         
                num_actions=2, 
                noise_std=0.01,
                pi='eps-greedy', 
                eps=0.1, 
                subset_r=0.5
                ):
        """
        Initialize the Sepsis dataset for contextual bandit setting.
        :param file_name: path to the sepsis dataset CSV file.
        :param train_patients_file: path to the numpy file containing patient IDs for training.
        :param test_patients_file: path to the numpy file containing patient IDs for testing.
        :param num_actions: number of actions (for binary sepsis classification, 2 actions: sepsis and non-sepsis).
        :param noise_std: noise to add to the rewards (set to 0 since you don't need noise).
        :param pi: policy to use ('eps-greedy' in this case).
        :param eps: epsilon value for epsilon-greedy policy.
        :param subset_r: subset ratio for selecting part of the data.
        """
        self.name = 'sepsis'
        self.num_contexts = num_contexts 
        self.num_test_contexts = num_test_contexts
        self.num_actions = num_actions
        self.noise_std = noise_std
        self.pi = pi
        self.eps = eps
        self.subset_r = subset_r
        self.num_train_sepsis_pat_win = num_train_sepsis_pat_win
        self.num_test_pat_septic_win = num_test_pat_septic_win
        self.test_wins = []
        self.train_wins = []

        # Is_window = True
        if is_window:
            # abandon the 0-1-0 patients and treat as error
            df = pd.read_csv(f'./data/fully_imputed_8windowed_max48_updated.csv')
            # Drop the column 'HospAdmTim' as per your requirement
            df = df.drop(['HospAdmTime'], axis=1)
            # self.num_test_pat_septic_win =10
            # self.num_train_sepsis_pat_win = 100
            # Balanced training samples ratio (1:1): This can help the model learn to recognize septic cases better 
            # but it may affect generalization, especially if the real-world incidence of sepsis is low.
            self.num_train_nosepsis_pat_win = self.num_train_sepsis_pat_win 
            # Maintain Real-World Distribution for testing dataset
            self.num_test_pat_noseptic_win = math.floor(self.num_test_pat_septic_win * 12)


            sepsis_train_wins = np.load('./data/sepsis_train_wins.npy')
            sepsis_train_wins = sepsis_train_wins.tolist()
            nosepsis_train_wins = np.load('./data/nosepsis_train_wins.npy')
            nosepsis_train_wins = nosepsis_train_wins.tolist()

            
            test_septic_wins = np.load('./data/test_septic_wins.npy')
            test_septic_wins = test_septic_wins.tolist()
            test_noseptic_wins = np.load('./data/test_noseptic_wins.npy')
            test_noseptic_wins = test_noseptic_wins.tolist()
            

            sepsis_train_wins = sepsis_train_wins[0:self.num_train_sepsis_pat_win]
            nosepsis_train_wins = nosepsis_train_wins[0:self.num_train_nosepsis_pat_win]
            test_septic_wins =  test_septic_wins[0:self.num_test_pat_septic_win]
            test_noseptic_wins =  test_noseptic_wins[0:self.num_test_pat_noseptic_win]
            test_wins = []
            train_wins = []
            # print(f'sepsis_train_wins  = {sepsis_train_wins }')
            # print(f'nosepsis_train_wins = {nosepsis_train_wins }')
            # print(f'test_septic_wins  = {test_septic_wins }')
            # print(f'test_noseptic_wins = {test_noseptic_wins }')
            # print(f'sepsis_train_wins.type = {sepsis_train_wins.type}')
            # print(f'nosepsis_train_wins.type = {nosepsis_train_wins.type}')
            # print(f'test_septic_wins.type = {test_septic_wins.type}')
            # print(f'test_noseptic_wins.type = {test_noseptic_wins.type}')
            train_wins = sepsis_train_wins + nosepsis_train_wins
            test_wins = test_septic_wins + test_noseptic_wins
            train_df = df[df['pat_id'].isin(train_wins)]
            test_df = df[df['pat_id'].isin(test_wins)]
            # train_df = train_df[:min(num_contexts, len(train_df))].reset_index(drop=True)
            # test_df = test_df[:min(num_test_contexts, len(test_df))].reset_index(drop=True)
            self.train_wins = train_wins
            self.test_wins = test_wins
            


        else:    
            file_name = './data/fully_imputed.csv'
            # file_name = './data/SepsisData/fully_imputed_8windowed_max48_updated.csv'
            # Load dataset
            df = pd.read_csv(file_name)
            train_patients_file = './data/train_set.npy'
            test_patients_file = './data/test_set.npy'

            # Load train and test patient IDs
            train_patients = np.load(train_patients_file, allow_pickle=True)
            test_patients = np.load(test_patients_file, allow_pickle=True)

            train_patients_ids = [patient.replace('.psv', '') for patient in train_patients]
            test_patients_ids = [patient.replace('.psv', '') for patient in test_patients]



            # Drop the column 'HospAdmTim' as per your requirement
            df = df.drop(['HospAdmTime'], axis=1)
            # print(f'df.head():{df.head()}')
            
            # print(f'train_patients:{train_patients_ids}')
            # print(f'test_patients:{test_patients_ids}')
            # Separate dataset into train and test sets based on patient IDs
            train_df = df[df['pat_id'].isin(train_patients_ids)]
            test_df = df[df['pat_id'].isin(test_patients_ids)]


        train_df = train_df[:min(num_contexts, len(train_df))].reset_index(drop=True)
        test_df = test_df[:min(num_test_contexts, len(test_df))].reset_index(drop=True)
        
        # Extract labels (SepsisLabel) and drop it from the feature set
        train_labels = train_df['SepsisLabel'].to_numpy()
        test_labels = test_df['SepsisLabel'].to_numpy()

        # Drop unnecessary columns
        train_df = train_df.drop(['SepsisLabel', 'pat_id','hours2sepsis'], axis=1)
        test_df = test_df.drop(['SepsisLabel', 'pat_id', 'hours2sepsis'], axis=1)
        # print(f'train_df:{train_df.head()}')
        # print(f'test_df:{test_df.head()}')
        # Convert the remaining columns (features) to numpy arrays
        train_contexts = train_df.to_numpy()
        test_contexts = test_df.to_numpy()
        
        # Convert the classification problem into a contextual bandit problem
        self.train_contexts, self.train_rewards = classification_to_bandit_problem_sepsis(train_contexts, train_labels,train_df,  num_actions)
        self.test_contexts, self.test_rewards = classification_to_bandit_problem_sepsis(test_contexts, test_labels, test_df, num_actions)
        print('////////////AAAAAAAAfter running classification_to_bandit_problem_sepsis().........../////////')
        

        print(f'train_rewards.shape:{self.train_rewards.shape}')
        print(f'test_rewards.shape:{self.test_rewards.shape}')
        
        # Set the context dimension
        self.context_dim = self.train_contexts.shape[1]  # Number of features

        print(f'SepsisData: train_samples: {self.num_train_samples}, test_samples: {self.num_test_samples}')
    
    @property
    def num_train_samples(self):
        return self.train_contexts.shape[0]
    
    @property
    def num_test_samples(self):
        return self.test_contexts.shape[0]

    def reset_data(self, sim_id=0):
        """
        Reset the dataset using random indices for simulation purposes.
        """
        print(f'^^^^^^^^^^^ Running SepsisData.reset_data() ^^^^^^^^^^^^^')
        # No shuffling or leakage is allowed since we are grouping by patient ID
        train_indices = np.arange(self.num_train_samples)
        test_indices = np.arange(self.num_test_samples)
        
        contexts = self.train_contexts[train_indices, :]
        mean_rewards = self.train_rewards[train_indices, :]
        test_contexts = self.test_contexts[test_indices, :]
        test_rewards2D = self.test_rewards[test_indices, :]
        
        # Generate rewards (if needed, noise can be added here)
        rewards = mean_rewards + self.noise_std * np.random.normal(size=mean_rewards.shape)
        
        # Simulate actions based on epsilon-greedy policy

        # !!! so the actions after reset() is already chosen
        actions = sample_offline_policy(mean_rewards, self.num_train_samples, self.num_actions, self.pi, self.eps, self.subset_r, contexts, rewards)
        rewards_selected = rewards[np.arange(len(actions)), actions].reshape(-1, 1)
        dataset = (contexts, actions, rewards_selected, test_contexts, test_rewards2D)
        

        print(f'^^^^^^^^^^^ After running SepsisData.reset_data() ^^^^^^^^^^^^^')
        print(f'contexts.shape:{contexts.shape}')
        print(f'actions.shape:{actions.shape}')
        print(f'rewards.shape:{rewards.shape}')
        print(f'test_contexts.shape:{test_contexts.shape}')
        print(f'test_rewards2D.shape:{test_rewards2D.shape}')
        print('train_wins: ', {self.train_wins})
        print(f'test_wins: {self.test_wins}')
        return dataset,self.train_wins,self.test_wins
    


 