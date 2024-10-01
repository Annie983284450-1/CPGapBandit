
import numpy as np

class OfflineContextualBandit(object):
    def __init__(self, contexts, actions, rewards, test_contexts, test_mean_rewards):
        """
        Args:
            contexts: (None, context_dim) 
            actions: (None,) 
            mean_rewards: (None, num_actions) 
            test_contexts: (None, context_dim)
            test_mean_rewards: (None, num_actions)
        """
        self.contexts = contexts
        self.actions = actions
        self.rewards = rewards
        self.test_contexts = test_contexts
        self.test_mean_rewards = test_mean_rewards
        self.order = range(self.num_contexts) 

    '''
    shuffle the order in which contexts (and their associated rewards and actions) are presented
    to the bandit algorithms. simulte a more realistic scenario where the order of the encountering
    different conexts is not fixed but random. 

    We cannot do this on Sepsis dataset, for there might be data leakage??
    '''
    def reset_order(self): 
        # np.permutation ensures that each interger is unique
        # range from 0 to num_contexts - 1
        self.order = np.random.permutation(self.num_contexts)
    # def reset_order(self, sim): 
        # np.random.seed(sim)
    #     self.order = np.random.permutation(self.num_contexts)

    def get_data(self, number): 
        ind = self.order[number]
        a = self.actions[ind]
        # The expression self.rewards[ind:ind+1, a:a+1] is used to select the reward for the chosen action
        return self.contexts[ind:ind+1], self.actions[ind:ind+1], self.rewards[ind:ind+1, a:a+1] 

 
        
        
    @property 
    def num_contexts(self): 
        return self.contexts.shape[0] 

    @property 
    def num_actions(self):
        return self.test_mean_rewards.shape[1] 
    
    @property  
    def context_dim(self):
        return self.contexts.shape[1]

    @property 
    def num_test_contexts(self): 
        return self.test_contexts.shape[0]
