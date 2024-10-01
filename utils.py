"""Define ultility functions used by multiple algorithms. """

 
import numpy as np 
 

 
 


 

def sample_offline_policy(mean_rewards, num_contexts, num_actions, pi='eps-greedy', eps=0.1, subset_r = 0.5, 
                contexts=None, rewards=None): 
    """Sample offline actions 

    Args:
        mean_rewards: (num_contexts, num_actions)
        num_contexts: int 
        num_actions: int
        pi: ['eps-greedy', 'subset', 'online']
    """
    if pi == 'subset':
        subset_s = int(num_actions * subset_r)
        subset_mean_rewards = mean_rewards[np.arange(num_contexts), :subset_s]
        actions = np.argmax(subset_mean_rewards, axis=1)
        return actions 
    # focus on this case
    elif pi == 'eps-greedy':
        # This line generates a random action for each context. 
        # The actions are uniformly distributed integers between 0 and num_actions - 1.
        uniform_actions = np.random.randint(low=0, high=num_actions, size=(num_contexts,)) 
        opt_actions = np.argmax(mean_rewards, axis=1)
        # This generates random values between 0 and 1 for each context. 
        delta = np.random.uniform(size=(num_contexts,))
        # This creates a binary selector array where each element is 1 
        # if the corresponding delta is less than or equal to eps (indicating exploration) 
        # and 0 otherwise (indicating exploitation).
        selector = np.array(delta <= eps).astype('float32') 
        '''
        This line combines the exploration and exploitation actions. If selector is 1 (exploration), 
        the action is taken from uniform_actions; otherwise (exploitation), the action is taken from opt_actions.
        '''
        actions = selector.ravel() * uniform_actions + (1 - selector.ravel()) * opt_actions 
        actions = actions.astype('int')
        return actions
    else:
        raise NotImplementedError('{} is not implemented'.format(pi))

# def mixed_policy(num_actions, exp_reward, p_opt, p_uni):
#     num_contexts = exp_reward.shape[0]

#     opt_action = np.argmax(exp_reward, axis=-1).reshape(-1,1) 
#     uni_action = np.random.randint(0, num_actions, size=(num_contexts, 1))
#     subopt_action = ( opt_action + np.random.randint(1, num_actions, size=(num_contexts, 1)) ) % num_actions 
#     sel = np.random.uniform(size=(num_contexts,1)) 
#     sel_opt = np.asarray(sel < p_opt, dtype='float32')
#     sel_uni = np.asarray(1 - sel < p_uni, dtype='float32')
#     sel_non = 1 - sel_opt - sel_uni 
#     off_action = sel_opt * opt_action + sel_uni * uni_action + sel_non * subopt_action
    
#     return off_action.astype('int32')