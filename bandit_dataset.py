"""Define a data buffer for contextual bandit algorithms. """

import numpy as np 


class BanditDataset(object):
    """Append new data and sample random minibatches. """
    def __init__(self, context_dim, num_actions, buffer_s = -1, name='bandit_data'):
        """
        Args:
            buffer_s: Buffer size. Only last buffer_s will be used. 
                If buffer_s = -1, all data is used. 
                the default value of buffer_s is -1
        """
        self.name = name 
        self.context_dim = context_dim 
        self.num_actions = num_actions 
        self.buffer_s = buffer_s 
        self.contexts = None # An array of d-dim contexts
        self.actions = None # An array of actions 
        self.rewards = None # An array of one-hot rewards 

    def reset(self):
        self.contexts = None 
        self.actions = None 
        self.rewards = None 

    def add_with_onehot(self, context, action, reward):
        """Add one or multiple samples to the buffer. 

        Args:
            context: An array of d-dimensional contexts, (None, context_dim)
            action: An array of integers in [0, K-1] representing the chosen action, (None,1)
            reward: An array of real numbers representing the reward for (context, action), (None,1)
             
     
        """
        # the use of JAX allows for Jax's accelerated computing capabilities
        #  The -1 argument tells numpy or JAX to automatically calculate the number of rows such that the total size of the array remains constant. 
        # each row has self.conext_dim columns
        c = context.reshape(-1, self.context_dim) 
        if self.contexts is None: # first batch of context is being added
            self.contexts = c 
        else:  #  vertically stack the existing contexts with new batch of contexts (c)
            self.contexts = np.vstack((self.contexts, c))
        
        # action.ravel() would flatten action array
        # jax.nn.one_hot(action.ravel(), self.num_actions) converts each action into a one-hot encoded vector
        # where the length of each vector is equal to "num_actions" 
        '''
        if action = [0,1,0,1]
        jax.nn.one_hot(action.ravel(), self.num_actions)  = 
        [
        [1,0],
        [0,1],
        [1,0],
        [0,1]
        ]
        reward.reshape(-1,1):
        This reshapes the reward array into a 2D array with one column. 
        The -1 instructs the reshape operation to automatically determine the number of rows based on the length of reward, 
        ensuring that each reward value gets its own row.
        '''
        print(f'&&&&&&&&&&&& adding new (c,a,r)')
        r = jax.nn.one_hot(action.ravel(), self.num_actions) * reward.reshape(-1,1)
        print(f'%%%%%%%%% reward.shape after nn_one_hot: {r.shape} %%%%%%%%%')
        # similar as adding contexts
        if self.rewards is None: 
            self.rewards = r 
        else: 
            self.rewards = np.vstack((self.rewards, r)) 

        if self.actions is None: 
            self.actions = action.reshape(-1,1) 
        else:
            self.actions = np.vstack((self.actions, action.reshape(-1,1)))

    def add(self, context, action, reward):
        """Add one or multiple samples to the buffer. 

        Args:
            context: An array of d-dimensional contexts, (None, context_dim)
            action: An array of integers in [0, K-1] representing the chosen action, (None,1)
            reward: An array of real numbers representing the reward for (context, action), (None,1)
             
     
        """
        # the use of JAX allows for Jax's accelerated computing capabilities
        #  The -1 argument tells numpy or JAX to automatically calculate the number of rows such that the total size of the array remains constant. 
        # each row has self.conext_dim columns
        c = context.reshape(-1, self.context_dim) 
        if self.contexts is None: # first batch of context is being added
            self.contexts = c 
        else:  #  vertically stack the existing contexts with new batch of contexts (c)
            self.contexts = np.vstack((self.contexts, c))
        
        # action.ravel() would flatten action array
        # jax.nn.one_hot(action.ravel(), self.num_actions) converts each action into a one-hot encoded vector
        # where the length of each vector is equal to "num_actions" 
        '''
        if action = [0,1,0,1]
        jax.nn.one_hot(action.ravel(), self.num_actions)  = 
        [
        [1,0],
        [0,1],
        [1,0],
        [0,1]
        ]
        reward.reshape(-1,1):
        This reshapes the reward array into a 2D array with one column. 
        The -1 instructs the reshape operation to automatically determine the number of rows based on the length of reward, 
        ensuring that each reward value gets its own row.
        '''
        print(f'&&&&&&&&&&&& adding new (c,a,r)')
        r = jax.nn.one_hot(action.ravel(), self.num_actions) * reward.reshape(-1,1)
        # r = reward.reshape(-1, 1)  # This ensures the reward is stored as (n, 1), either 1 or 0
        # print(f'%%%%%%%%% reward.shape after reshape: {r.shape} %%%%%%%%%')
        print(f'%%%%%%%%% reward.shape after nn_one_hot: {r.shape} %%%%%%%%%')
        # similar as adding contexts
        if self.rewards is None: 
            self.rewards = r 
        else: 
            self.rewards = np.vstack((self.rewards, r)) 

        # Reshape and store the action
        action = action.reshape(-1, 1)  # Ensure the action is stored as (n, 1)
        if self.actions is None: 
            self.actions = action.reshape(-1,1) 
        else:
            self.actions = np.vstack((self.actions, action.reshape(-1,1)))

        # print(f'&&&&&&&&&&&& After adding new (c,a,r)')
        # print(f'&&&&&&& context.shape: {self.contexts.shape}')
        # print(f'&&&&&&& action.shape: {self.actions.shape}')
        # print(f'&&&&&&& reward.shape:{self.rewards.shape}')
        # &&&&&&& context.shape: (1, 13)
        # &&&&&&& action.shape: (1, 1)
        # &&&&&&& reward.shape:(1, 2)


    def get_batch_with_weights(self, batch_size):
        """
        Return:
            x: (batch_size, context_dim)
            w: (batch_size, num_actions)
            y: (batch_size, num_actions)
        """
        
        self.contexts = np.array(self.contexts)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)


        n = self.num_samples 
        if self.buffer_s == -1:
            ind = np.random.choice(range(n), batch_size) 
        else: 
            ind = np.random.choice(range(max(0, n - self.buffer_s), n), batch_size)
        ind = np.array(ind)
        print(f'................. Testing get_batch_with_weights .................')
        # print(f'batch_size:{batch_size}')
        # # print("Contexts shape:", self.contexts.shape)
        # print("Actions shape:", self.actions.shape)
        # print("Rewards shape:", self.rewards.shape)
        # print("Index shape:", ind.shape)

        # context_batch = self.contexts[ind, :]  # Check if this works
        # print(f'context_batch:{context_batch}')
        # sys.exit()
        # action_one_hot = jax.nn.one_hot(self.actions[ind, :].ravel(), self.num_actions)  # Check if this works
        # reward_batch = self.rewards[ind, :]  # Check if this works
        

        return self.contexts[ind, :], jax.nn.one_hot(self.actions[ind,:].ravel(), self.num_actions), self.rewards[ind, :]
    '''
    get_batch():
    retrieves a batch of data for training or evaluation. 
    selects batch_size samples of contexts (x), actions (a), and corresponding rewards (y) from the dataset. 
    The method supports both random sampling and sequential sampling of the most recent data based on the rand parameter 
    and the buffer_s attribute, 
    which controls the size of the buffer used for sampling. 
    ensures that batches of data can be efficiently drawn for the iterative training processes  
    '''
    # when called in NeuralBanditModelV2(): x,a,y = data.get_batch(self.hparams.batch_size, self.hparams.data_rand) 
    # flags.DEFINE_integer('batch_size', 32, 'Batch size')
    # flags.DEFINE_integer('buffer_s', -1, 'Size in the train data buffer.')


    def get_batch(self, batch_size, rand=True):
        """
        Return:
            x: (batch_size, context_dim)
            a: (batch_size, )
            y: (batch_size, )
        """        
        self.contexts = np.array(self.contexts)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)        
        # print(f'||||||||||||||||||||data shapes before runing get_batch():')
        # print(f'self.contexts.shape:{self.contexts.shape}')
        # print(f'self.actions.shape:{self.actions.shape}')
        # print(f'self.rewards.shape:{self.rewards.shape}')

        # print(f'................. Start Running get_batch().................')

        # available_samples
        n = self.num_samples 
        
        assert n > 0 
        # if rand:
        #     if self.buffer_s == -1:
        #         ind = np.random.choice(n, batch_size) 
        #     else: 
        #         ind = np.random.choice(range(max(0, n - self.buffer_s), n), batch_size)
        # else:
        #     ind = range(n - batch_size,n)


        if rand:
            if self.buffer_s == -1:
                
                ind = np.random.choice(range(n), min(batch_size, n), replace=False)
                # ind = np.random.choice(n, batch_size) 
            else: 
                ind = np.random.choice(range(max(0, n - self.buffer_s), n), min(batch_size, n), replace=False)

                # ind = np.random.choice(range(max(0, n - self.buffer_s), n), batch_size)
        else:
            ind = range(n - batch_size,n)

        a = self.actions[ind,:].ravel().astype(int)
        rewards_batch = self.rewards[ind, :]  # First select the rows using ind
        rewards_selected = rewards_batch[np.arange(len(a)), a].reshape(-1, 1)
        '''
        debug log
        ||||||||||||||||||||data shapes AFTER runing get_batch():
        context_batch shape:(32, 13)
        a = self.actions[ind,:].ravel().astype(int):(32,)
        rewards_batch shape:(32, 1)
        Rewards selected shape: (32, 32)
        step:62
        a is a 1D array of shape (32,), containing the chosen actions for each context.
        When you use rewards_batch[:, a], 
        you are selecting multiple columns (corresponding to the actions in a) for each row, 
        which results in a shape of (32, 32).

        '''
        # rewards_selected = rewards_batch[:, a]
        
        context_batch = self.contexts[ind, :]
        # Debugging: print the shapes to verify consistency
        # print("Context batch shape:", context_batch.shape)
        # print("Actions shape:", a.shape)
        # print(f'actions:{a}')
        # print("Rewards batch shape:", rewards_batch.shape)

        # print(f'||||||||||||||||||||data shapes AFTER runing get_batch():')
        # print(f'context_batch shape:{context_batch.shape}')
        # print(f'a = self.actions[ind,:].ravel().astype(int):{a.shape}')
        # print(f'rewards_batch shape:{rewards_batch.shape}')
        # print("Rewards selected shape:", rewards_selected.shape)
        
        # print(f'|||||||||||||||||||| get_batch() finished!!!!!!!!!!!!!!!!')
        
        return context_batch, a, rewards_selected 

        # return context_batch, a, rewards_batch


    @property 
    def num_samples(self): 
        return 0 if self.contexts is None else self.contexts.shape[0]


# def test_BanditDataset():
#     context_dim = 5 
#     num_actions = 2 
#     n = 8 
#     bd = BanditDataset(context_dim, num_actions)


#     contexts = np.random.uniform(size=(n, context_dim)) 
#     actions = np.random.randint(0,num_actions, (n,1)) 
#     rewards = np.random.randn(n,1) 
#     print('Testing Banditdataset ..........................')

#     print('contexts.shape, actions.shape, rewards.shape:')
#     print(contexts.shape, actions.shape, rewards.shape)

#     bd.add(contexts, actions, rewards) 
#     print("bd.contexts.shape, bd.actions.shape, bd.rewards.shape:")
#     print(bd.contexts.shape, bd.actions.shape, bd.rewards.shape)

#     print(bd.actions)
#     print(bd.rewards)
#     print('===================================')
#     c,w,r = bd.get_batch_with_weights(batch_size=1)
#     # print(f'c.shape, w.shape, r.shape:{c.shape},{w.shape}, {r.shape}')
#     print(w)
#     print(r)


# test_BanditDataset()