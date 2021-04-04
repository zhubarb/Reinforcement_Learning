from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization

from collections import deque
import copy
import random
import os
import numpy as np
import itertools
import pickle
import scipy


class Agent(object):
    '''
    Generic Reinforcement Learning abstract class.
    '''
    def __init__(self, environment):
        '''
        Parameters:
        -----------
        environment - (emulator class) can be an open AI gym emulator
                      or any other environment that follows the open AI
                      gym format.
        '''
        # state space size
        self.state_space_dim = environment.observation_space.shape[0] 
        # number of actions
        self.num_actions = environment.action_space.n 


    def init_func_approximator(self):
        '''
        Simple look-up table, n-d array or neural network.
        '''

        return None

    def pick_action(self, observation):
        '''
        The agent exists to pick actions, so this is the most important
        method.
        '''
        action = np.random.choice(range(self.num_actions) )

        return action

class TD_Learner(Agent):
    ''' 
    Epsilon-greedy temporal difference (td) learner that is trained based 
    on the equation:
    Q(s,a) <-- Q(s,a) + alpha * [target - prediction], where:
    prediction = Q(s,a), 
    and 
    target = r + gamma * max_a'[Q(s',a')] for Q-learning,
    or
    target = r + gamma * [ (1-epsilon)* max_a'[Q(s',a')] + 
                           epsilon* mean[Q(s',a') |a'!= optimal a'] ] for SARSA.
    
    Attributes:
    ------------ 
    environment      - (emulator Class)   
    learning         - {'off-policy', 'on-policy'} defines whether update_q_table()
                       operates off-policy (i.e. Q-Learning) or on-policy (SARSA)
    epsilon          - (float) the randomness factor. For instance, if this is 0.2,
                       the agent would act randomly 20% of the time and pick
                       the optimal action 80% of the time.
                       This is annealed from 1 to 0.05 during training to allow 
                       some state space exploration.
                       In other words, this adjusts the exploration / exploitation
                       balance.
    learn_rate       - (float) the learning rate (alpha) for the td-update formula 
                       given above.
    gamma            - (float) the future reward discount factor in the td-update 
                       formula given above. Its choice should be informed by
                       average episode (game-play) duration.
    action_set_size  - (int) the number of available actions. By default, if the agent 
                        is used with the tic-tac-toe environment, this is equal to the
                        state dimension size, i.e. 3*3
    q_dict           - (dict) the Q-value lookup table of the td-learner, implemented
                        as a dictionary where keys are tuples of unique states and the
                        values are the available actions per each state.
                        This can either be initialised empty or loaded from an existing
                        pickled q value dictionary.       
    q_table_dir      - (str) the pickled file location of a previously trained
                       Q value dictionary. If this is not None, instead of creating
                       an empty Q value dictionary, the object is initialised by 
                       reading in the pickled dictionary at this location.
    '''

    def __init__(self, environment, learning, epsilon, learn_rate, gamma,
                 q_table_dir = None):
        '''
        Creates a Temporal-Difference Learner object. Depending on the user-defined
        'learning' parameter, the agent either does off-policy (Q-Learning) or 
        on-policy (SARSA) learning.
        '''         
        
        super(TD_Learner, self).__init__(environment)
        self.epsilon = epsilon
        self.final_epsilon = 0.05 # hard-coded, make dynamic
        self.learning_rate = learn_rate
        self.gamma = gamma
        self.learning = learning

        if q_table_dir is None:
            self.q_dict = self.init_func_approximator()
        else:
            self.q_dict = self.load_q_dict(q_table_dir)

    def init_func_approximator(self):
        '''
        Create a Q lookup dict by taking the Cartesian product of all the dimensions in the  
        state space. For the 3*3 tic-tac-toe environment, each cell can have [-1,0,1]. 
        So there are 3**(3*3) configurations. Some of these are not legal game plays,
        e.g.(1,1,1,1,1,1,1,1,1) but for the time being (due to time constraints) we do 
        not worry about these.
        Each product is then used as a tuple key, pointing to a numpy array of size 9, each 
        representing an available location, i.e. action index.
        The lookup dictionary is constrained in the sense that when we know a certain state 
        can not allow an action (i.e. that particular location is already occupied),
        we populate the Q value for that action as np.nan.
        For instance:
        q_dict[(1,1,1,1,1,1,1,1,1)] = array([ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan])
        q_dict[(0,0,1,-1,0,0,0,0,0)] = array([-0.06, -0.04,   nan,   nan, -0.03, -0.03, -0.07, 
                                               0.04,  0.06])
        etc..
        Parameters:
        ------------
        state_dimensions - (list) the state dimensions for the environment that the 
                           agent will interact with. For the tic_tac_toe env, this is 
                           a list of all possible markers [-1,0,1] repeated 3*3 times.           
        '''
        n = self.__num_actions # for brevity below, create temp variable
        q_dict = dict([(element, np.array([(i == 0 and [np.random.uniform(-0.1, 0.1)] 
                                            or [np.nan])[0]for i in element]))
                      for element in itertools.product(*self.state_space_dim )])
        return q_dict
    

    def set_epsilon(self, val):
        '''
        Manually adjust the td_learner agent's epsilon. This is not very clean
        but is done while the td-learner is playing against a manual_agent (human)
        to ensure that we play with a fully exploitative, non-random td-learner agent.
        Parameters:
        ----------------
        val  - (float) the value we want to set the epsilon to
        '''
        self.epsilon = val

    def save_q_dict(self, name):
        '''
        Pickles and saves the current Q-value dictionary of the agent
        with the provided file name.
        This is used to save the results of a trained agent so we can play
        with it without having to retrain.
        Parameters:
        ----------------
        name  - (str) the directory and name that we want to give to the 
                pickled Q dictionary file. Example: 'on_policy_trained.p'
        '''
        with open(name, 'wb') as fp:
            pickle.dump(self.q_dict, fp)

    
    def load_q_dict(self, dir):
        '''
        Loads the pickled dictionary in the provided location as the agent's 
        q-value dictionary play. We can initialise a td-learner in this way and
        play with it directly, without having to retrain a blank one.
        Parameters:
        ----------------
        name  - (str) the directory and name of the pickled Q value dictionary
                file that we want to load. 
        '''
        with open(dir, 'rb') as fp:
            q_dict = pickle.load(fp)

        return q_dict

    def pick_action(self, observation):
        '''
        Pick action in an epsilon-greedy way. By self.epsilon probability 
        this returns a random action, and by (1-epsilon)
        probability it returns the action with the maximum q-value
        for the current environment state.
        Parameters:
        ---------------
        obs - (np.array) the current (board) state to pick an action on.
              For example: np.array([0,0,0,0,0,0,0,0,0]) for an
              empty board. 
        
        '''
        if np.random.rand() < self.epsilon: # random action
            action = np.random.choice(np.where(observation==0)[0])
        else:                               # action with the max q-value
            action = np.nanargmax(self.__get_state_vals(observation))


        return action
 
      
    def update_func_approximator(self, obs, action, reward, next_obs, done, func):
        '''
        Implementation of the temporal difference learning update:
        Q(s,a) <-- Q(s,a) + alpha * [target - prediction].
        where:
        prediction = Q(s,a), 
        and 
        target = r + gamma * max_a'[Q(s',a')] for Q-learning,
        or
        target = r + gamma * [ (1-epsilon)* max_a'[Q(s',a')] + 
                               epsilon* mean[Q(s',a')] for SARSA.
        
        The definition of the target changes depending on whether the learning is done
        off-policy (Q-Learning) or on-policy (SARSA).
        Off-policy (Q-Learning) computes the difference between Q(s,a) and the maximum  
        action value, while on-policy (SARSA) computes the difference between Q(s,a) 
        and the weighted sum of the average action value and the maximum.

        Parameters:
        ---------------
        obs      - (np.array), the state we transitioned from (s).
        action   - (int) the action (a) taken at state=s.
        reward   - (int) the reward (r) resulting from taking the specific action (a)
                   at state = s.
        next_obs - (np.array) the next state (s') we transitioned into
                   after the taking the action at state=s.
        done     - (bool) episode termination indicator. If True, target (above) is
                   only equal to the immediate reward (r) and there is no discounted
                   future reward
        func     - (np.nanmax, np.nanmin) Should update with max if it is the agent's turn  
                   and should take min if the opponent's turn
        '''
        if self.learning == 'off-policy':  # Q-Learning

            if done: # terminal state, just immediate reward
                target = reward
            else: # within episode
                target = reward + self.gamma*func(self.__get_state_vals(next_obs))            
            prediction = self.__get_state_vals(obs)[action]
            updated_q_val = prediction + self.learning_rate *(target - prediction)
            # update the q-value for the observed state,action pair     
            self.__set_q_val(obs, action, updated_q_val)

        elif self.learning == 'on-policy': # SARSA

            if done: # terminal state, just immediate reward
                target = reward
            else: # within episode
                on_policy_q = self.epsilon * np.nanmean(self.__get_state_vals(next_obs)) + \
                              (1- self.epsilon) * func(self.__get_state_vals(next_obs)) 
                target = reward + self.gamma*on_policy_q           
            prediction = self.__get_state_vals(obs)[action]
            updated_q_val = prediction + self.learning_rate *(target - prediction)

            # update the q-value for the observed state,action pair     
            self.__set_q_val(obs, action, updated_q_val)
        else:
            raise ValueError ('Learning method is not known.')


    def __set_q_val(self, state, action, q_val):
        '''
        Set the q value for a state-action pair in the object's q val dictionary.
        Parameters:
        -----------------
        state  -(list) the state index, for a 3*3 board
        action -(int) the action index
        q_val  -(float) the Q value to appoint to the state-action pair
        '''   
        self.q_dict[tuple(state)][action]  = q_val 

 
    def __get_state_vals(self, state):
        '''
        For a given state, look up and return the the action values from 
        the object's q val dictionary.The q values are returned as a dictionary with
        keys equal to action indices and the values the corresponding q values.
        The output is a dictionary to facilitate post-processing and filtering 
        out some q-values that belong to unavailable action locations.

        Parameters:
        -----------------
        state  -(list) the state index, for a 3*3 board
        '''   
        d = self.q_dict[tuple(state)]
        return d 
    
class DQN(Agent):
    '''
    Attributes:
    ---------------------
    num_actions     - the number of actions for the given environment
                        the output layer of the agent's model would have this many nodes
    state_space_dim - the size of the state for the given environment
                        the input layer of the agent's model would have this many nodes
    init_epsilon    - this regulates the probability of choosing a random action 
                        hence balancing exploration versus exploitation.
                        Begin with initial value 1.0, anneal gradually after experience pool collection
                        is complete. Final epsilon value (mainly exploitative) is 0.05 by deafult.
                        Decrement init_epsilon gradually after training starts 
                        The rate with which you anneal init_epsilon is very crucial in the training quality
                        For the CDS Index AMM, we found that we were annealing too early, not allowing the
                        agent to explore sufficiently
    gamma           - The decay coefficient for future rewards
                        Probably the most important parameter, set it to less than 1 for discount.
                        A value of 1 indicates no discount. Gamma decays exponentially e.g. for two time steps
                        discount of an immediate reward of 100 will be 100*gamma^2, and likewise for
                        100 time steps it will be gamma^100. 
                        Choose gamma wisely, depending on the expected and max lengths of an episode.
                        Check gamma_values.ods. Initially I was using gamma = 0.75 for the cartpole problem
                        which is grossly small, check the Excel sheet
    experience_pool_size - the size of the agent's experience pool (i.e. buffer, or memory)
                        This is implemented as a deque and has to be a sufficient size.
                        If the experience pool is too small, this affects learning negatively.I have set it
                        to 1500 to improve results compared to when it was 500. 
    model           - Q, the current version of the agent's function approximator
    frozen_model    - Q_hat, the old version of the agent's function approximator
    '''
    def __init__(self, environment, pool_size, gamma, init_epsilon, hidden_layer_dim):
        '''
        https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        Build and compile keras q-net.
        '''    
        super(DQN, self).__init__(environment)
        
        self.epsilon = init_epsilon
        self.gamma = gamma
        self.experience_pool = deque([], pool_size)
        self.hold_out_experiences = None # fill it from the experience pool
        
        self.model = self.init_func_approximator(hidden_layer_dim)
        self.frozen_model = copy.deepcopy(self.model) # initialise Q_hat to Q
        

    def init_func_approximator(self, hidden_layer_dim):
        '''
        Build the Q function approximator, in this case a Sequential Dense
        keras model with 2 hidden layers. 
        Parameters:
        ------------------
        hidden_layer_dim - the number of nodes in the 1st hidden layer
        '''
        model = Sequential()
        model.add(Dense(hidden_layer_dim, init='glorot_normal', input_shape=(self.state_space_dim,)))
        model.add(Activation('relu'))
        model.add(BatchNormalization() ) 
        
        model.add(Dense(hidden_layer_dim/2, init='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization()  )
        
        model.add(Dense(self.num_actions, init='glorot_normal'))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs
        
        model.compile(loss='mse', optimizer=RMSprop(), metrics= ['accuracy'])
        
        return model

    def pick_action(self, observation):
        '''
        Given the current state of the environment, i.e. observation, either pick
        a random or the best action (with the max Q value) according to the agent.
        '''
        # pick action
        if (random.random() < self.epsilon): #choose random action
            action = np.random.choice(range(self.num_actions))
        else: #choose best action from Q(s,a) values
            qval = self.model.predict(observation.reshape(1,self.state_space_dim), batch_size=1)
            action = (np.argmax(qval))
        return action
    
    def update_frozen_model(self, q_hat_path):
        '''
        Updates the agent's frozen model (Q_hat) by saving the weights of the model to an .h5 file
        and loading them to the frozen model.
        '''
        if not os.path.exists(q_hat_path):  os.makedirs(q_hat_path) # make dir if it does not exists. 
        #frozen_model_path = q_hat_path +  time.strftime("%Y_%m_%d %H_%M_%S")+".h5" # full file path. 
        frozen_model_path = q_hat_path + 'frozen_model.h5'
        # dump the online model weights.   
        self.model.save_weights(frozen_model_path, overwrite=True)   
        # load the online model weights into Q_hat. 
        self.frozen_model.load_weights(frozen_model_path) # load the model weights. 
        #print self.model.get_weights()[0].flatten()[0:5]
        #print self.frozen_model.get_weights()[0].flatten()[0:5] 
    
    def calc_average_q_value(self):
        '''
        Calculate average of the max q values for the hold-out set of experiences.
        (Currently in-sample and copied from the initial experience pool)
        '''
        q_vals = np.zeros((len(self.hold_out_experiences), self.num_actions))
        for i in range( len(self.hold_out_experiences) ):
            s, action, reward, s_prime, termin_st = self.hold_out_experiences[i]
            q_vals[i]  = self.model.predict(s.reshape(1,self.state_space_dim), batch_size=1)
            
        # Average of the MAX q values.    
        q_avg = np.mean(map(lambda x: max(x), q_vals))
        
        return q_avg
       
    def sample_mini_batch(self, batchSize): 
        '''
        Sample minibatch from self.experience pool for training self.model
        The function outputs X_train and y_train
        This is the crux of the Q_net algorithm
        ''' 
        #randomly sample our experience replay memory
        minibatch = random.sample(self.experience_pool, batchSize)
        X_train = []
        y_train = []
        
        for experience in minibatch: # for each experience in minibatch, calc target and pred Q
            #Get max_Q(S',a)
            s, action, reward, s_prime, termin_st = experience
            pred_Q_vals = self.model.predict(s.reshape(1,self.state_space_dim), batch_size=1)# calc by current Q net
            target_Q_vals = self.frozen_model.predict(s_prime.reshape(1,self.state_space_dim), batch_size=1)# calc by frozen Q net            
            maxQ = np.max(target_Q_vals)
            y = np.zeros((1,self.num_actions))
            y[:] = pred_Q_vals[:]
            if not termin_st : #non-terminal state
                update = (reward + (self.gamma * maxQ))
            else: #terminal state
                update = reward
            y[0][action] = update
            X_train.append(s.reshape(self.state_space_dim,))
            y_train.append(y.reshape(self.num_actions,))  
                     
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        return X_train, y_train
    
    def update_func_approximator(self):
        '''
        Train the agent's fucntion approximator.
        TO DO
        '''
        pass

class PG_Learner(Agent):

    def __init__(self, environment, learning_rate, lr_decay, gamma, model_dir=None):
        '''
        http://karpathy.github.io/2016/05/31/rl/
        '''
        super(PG_Learner, self).__init__(environment)
        self.gamma = gamma # episode label calc reward decay
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay # model optimizer learning rate global decay param

        # episodic containers
        self.aprobs = list() # episodic predictions of the func approximator
        self.dlogps = list() # episodic gradients of the func approximator
        self.drs = list() # episodic rewards from the emulator
        self.xs = list() # observed states from an episode

        # batch containers
        self.batch_labels = list()
        self.batch_obs = list()

        # initialise or load the neural net
        self.model = self.init_func_approximator()
        if model_dir is not None:
            self.model.load_weights(model_dir)


    def init_func_approximator(self):
        '''
        Build keras model given the number of input (predictive) and output
        features, along wiht the weight decay (shrinkage) mode.
        Assumes classification (so output activation is either sigmoid or softmax)
        initialisations: https://keras.io/initializations/
        activation funcs: https://keras.io/activations/
        objective funcs: https://keras.io/objectives/
        Parameters:
        -------------
        lr_decay      - (float) Between 0 and 1, learning rate decay over each update.    
        '''
    
        model = Sequential()

        model.add(BatchNormalization(input_shape=(self.state_space_dim,)))

        # 1st hidden layer
        model.add(Dense(100,
                        activation='relu', 
                        init='glorot_normal'))
        model.add(BatchNormalization())

        # Output layer
        output_dim =1
        model.add(Dense(output_dim, activation='sigmoid', 
                        init='glorot_normal')) 
       
        # Compile the architecture
        model.compile(loss='mse',
                      optimizer= RMSprop(lr= self.learning_rate) ,
                      metrics=['accuracy'])

        return model


    def update_func_approximator(self):
        '''
        Mini-batch update the policy function approximator of the PG_Learner object with
        the mini-batch observations and labels
        Parameters:
        -----------
        batch_obs   - (list of np.arrays) where each array conains the observations from a
                      single episode, these are concatenated within the method to form the 
                      mini-batch observations
        batch_labels- (list of np.arrays) where each array contains the labels from a single
                      episode, these are concatenated within the method t form the mini-batch
                      labels
        '''
        X_train = np.concatenate(self.batch_obs)
        y_train = np.concatenate(self.batch_labels)

        self.model.train_on_batch(x= X_train, 
                                  y= y_train)
        #self.model.fit(X_train, y_train, nb_epoch=1, batch_size=y_train.shape[0])
        
        # flush batch-containers after they have been used
        self.batch_labels, self.batch_obs = [], []

    def discount_rewards_cartpole(self, r, N):
        """ 
        Credit Assignment method, specific to the CartPole open AI gym 
        environment.
        Make last reward -1, decay it for N ticks. For the rest of the rewards, 
        leave as is, i.e. 1.
        Parameters:
        -------------
        r - (np.array) stacked vertical rewards array containing all episode
            rewards - which will be 1's for the cartpole environment 
        N - (int) the number of ticks to decay and roll back the negative reward
            of -1, i.e. blame the pole falling on the past N actions
        """

        discounted_r = np.zeros_like(r)
        # roll back final -1 reward back for N ticks
        rollback_lim = max(0, r.size-N )
        running_add= -1
        for t in reversed(range(0, r.size)):
            if t >= rollback_lim:
                running_add = running_add * self.gamma 
                discounted_r[t] = running_add
            else: 
                discounted_r[t] = r[t]
        return discounted_r

    def pick_action(self, observation):
        '''
        Forward-pass through the poilcy network (self.model) to get the
        action probabilities.
        '''
        aprob = self.model.predict(observation)
        self.aprobs.append(aprob)

        # Choose action from stochastic policy, according to softmax probs
        if len(aprob) == 1: # sigmoid
            action_idx = np.random.choice(a= self.num_actions, 
                                      p = [1-aprob.flatten()[0], aprob.flatten()[0]] )
            action = int(action_idx)
            #action = 1 if np.random.uniform() < aprob else 0       
        else: # softmax
            action_idx = np.random.choice(a= self.num_actions, 
                                          p = aprob.flatten() )
            action = np.zeros_like(aprob); 
            action[action_idx] = 1
        
        # add action-aporb grad to deriv log probs container
        self.append_grad(aprob, action)

        return action

    def record_episode_obs_n_rewards(self, x, reward):
        '''
        Parameters:
        -----------
        x      - (np.array) the (prev) observation, reshaped to be inputable to
                 self.model.
        reward - (float) the reward observed from the emulator for the current
                 tick
        '''
        self.xs.append(x)
        self.drs.append(reward)

    def append_grad(self, aprob, action):
        ''' 
        grad that encourages the action that was taken to be taken 
        (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        '''
        self.dlogps.append(action-aprob)

    def append_to_minibatch(self):
        '''
        Called at the end of each episode, consolidates the episode observations
        and episode labels (calculated by gradients + discounted_rewards), appending
        them to the batch containers.
        '''
        # calc episode labels
        eplabels = self.calc_episode_labels(10, True)
        
        # append to batch labels and observations
        self.batch_labels.append(eplabels)
        self.batch_obs.append(np.vstack(self.xs) )

        # done with the containers for this episode, flush them.
        self.reset_epis_containers()

    def calc_episode_labels(self, N, standardize_reward):
        '''
        Calculate the action labels based on the gradients calculated and the 
        discounted rewards appointed to each of the actions.
        Parameters:
        ------------
        N                 - (int) the number of ticks to decay and roll back the 
                             negative reward of -1, i.e. blame the pole falling on
                             the past N actions
        standardize_reward- (bool) whether to mean and variance scale the episode 
                            rewards
        '''

        # compute the discounted reward backwards through time
        epr = np.vstack(self.drs)
        discounted_epr = self.discount_rewards_cartpole(epr, N)

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        if standardize_reward:
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)     
                           
        # modulate the gradient with advantage (PG magic happens right here.) 
        # Keras-hack, define y as predictions on X_train plus error
        predictions = np.vstack(self.aprobs)
        reinforcements = ( np.vstack(self.dlogps) * discounted_epr )

        ep_y = predictions + reinforcements

        return ep_y

    def reset_epis_containers(self):
        '''
        Flush the episodic probability (aprobs), gradient (dlogps), observation (xs),
        and rewards (drs) containers
        '''
        self.aprobs = [] # episode predictions
        self.dlogps = [] # episode gradients
        self.xs = []     # episode observations
        self.drs = []    # episode rewards

    def save_weights(self, name,  dir=None):
        '''
        Save model weights to 'dir' location
        Parameters:
        -------------
        name  - (str) the name to use while saving the model
        dir   - (str) the directory to save the model in
        '''
        if dir is None:
            dir = os.getcwd()

        self.model.save_weights(os.path.join(dir, name) + '.h5')

class PG_Learner_softmax(Agent):

    def __init__(self, environment, learning_rate, lr_decay, gamma, model_dir=None):
        '''
        http://karpathy.github.io/2016/05/31/rl/
        '''
        super(PG_Learner_softmax, self).__init__(environment)
        self.gamma = gamma # episode label calc reward decay
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay # model optimizer learning rate global decay param

        # episodic containers
        self.aprobs = list() # episodic predictions of the func approximator
        self.dlogps = list() # episodic gradients of the func approximator
        self.drs = list() # episodic rewards from the emulator
        self.xs = list() # observed states from an episode
        self.acts = list() # vectorised action list for a single episode

        # batch containers
        self.batch_labels = list()
        self.batch_obs = list()

        # initialise or load the neural net
        self.model = self.init_func_approximator()
        if model_dir is not None:
            self.model.load_weights(model_dir)


    def init_func_approximator(self):
        '''
        Build keras model given the number of input (predictive) and output
        features, along wiht the weight decay (shrinkage) mode.
        Assumes classification (so output activation is either sigmoid or softmax)
        initialisations: https://keras.io/initializations/
        activation funcs: https://keras.io/activations/
        objective funcs: https://keras.io/objectives/
        Parameters:
        -------------
        lr_decay      - (float) Between 0 and 1, learning rate decay over each update.    
        '''
    
        model = Sequential()

        model.add(BatchNormalization(input_shape=(self.state_space_dim,)))

        # 1st hidden layer
        model.add(Dense(100,
                        activation='relu', 
                        init='glorot_normal'))
        model.add(BatchNormalization())
        
        # Output layer
        model.add(Dense(self.num_actions, activation='softmax', 
                        init='glorot_normal')) 
       
        # Compile the architecture
        model.compile(loss='mse',
                      optimizer= RMSprop(lr= self.learning_rate) ,
                      metrics=['accuracy'])

        return model


    def update_func_approximator(self):
        '''
        Mini-batch update the policy function approximator of the PG_Learner object with
        the mini-batch observations and labels
        Parameters:
        -----------
        batch_obs   - (list of np.arrays) where each array conains the observations from a
                      single episode, these are concatenated within the method to form the 
                      mini-batch observations
        batch_labels- (list of np.arrays) where each array contains the labels from a single
                      episode, these are concatenated within the method t form the mini-batch
                      labels
        '''
        X_train = np.concatenate(self.batch_obs)
        y_train = np.concatenate(self.batch_labels)

        self.model.train_on_batch(x= X_train, 
                                  y= y_train)
        #self.model.fit(X_train, y_train, nb_epoch=1, batch_size=y_train.shape[0])
        
        # flush batch-containers after they have been used
        self.batch_labels, self.batch_obs = [], []

    def discount_rewards_cartpole_old(self, r, N):
        """ 
        Credit Assignment method, specific to the CartPole open AI gym 
        environment.
        Make last reward -1, decay it for N ticks. For the rest of the rewards, 
        leave as is, i.e. 1.
        Parameters:
        -------------
        r - (np.array) stacked vertical rewards array containing all episode
            rewards - which will be 1's for the cartpole environment 
        N - (int) the number of ticks to decay and roll back the negative reward
            of -1, i.e. blame the pole falling on the past N actions
        """

        discounted_r = np.zeros_like(r)
        # roll back final -1 reward back for N ticks
        rollback_lim = max(0, r.size-N )
        running_add= -1
        for t in reversed(range(0, r.size)):
            if t >= rollback_lim:
                running_add = running_add * self.gamma 
                discounted_r[t] = running_add
            else: 
                discounted_r[t] = r[t]
        return discounted_r

    def discount_rewards_cartpole(self, r, N):
        """ 
        Credit Assignment method, specific to the CartPole open AI gym 
        environment.
        Taken from John Schulman's modular_rl-master code, misc.utils.discount()
        Parameters:
        -------------
        r - (np.array) stacked vertical rewards array containing all episode
            rewards - which will be 1's for the cartpole environment 
        N - (int) the number of ticks to decay and roll back the negative reward
            of -1, i.e. blame the pole falling on the past N actions
        """

        discounted_r = scipy.signal.lfilter([1],[1,-self.gamma],
                                            r[::-1], axis=0)[::-1]
        return discounted_r


    def pick_action(self, observation):
        '''
        Forward-pass through the policy network (self.model) to get the
        action probabilities.
        '''
        aprob = self.model.predict(observation)
        self.aprobs.append(aprob)

        # Choose action from stochastic policy, according to softmax probs
        if len(aprob.flatten()) == 1: # sigmoid output layer
            action_idx = np.random.choice(a= self.num_actions, 
                                      p = [1-aprob.flatten()[0], aprob.flatten()[0]] )
            self.acts.append(np.array([action_idx]) )
            #action = 1 if np.random.uniform() < aprob else 0       
        else: # softmax
            action_idx = np.random.choice(a= self.num_actions, 
                                          p = aprob.flatten() )
            action = np.zeros_like(aprob); 
            action[0][action_idx] = 1
            self.acts.append(action)
        
        # add action-aporb grad to deriv log probs container
        self.append_grad(aprob, action)

        return int(action_idx)

    def record_episode_obs_n_rewards(self, x, reward):
        '''
        Parameters:
        -----------
        x      - (np.array) the (prev) observation, reshaped to be inputable to
                 self.model.
        reward - (float) the reward observed from the emulator for the current
                 tick
        '''
        self.xs.append(x)
        self.drs.append(reward)

    def append_grad(self, aprob, action):
        ''' 
        grad that encourages the action that was taken to be taken 
        (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        '''
        self.dlogps.append(action-aprob)

    def append_to_minibatch(self):
        '''
        Called at the end of each episode, consolidates the episode observations
        and episode labels (calculated by gradients + discounted_rewards), appending
        them to the batch containers.
        '''
        # calc episode labels
        eplabels = self.calc_episode_labels(N=10, standardize_reward=True)
        
        # append to batch labels and observations
        self.batch_labels.append(eplabels)
        self.batch_obs.append(np.vstack(self.xs) )

        # done with the containers for this episode, flush them.
        self.reset_epis_containers()

    def calc_episode_labels(self, N, standardize_reward):
        '''
        Calculate the action labels based on the gradients calculated and the 
        discounted rewards appointed to each of the actions.
        Parameters:
        ------------
        N                 - (int) the number of ticks to decay and roll back the 
                             negative reward of -1, i.e. blame the pole falling on
                             the past N actions
        standardize_reward- (bool) whether to mean and variance scale the episode 
                            rewards
        '''

        # compute the discounted reward backwards through time
        epr = np.vstack(self.drs)
        discounted_epr = self.discount_rewards_cartpole(epr, N)

        # standardize the rewards to be unit normal (helps control the gradient 
        # estimator variance)
        if standardize_reward:
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)     
                           
        # modulate the gradient with advantage (PG magic happens right here.) 
        # Keras-hack, define y as predictions on X_train plus error
        predictions = np.vstack(self.aprobs)
        vectorised_actions = np.vstack(self.acts)# e.g. action = 1 would be [0,1]
        vectorised_discounted_epr = vectorised_actions * discounted_epr
        reinforcements = ( np.vstack(self.dlogps) * vectorised_discounted_epr )

        ep_y = predictions + reinforcements

        return ep_y

    def reset_epis_containers(self):
        '''
        Flush the episodic probability (aprobs), gradient (dlogps), observation (xs),
        and rewards (drs) containers
        '''
        self.aprobs = [] # episode predictions
        self.dlogps = [] # episode gradients
        self.xs     = [] # episode observations
        self.drs    = [] # episode rewards
        self.acts   = [] # vectorised episode actions

    def save_weights(self, name,  dir=None):
        '''
        Save model weights to 'dir' location
        Parameters:
        -------------
        name  - (str) the name to use while saving the model
        dir   - (str) the directory to save the model in
        '''
        if dir is None:
            dir = os.getcwd()

        self.model.save_weights(os.path.join(dir, name) + '.h5')
