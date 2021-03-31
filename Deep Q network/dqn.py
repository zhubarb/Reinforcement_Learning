from collections import deque
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import copy
import random
import os
import numpy as np


class Q_learner(object):
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
        
        self.epsilon = init_epsilon
        self.gamma = gamma
        self.experience_pool = deque([], pool_size)
        self.num_actions = environment.action_space.n # number of actions
        self.state_space_dim = environment.observation_space.shape[0] # state space size
        self.hold_out_experiences = None # fill it from the experience pool
        
        self.model = self.build_model(hidden_layer_dim)
        self.frozen_model = copy.deepcopy(self.model) # initialise Q_hat to Q
        

    def build_model(self, hidden_layer_dim):
        '''
        Parameters:
        ------------------
        hidden_layer_dim - the number of nodes in the 1st hidden layer
        '''
        model = Sequential()
        model.add(Dense(hidden_layer_dim, init='glorot_normal', 
                        input_shape=(self.state_space_dim,)))
        model.add(Activation('tanh'))
        
        model.add(Dense(hidden_layer_dim, init='glorot_normal'))
        model.add(Activation('tanh'))
        
        model.add(Dense(self.num_actions, init='glorot_normal'))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs
        
        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)
        
        return model

    def pick_action(self, state, environment):
        '''
        Epsilon-greedy action picker. Given the current state of the environment,
        either pick a random or the best action (with the max Q value) according 
        to the agent.
        '''
        # pick action
        if (random.random() < self.epsilon): #choose random action
            action = environment.action_space.sample()
        else: #choose best action from Q(s,a) values
            qval = self.model.predict(state.reshape(1,self.state_space_dim), batch_size=1)
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
        q_avg = np.apply_over_axes(np.max,q_vals,0).mean()
        
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