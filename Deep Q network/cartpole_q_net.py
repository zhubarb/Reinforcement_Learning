import numpy as np
import traceback
from datetime import datetime
import os 
import pandas as pd
import random
import gym
from dqn import Q_learner

def run_experiment(nr_episodes, gamma, init_epsilon, batchSize, poolSize, 
                   hidden_layer_dim, freeze_every, q_hat_pathm, monitor):
    '''
    Run a single experiment
    Parameters:
    ----------------------- 
    nr_episodes       - number of episodes to run the experiment for after experience 
                        pool is full
    gamma             - the discount factor for the future rewards
    init_epsilon           - the randomness factor with which the agent picks random actions,
                        i.e. explores.
    batchSize         - batch size for the stochastic gradient descent training
    poolSize          - the pool size (deque) for the experience replay
    hidden_layer_dim  - the number of nodes in the 1st hidden layer
    freeze_every      - save q-hat with this interval
    q_hat_path        - the path to save the frozen deep net (q-hat) 
    monitor           - (bool) whether or not to environment.monitor
    '''
    # Needed for reproducibility of results, set seeds
    np.random.seed(1) # seed has to be set before Keras imports for reproducibility
    np.random.RandomState(1) # allows to initialise with same coeffs and play same episodes
    random.seed(1) # set seed of the Python package for generating random numbers
    
    # environment variables
    print('Current directory is ' + os.getcwd())
    env = gym.make('CartPole-v0') # https://gym.openai.com/envs/CartPole-v0
    if monitor:
        env.monitor.start('/tmp/cartpole-experiment-'+ datetime.now().strftime("%Y-%m-%d_%H.%M"))
    print(env.action_space) # print action space for current environment
    print(env.observation_space) # print state space for current environment
    state_space_dim = env.observation_space.shape[0] 
    
    # Create agent for this experiment
    agent = Q_learner(env, poolSize, gamma, init_epsilon, hidden_layer_dim) # create new q-learner
    assert len(agent.model.get_weights()[0].flatten()) == state_space_dim * hidden_layer_dim # first layers weights 
    
    epis_durations = list()
    q_stat_interval = int(nr_episodes)/10 # calc q-values on the hold-out set 10 times during training
    avg_q_vals = list()
    # start experience collection and training
    i = 0 # episode counter
    while i <= nr_episodes: # loop through episodes
        if i==0:message= 'Collected %i experiences of %i'%(len(agent.experience_pool), poolSize)  
        else: message = "Training Episode %i of %i"%(i, nr_episodes)
        print(message)
        done = False

        state = env.reset() # initial state for this episode
        t=0 # time step counter
        while not done: # loop through time points in a single episode
            if render:
                env.render()
            if verbose: print('Cur state: ' + str(state) )
            
            action = agent.pick_action(state, env)
            # carry out action and save experience
            nxt_state, reward, done, info = env.step(action) # reward +1 for each t that pole remains upright
            agent.experience_pool.append([state, action, reward, nxt_state, done]) # exp = {s,a,r,s',termin_st}
            state = nxt_state # move to the next state
            
            # experience generation complete, start training
            if len(agent.experience_pool) >= poolSize: 
                
                # Update the agent's frozen model, i.e. Q_hat
                if i % freeze_every == 0: 
                    agent.update_frozen_model(q_hat_path)
                    
                # Crux, draw minibatch from agent's experience_pool
                X_train, y_train = agent.sample_mini_batch(batchSize)
                
                # Train agent's model based on the minibatch drawn
                agent.model.train_on_batch(X_train, y_train)
                
                # Decrement init_epsilon gradually after training starts 
                # The rate with which you anneal init_epsilon is very crucial in the training quality
                # For the CDS Index AMM, we found that we were annealing too early, not allowing the
                # agent to explore sufficiently
                if agent.epsilon > final_epsilon:  agent.epsilon -= 1.0/ (nr_episodes/0.05)
            
            # only increment if experience pool is full and an episode ended    
            if done and len(agent.experience_pool) >= poolSize:  
                     
                epis_durations.append(t)
                
                # Create hold-out set to calculate q-value stats
                if agent.hold_out_experiences is None:
                    agent.hold_out_experiences = list(agent.experience_pool)[0:100]
                
                if i %q_stat_interval == 0:
                    avg_q_val = agent.calc_average_q_value() 
                    avg_q_vals.append( avg_q_val)
                    print('   Current average q-value is %f'%(avg_q_val))
                
                i += 1 
                
                
            t += 1  
        # CartPole-v0 defines solving as getting avg reward of 195 over 100 consec trials    
        mov_avg = np.mean(epis_durations[max(0,i-100):len(epis_durations)]) 
        print ('   Current mavg is %f, init_epsilon is %f, last episode duration was %i.'%(mov_avg,
                                                                np.round(agent.epsilon,2), t))
        

    
    if monitor:
        env.monitor.close()  
        
    return epis_durations


if __name__ == '__main__':
    # q-net variables
    nr_episodes = 100 # number of episodes to train (start incrementing after experience pool is full)
    gamma = 0.95 # discount variable
    init_epsilon = 1.0 # initial epsilon value 
    final_epsilon = 0.05 # finally anneal epsilon to this (random 5% of the time)
    hidden_layer_dim = 128
    verbose = False # output states during game play
    q_hat_path = 'model/Q_hat/' # file dir.
    pool_size = 1500
    render = False # monitor render
    monitor = False # monitor environment
        
    ### Run experiments ###
    #batchSizes = [16, 32, 64, 128] # increasing from 16 -> 32 -> 64 -> 128 consistently boosts agent's training speed. 
    batchSize = 64
    freeze_everys = [1, 5, 10]
    experiment_results = pd.DataFrame(data = np.zeros((nr_episodes+1, len(freeze_everys))), columns = freeze_everys)

    for freeze_every in freeze_everys:
        experiment_results.loc[:, freeze_every] = run_experiment(nr_episodes, gamma, init_epsilon, batchSize, pool_size,
                                                              hidden_layer_dim, freeze_every, q_hat_path, monitor)

    experiment_results.to_csv('results_freezeintervs.csv')
