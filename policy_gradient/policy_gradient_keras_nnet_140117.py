# policy_gradient_keras_nnet.py
""" 
26 December 2016:
Trains an agent with (stochastic) Policy Gradients on CartPole. Uses OpenAI Gym. 
"""

import numpy as np
import gym
from matplotlib import pyplot as plt
from RL_Agent_140117 import PG_Learner, PG_Learner_softmax


def plot_durations(ts):
    '''
    Parameters:
    ------------
    ts - (list) contains episode durations
    '''
    plt.figure()
    plt.title("100-running avg episode durations")
    plt.plot(np.array(range(len(ts))), ts)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    
    # hyper-parameters
    batch_size = 5 # every how many episodes to do a param update?
    gamma = 0.95 # discount factor for reward
    learning_rate = 1e-3 # learning rate
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
    resume = False # resume from previous checkpoint?
    render = False
    verbose = False
    standardize_reward = True
    running_rewards = list()

    # Make environment
    env = gym.make('CartPole-v0')
    observation = env.reset()
    
    # Initialise agent
    pgl = PG_Learner_softmax(env, learning_rate, decay_rate, gamma)

    ts = [] # episode durations container (for monitoring)
    batch_labels, batch_obs= [], []

    # start training routine
    episode_number = 0 # episode number
    t = 0  # episode number
    while episode_number <= 10e3:
        
        if render: env.render()
       
        # forward the policy network and sample an action from the returned probability
        x = observation.reshape(1,pgl.state_space_dim) 
        action = pgl.pick_action(x)

        # step the environment and get new observation
        observation, reward, done, info = env.step(action)
        # let agent keep track of the prev obs, x, and reward 
        pgl.record_episode_obs_n_rewards(x, reward)
  
        if done:
            episode_number += 1 # increment episode number
            ts.append(t) # append current episode duration to t's
            t = 0 # reset tick number
            
            # append episode obs and labels to the learner's batch containers
            pgl.append_to_minibatch()

            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                pgl.update_func_approximator()

            # book-keeping
            running_reward = np.mean(ts[-100:len(ts)]) # 100-episode running avg reward
            running_rewards.append(running_reward)

            if episode_number % 100 == 0: 
                print ' Episode %i (%i), 100-episode mean: %f' % (episode_number, ts[-1], 
                                                              np.round(running_reward,2))
            observation = env.reset() # reset env
            
        # increment t by 1
        t += 1

    # plot durations
    plot_durations(running_rewards)

    # save weights
    pgl.save_weights(type(env).__name__)
