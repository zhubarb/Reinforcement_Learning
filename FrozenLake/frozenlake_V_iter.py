# https://gym.openai.com/envs/#classic_control
# https://gym.openai.com/docs/
# https://github.com/realdiganta/solving_openai/tree/master/FrozenLake8x8
# https://www.oreilly.com/radar/introduction-to-reinforcement-learning-and-openai-gym/
import numpy as np
import gym
import pandas as pd
import seaborn as sns
import copy
from tqdm import tqdm


def choose_action(v, s, env, state, epsilon):
    '''

    :param vtable: (np.array)
    :param trans: (np.array)
    :param env: (Open AI Gym environment)
    :param state: (int)
    :param epsilon: (float)
    :return:
    '''
    random_number = np.random.uniform(0,1) # generate random number

    if random_number<=epsilon: # explore
        action = env.action_space.sample()

    else: # choose action index with the max q value
        next_state_values= s[state]@v # P(s'|s,a) * V(s')
        action = np.argmax(next_state_values) # argmax_a(V(s'))

    return action


def update_v_table_v1(v, s, exp, gamma, learning_rate):

    state, action, next_state, reward, done = exp

    prediction = v[state]
    target = s[state,action,next_state] * (reward + gamma * v[next_state]) # P(s'|s,a)*(reward + disc_factor* V(s')
    v[state] = prediction + learning_rate * (target - prediction)
    if done: # try this
        v[next_state]=reward

    return v


def update_v_table_v2(v, s, exp, gamma, learning_rate):

    state, action, next_state, reward, done = exp

    v_temp = copy.deepcopy(v)
    v_temp[next_state] = reward + gamma * v[next_state]

    prediction = v[state]
    target = np.max(s[state]@v_temp)
    v[state] = prediction + learning_rate * (target - prediction)

    return v


def update_v_table(v, s, exp, gamma, learning_rate):

    state, action, next_state, reward, done = exp

    prediction = v[state]

    target = np.max(reward + gamma*(s[state]@v))

    v[state] = prediction + learning_rate * (target - prediction)

    if done:
        v[next_state]=reward

    return v


def get_actual_trans_matrix(env):
    s = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    for state in range(s.shape[0]):  # loop through state: s
        for action in range(s.shape[1]):  # loop through actions: a
            transitions_to_next_state = [(t[1], t[0]) for t in env.env.P[state][action]]
            for t in transitions_to_next_state:
                s[state, action, t[0]] += t[1]
    return s


def learn_trans_matrix(env, n_trans_matrix):

        # initialise state, action, next_state transition matrix: P(s'|s,a)
        s = np.zeros((env.observation_space.n, env.action_space.n,  env.observation_space.n))

        # play for n_trans_matrix_episodes to observe P(s'|s,a) frequencies
        i_episode = 0
        state = env.reset()
        while i_episode <= n_trans_matrix:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            s[state, action, next_state] += 1
            state = next_state
            if done:
                state = env.reset()
                i_episode += 1
        env.close()

        # Normalise P(s'|s,a) frequenciesto get probabilities
        for state in range(s.shape[0]): # loop through state: s
            for action in range(s.shape[1]): # loop through actions: a
                s[state, action, :] = s[state, action, :] / s[state, action, :].sum()

        print('Transition matrix s learned.')
        return s


def visualise_q_table(q_table):
    '''

    :param q_table:(dict)
    :return:
    '''
    qtable_df= pd.DataFrame(q_table)
    sns.heatmap(qtable_df.astype(float), annot=True)



if __name__ == '__main__':

    env = gym.make('FrozenLake-v0')  # https://gym.openai.com/envs/FrozenLake-v0/
    print(env.action_space)  # print action space for cartpole
    print(env.observation_space)  # print state space for cartpole

    v= np.zeros(env.observation_space.n) # state value table
    #v[:5]=np.array([1,2,3,4,5])

    epsilon= 1
    min_epsilon=0.01
    max_epsilon=1
    gamma = 0.95 # for this task the closer this is to 1, the more reliable is the avg reward
    learning_rate = 0.8 # if this is >=1, the training overshoots
    decay_rate = 1e-2
    n_episodes= 3000
    n_trans_matrix = 10000
    render=False
    rewards = list()
    durations = list()
    epsilons = list()

    # learn the transition matrix
    #s = learn_trans_matrix(env, n_trans_matrix)
    s = get_actual_trans_matrix(env)

    for i_episode in tqdm(range(n_episodes)):
        state = env.reset()
        for t in range(100):
            if render:
                env.render()
            #Pick Action
            action = choose_action(v, s, env, state, epsilon)
            next_state, reward, done, info = env.step(action)

            # Update State Value Table
            exp = (state, action, next_state, reward, done)
            v = update_v_table(v, s, exp, gamma, learning_rate)

            state=next_state
            if done:
                if render:
                    print("Episode finished with reward= %f after %i timesteps"%(reward, t+1))
                rewards.append(reward)
                durations.append(t)
                epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate* i_episode)
                epsilons.append(epsilon)
                policy = [0]*env.observation_space.n
                for state in range(len(policy)):
                    policy[state] = np.argmax(s[state]@v)
                 #learning_rate = max(0.05,1-(i_episode/n_episodes))#adjust_learning_rate(learning_rate, 10, rewards)
                break

    pd.DataFrame(rewards).rolling(100).mean().plot()
    pd.concat([pd.DataFrame(rewards).rolling(100).mean(),
               pd.DataFrame(durations).rolling(100).mean()], axis=1).to_csv('exp_results.csv')
    pd.DataFrame(epsilons).plot()

