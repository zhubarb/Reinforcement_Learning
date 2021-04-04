# https://gym.openai.com/envs/#classic_control
# https://gym.openai.com/docs/
# https://github.com/realdiganta/solving_openai/tree/master/FrozenLake8x8
# https://www.oreilly.com/radar/introduction-to-reinforcement-learning-and-openai-gym/
import numpy as np
import gym
import pandas as pd
import seaborn as sns
from itertools import product # cartesian product
from tqdm import tqdm


def choose_action(vtable, trans, env, state, epsilon):
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
        action = np.argmax(vtable[state, :])

    return action


def update_q_table(q, gamma, learning_rate):
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

    :param q: (np.array)
    :return: (dict) updated q (table)
    '''

    prediction = q[state, action]
    target = reward + gamma * np.max(q[next_state, :])
    q[state, action] = prediction + learning_rate * (target - prediction)

    return q

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
    s = np.zeros((env.observation_space.n,env.action_space.n,
                 env.observation_space.n)) # state transition matrix P(s'|s,a)
    epsilon= 1
    min_epsilon=0.01
    max_epsilon=1
    gamma = 0.99 # for this task the closer this is to 1, the more reliable is the avg reward
    learning_rate = 0.8 # if this is >=1, the training overshoots
    decay_rate = 1.5e-3
    n_episodes= 10000
    n_trans_matrix = 1000
    render=False
    rewards = list()
    durations = list()
    epsilons = list()

    for i_episode in tqdm(range(n_episodes)):

        # learn the transition matrix
        while i_episode <= n_trans_matrix:
            state = env.reset()
            action=env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            s[state, action, next_state] += 1
            state = next_state
            if done:
                i_episode +=1
        for t in range(100):
            if render:
                env.render()
            #Pick Action
            action = choose_action(v, s, env, state, epsilon)
            next_state, reward, done, info = env.step(action)
            # Update State Transition Matrix
            s[state,action,next_state]+=1
            # Update State-Action Value Table
            #q = update_q_table(q, gamma, learning_rate)

            state=next_state
            if done:
                if render:
                    print("Episode finished with reward= %f after %i timesteps"%(reward, t+1))
                rewards.append(reward)
                durations.append(t)
                epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate* i_episode)
                epsilons.append(epsilon)
                 #learning_rate = max(0.05,1-(i_episode/n_episodes))#adjust_learning_rate(learning_rate, 10, rewards)
                break

    pd.DataFrame(rewards).rolling(100).mean().plot()
    pd.concat([pd.DataFrame(rewards).rolling(100).mean(),
               pd.DataFrame(durations).rolling(100).mean()], axis=1).to_csv('exp_results.csv')
    pd.DataFrame(epsilons).plot()


def update_q_table(self, obs, action, reward, next_obs, done, func):
    '''


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

        if done:  # terminal state, just immediate reward
            target = reward
        else:  # within episode
            target = reward + self.gamma * func(self.__get_state_vals(next_obs))
        prediction = self.__get_state_vals(obs)[action]
        updated_q_val = prediction + self.learning_rate * (target - prediction)
        # update the q-value for the observed state,action pair
        self.__set_q_val(obs, action, updated_q_val)

    elif self.learning == 'on-policy':  # SARSA

        if done:  # terminal state, just immediate reward
            target = reward
        else:  # within episode
            on_policy_q = self.epsilon * np.nanmean(self.__get_state_vals(next_obs)) + \
                          (1 - self.epsilon) * func(self.__get_state_vals(next_obs))
            target = reward + self.gamma * on_policy_q
        prediction = self.__get_state_vals(obs)[action]
        updated_q_val = prediction + self.learning_rate * (target - prediction)

        # update the q-value for the observed state,action pair
        self.__set_q_val(obs, action, updated_q_val)
    else:
        raise ValueError('Learning method is not known.')