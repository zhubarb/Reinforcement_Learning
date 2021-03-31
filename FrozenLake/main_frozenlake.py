# https://gym.openai.com/envs/#classic_control
# https://gym.openai.com/docs/
# https://github.com/realdiganta/solving_openai/tree/master/FrozenLake8x8
# https://www.oreilly.com/radar/introduction-to-reinforcement-learning-and-openai-gym/
import numpy as np
import gym
import pandas as pd
import seaborn as sns
env = gym.make('FrozenLake-v0') # https://gym.openai.com/envs/FrozenLake-v0/
print(env.action_space) # print action space for cartpole
print(env.observation_space) # print state space for cartpole
from itertools import product # cartesian product
from tqdm import tqdm

def create_q_table(n_obs_space, n_action_space):#
    '''
    Populate q_table that contains (state, action): q_value where initial q_values are sampled
    from a uniform distribution.
    :return:
    '''

    # cartesian product of the discrete observation space and action space
    q_table_keys = [element for element in product(*[list(range(n_obs_space)),
                                                list(range(n_action_space))])]
    #q_table = dict(zip(q_table_keys,np.random.uniform(0, 1, len(q_table_keys))))
    q_table = dict(zip(q_table_keys,[0.5]*len(q_table_keys)))

    return q_table


def choose_action(q_table, env, state, epsilon):
    '''

    :param q_table: (dict)
    :param env: (Open AI Gym environment)
    :param state: (int)
    :param epsilon: (float)
    :return:
    '''
    random_number = np.random.rand() # generate random number

    if random_number<=epsilon: # explore
        action = env.action_space.sample()

    else: # choose action idnex with the max q value
        q_table_keys = [(state,action) for action in range(env.action_space.n)] # q vals for all actions on that state
        action = np.array([q_table[k] for k in q_table_keys]).argmax() # action that has the highest q val

    return action


def update_q_table_doesnt_work(q_table, experiences):
    '''
    Operates on end of episode reward rather than immediate experience reward. (So probably Monte Carlo method?)
    :param q_table: (dict)
    :param experiences: (list) of experience tuples (state, action, reward)
    :return: (dict) updated q (table)
    '''
    # update the q table with experiences from this episode
    episode_reward = experiences[-1][-1]# the reward from the final experience for that episode

    for exp in experiences:
        prev_qval = q_table[(exp[0],exp[1])]
        q_table[(exp[0],exp[1])] = (prev_qval + episode_reward)/2

    return q_table


def update_q_table(q_table, experiences, gamma, learning_rate):
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

    :param q_table: (dict)
    :param experiences: (list) of experience tuples (state, action, reward)
    :return: (dict) updated q (table)
    '''

    for exp in experiences:
        state, action, next_state, reward, done = exp
        if done:  # terminal state, just immediate reward
            target = reward
        else:  # within episode
            next_state_keys = [(state,action) for action in range(env.action_space.n)]
            next_state_qvals = [q_table[k] for k in next_state_keys]
            target = reward + gamma * np.max(next_state_qvals)
        prediction = q_table[state,action] # existing q value
        updated_q_val = prediction + learning_rate * (target - prediction)
        # update the q value
        q_table[state, action] = updated_q_val

    return q_table


def visualise_q_table(q_table):
    '''

    :param q_table:(dict)
    :return:
    '''
    states = list(set([k[0] for k in q_table.keys()]))
    actions = list(set([k[1] for k in q_table.keys()]))
    qtable_df= pd.DataFrame(index=states, columns=actions)
    for key,value in q_table.items():
        state= key[0]
        action = key[1]
        qtable_df.loc[state,action]=value
    sns.heatmap(qtable_df.astype(float), annot=True)


def adjust_learning_rate(lr, gradient_N, rewards):
    '''
    Based on the gradient of the avg reward slope for the past gradient_N episodes
    :param lr:
    :param rewards:
    :return:
    '''

    if len(rewards)<= gradient_N:
        return lr
    else:
        if len(rewards)%gradient_N == 0: # look at the gradient
            mavg_rewards = pd.Series(rewards).rolling(gradient_N).mean() # calc moving averages
            improvement_rate= mavg_rewards.iloc[-gradient_N:-1].diff(1).mean() # avg diff of moving averages is the last N obs
            lr *= (1+improvement_rate)

        return lr


if __name__ == '__main__':

    q= create_q_table(env.observation_space.n, env.action_space.n)
    epsilon= 1
    gamma = 0.9
    learning_rate = 1
    n_episodes= 10000
    render=False
    rewards = list()
    durations = list()

    for i_episode in tqdm(range(n_episodes)):
        state = env.reset()
        next_state = None
        episode_experiences=list()
        for t in range(100):
            state =  next_state if next_state is not None else state
            if render:
                env.render()
            #print('Cur state: ' + str(state) )
            action = choose_action(q, env, state, epsilon)
            #print('Cur action: ' + str(action))
            next_state, reward, done, info = env.step(action)
            episode_experiences.append((state,action,next_state, reward, done))
            if done:
                if render:
                    print("Episode finished with reward= %f after %i timesteps"%(reward, t+1))
                rewards.append(reward)
                durations.append(t)
                # Vanilla Q learning, avg. 50 episode reward oscillates a lot
                q = update_q_table(q, episode_experiences, gamma, learning_rate)
                epsilon = max(0.1,1-(i_episode/n_episodes)) # anneal epsilon with a floor of 5% random actions
                #learning_rate = max(0.05,1-(i_episode/n_episodes))#adjust_learning_rate(learning_rate, 10, rewards)
                break


    visualise_q_table(q)

    pd.concat([pd.DataFrame(rewards).rolling(100).mean(),
               pd.DataFrame(durations).rolling(100).mean()], axis=1).to_csv('exp_results.csv')



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