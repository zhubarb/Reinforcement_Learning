# https://gym.openai.com/envs/#classic_control
import numpy as np
import gym
env = gym.make('CartPole-v0')
print(env.action_space) # print action space for cartpole
print(env.observation_space) # print state space for cartpole
print(env.observation_space.high) #  check the state space's bounds
print(env.observation_space.low) #  check the state space's bounds

for i_episode in range(20):
    state = env.reset()
    print("Episode starting after environment reset")
    rewards=list()
    for t in range(100):
        env.render()
        print('Cur state: ' + str(state) )
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            print("Episode finished after %i timesteps"%(t+1))
            print("Total reward is %f"%(np.array(rewards).sum()))
            break