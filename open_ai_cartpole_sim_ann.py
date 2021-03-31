# -*- coding: utf-8 -*-
"""
Created on Sat May  7 19:54:23 2016

@author: oswald
"""

import gym
import random
env = gym.make('CartPole-v0')

#env.monitor.start('/tmp/cartpole-experiment-3', force=True)

# its like simulated annealing 

bestSteps = 0
best = [0, 0, 0, 0]
alpha = 1

for i_episode in range(80):
    
    test = [best[i] + (random.random() - 0.5)*alpha for i in range(4)]

    score = 0
    for ep in range(10):  # <-- key thing was to figure out that you need to do 10 tests per point
        state = env.reset()
        for t in range(200): # <-- because you can't go over 200 you need to gain score hight else where
            env.render()
            if sum(state[i]*test[i] for i in range(4)) > 0:
                action = 1
            else:
                action = 0
            state, reward, done, info = env.step(action)
            if done:
                break

        score += t

    if bestSteps < score:
        bestSteps = score
        best = test
        alpha *= .9

    print("test: " +str(test))
    print(str(score))
    print("best: " + str(best))
    print(str(bestSteps) +' ' + str(alpha))

print("best", best, bestSteps)

env.monitor.close()