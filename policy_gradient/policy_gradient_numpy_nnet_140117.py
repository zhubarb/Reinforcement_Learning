# policy_gradient_numpy_nnet.py
""" Trains an agent with (stochastic) Policy Gradients on CartPole. Uses OpenAI Gym. 
    The neural net implementation is manual and in numpy"""
import numpy as np
import cPickle as pickle
from matplotlib import pyplot as plt
import gym

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def discount_rewards_pong(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def discount_rewards_cartpole(r, gamma):
  """ make last reward -1, roll it back 50 ticks. For the rest of the rewards, 
  leave as is """
  discounted_r = np.zeros_like(r)
  N = 10 # blame the pole falling on the past N actions
  # roll back final -1 reward back for N ticks
  rollback_lim = max(0, r.size-N )
  running_add= -1
  for t in reversed(xrange(0, r.size)):
    if t >= rollback_lim:
        running_add = running_add * gamma 
        discounted_r[t] = running_add
    else: 
        discounted_r[t] = r[t]
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

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

    # hyperparameters
    H = 100 # number of hidden layer neurons
    batch_size = 5 # every how many episodes to do a param update?
    learning_rate = 1e-4 # rmsprop learning rate
    epsilon = 1e-5 # rmsprop epsilon
    gamma = 0.95 # discount factor for reward
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
    resume = False # resume from previous checkpoint?
    render = False
    verbose = False
    standardize_reward = True

    # Make environment
    env = gym.make('CartPole-v0') # env = gym.make("Pong-v0")
    observation = env.reset()

    # Neural Net initialization
    D = len(observation) if len(observation.shape)==1 else observation.shape

    if resume:
      model = pickle.load(open('save.p', 'rb'))
    else:
      model = {}
      model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
      model['W2'] = np.random.randn(H) / np.sqrt(H)
  
    grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
    rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

    # initialise containers needed for backprop
    xs,hs,dlogps,drs, ts = [],[],[],[],[]

    episode_number = 0
    t = 0  # tick number
    running_rewards = list()
    while episode_number < 1e4:

      if render: env.render()
      x = observation

      # forward the policy network and sample an action from the returned probability
      aprob, h = policy_forward(x) # aprob = probability of picking action 1
      action = 1 if np.random.uniform() < aprob else 0 # roll the dice!

      # record various intermediates (needed later for backprop)
      xs.append(x) # observation
      hs.append(h) # hidden state, needed for backprop
      y = action # no need for fake label as in: y = 1 if action == 2 else 0 # a "fake label"
      dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

      # step the environment and get new measurements
      observation, reward, done, info = env.step(action)

      drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

      if done: # an episode finished
        episode_number += 1 # increment episode number
        ts.append(t) # append current episode duration to t's
        t = 0 # reset tick number

        if verbose:
            print (info)

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards_cartpole(epr, gamma)
    
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        if standardize_reward:
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp) # keep track of weight grads for rmsprop
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
          for k,v in model.iteritems():
            g = grad_buffer[k] # gradient
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
            model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + epsilon) # update weights
            grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # boring book-keeping
        running_reward = np.mean(ts[-100:-1]) # 100-episode running avg reward
        running_rewards.append(running_reward)
        if episode_number % 100 == 0:
            pickle.dump(model, open('save.p', 'wb'))
            print ' Episode %i, 100-episode running mean: %f' % (episode_number, 
                                                                 running_reward)

        observation = env.reset() # reset env
  
      # increment t by 1
      t += 1
    
    # plot durations
    plot_durations(running_rewards)
  
