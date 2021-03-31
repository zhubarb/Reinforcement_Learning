""" Trains an agent with (stochastic) Policy Gradients on CartPole. Uses OpenAI Gym. 
    The neural net implementation is manual and in numpy"""
import numpy as np
import cPickle as pickle
import gym

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
    """ preprocess ping-pong image: 210x160x3 uint8 frame into 6400 
    (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def discount_rewards_cartpole(r):
    """ make last reward -1, roll it back 50 ticks. For the rest of the rewards, 
    leave as is """
    #r[-1] = -1 # appoint -1 as the last reward (failed)
    discounted_r = np.zeros_like(r)
    running_add = 0
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

if __name__ == '__main__':

    # hyperparameters
    batch_size = 1 # every how many episodes to do a param update?
    learning_rate = 1e-4
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
    resume = False # load save.p model and resume from previous checkpoint?
    render = False
    verbose = False
    standadrize_reward = True


    # Make environment
    env = gym.make('CartPole-v0') # env = gym.make("Pong-v0")
    observation = env.reset()

    # Neural Net initialization
    D = len(observation) if len(observation.shape)==1 else observation.shape

    if resume:
        model = pickle.load(open('save.p', 'rb'))
    else:
        H = 100 # number of hidden layer neurons
        model = {}
        model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
        model['W2'] = np.random.randn(H) / np.sqrt(H)
  
    # update buffers that add up gradients over a batch
    grad_buffer   = { k : np.zeros_like(v) for k,v in model.iteritems() } 
    # rmsprop memory
    rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() }

    # initialise containers needed for backprop
    xs,hs,dlogps,drs, ts = [],[],[],[],[]
    episode_number = 0
    t = 0  # tick number

    while True:

        if render: env.render()
        x = observation
        
        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)
        action = 1 if np.random.uniform() < aprob else 0 # roll the dice!
        
        # record various intermediates (needed later for backprop)
        xs.append(x) # observation
        hs.append(h) # hidden state
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
            discounted_epr = discount_rewards_cartpole(epr)
            
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            if standadrize_reward:
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
            
            epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
            grad = policy_backward(eph, epdlogp)
            for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch
            
            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                for k,v in model.iteritems():
                    g = grad_buffer[k] # gradient
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5) # update weights
                    grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
            
            # boring book-keeping
            running_reward = np.mean(ts[-100:len(ts)]) # 100-episode running avg reward
            print ' Episode %i (%i), 100-episode mean: %f' % (episode_number, ts[-1], 
                                                              np.round(running_reward,2))
            if episode_number % 1000 == 0: pickle.dump(model, open('save.p', 'wb'))
            observation = env.reset() # reset env
            
        # increment t by 1
        t += 1
