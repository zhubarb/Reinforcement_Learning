""" 
20 December 2016:
Trains an agent with (stochastic) Policy Gradients on CartPole. Uses OpenAI Gym. 
The neural net implementation is manual and in numpy
"""
from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.optimizers import  RMSprop
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
import gym


def discount_rewards_cartpole(r):
    """ make last reward -1, roll it back 50 ticks. For the rest of the rewards, 
    leave as is """
    #r[-1] = -1 # appoint -1 as the last reward (failed)
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

def build_model(input_dim, output_dim, w_reg):
    '''
    Build keras model given the number of input (predictive) and output
    features, along wiht the weight decay (shrinkage) mode.
    Assumes classification (so output activation is either sigmoid or softmax)
    initialisations: https://keras.io/initializations/
    activation funcs: https://keras.io/activations/
    objective funcs: https://keras.io/objectives/
    Parameters:
    -------------
    input_dim         - (int) the number of input features 
    output_dim        - (int) the output dimensionality. For linear or binary 
                        log reg, 1 for multi-class classification, greater than 1
    w_reg             - (str) weight decay (shrinkage) type: 'l1' or 'l2'
    output_activation - (str) e.g. 'sigmoid', 'softmax', etc.
    '''
    
    model = Sequential()

    # 1st hidden layer
    model.add(Dense(100, input_shape=(input_dim,),activation='relu', 
                    init='glorot_normal', W_regularizer=w_reg))
    # 2nd hidden layer                    
    #model.add(Dense(32, activation='relu',init='glorot_normal',
    #                W_regularizer=w_reg))
    # Output layer
    output_dim = 1
    model.add(Dense(output_dim, activation='sigmoid', 
                    init='glorot_normal', W_regularizer=w_reg))    

    # Compile the architecture
    learning_rate = 1e-4
    model.compile(loss='mse',
                  optimizer= RMSprop(lr=learning_rate) ,
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    
    # hyper-parameters
    batch_size = 10 # every how many episodes to do a param update?
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
    resume = False # resume from previous checkpoint?
    render = False
    verbose = False
    standardize_reward = True
    
    # Make environment
    env = gym.make('CartPole-v0')
    observation = env.reset()
    
    # Initialise or load / resume neural net
    w_reg = None
    state_size = env.observation_space.shape[0] if len(env.observation_space.shape)==1 \
                                                else env.observation_space.shape
    action_size = env.action_space.n
    if resume:
        empty_model = Sequential()
        model = empty_model.load_weights('cartpole_keras.h5')
    else:
        model =  build_model(state_size, action_size, w_reg)

    # initialise containers
    batch_labels, batch_obs = [], [] # initialise batch containers
    xs, aprobs, drs = [], [], [] # initialise episodic containers
    ts = [] # episode durations container (for monitoring)
    episode_number = 0 # episode number
    t = 0  # tick number
    
    # start training routine
    while True:
        
        if render: env.render()
        x = observation.reshape(1,state_size) # reshape for input to the Keras model
        
        # forward the policy network and sample an action from the returned probability
        aprob = model.predict(x)
        action = 1 if np.random.uniform() < aprob else 0 # roll the dice!
        # grad that encourages the action that was taken to be taken 
        # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        # grad = aprob - action  
        
        aprobs.append(action)
        xs.append(x) # observation        

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        drs.append(reward) # record reward         
        
        if done:
            episode_number += 1 # increment episode number
            ts.append(t) # append current episode duration to t's
            t = 0 # reset tick number
            
            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards_cartpole(np.vstack(drs))
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            if standardize_reward:
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)     
                           
            # modulate the gradient with advantage (PG magic happens right here.) 
            batch_labels.append(np.vstack(aprobs) * discounted_epr)       
            batch_obs.append(np.vstack(xs))     
            # reset episode containers: aprobs, xs, drs 
            aprobs, xs, drs = [], [], []
            
            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                X_train = np.concatenate(batch_obs)
                y_train =  -1* np.concatenate(batch_labels)
                #model.fit(X_train, y_train, nb_epoch=1)
                model.train_on_batch(X_train, y_train)
                # reset the batch containers
                batch_labels, batch_obs = [], []
            
            # boring book-keeping
            running_reward = np.mean(ts[-100:len(ts)]) # 100-episode running avg reward
            print ' Episode %i (%i), 100-episode mean: %f' % (episode_number, ts[-1], 
                                                              np.round(running_reward,2))
            if episode_number % 100 == 0: 
                pass#model.save_weights('cartpole_keras.h5') 
            observation = env.reset() # reset env
            
        # increment t by 1
        t += 1
