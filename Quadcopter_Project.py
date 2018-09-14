
# coding: utf-8

# # Project: Train a Quadcopter How to Fly
# 
# Design an agent to fly a quadcopter, and then train it using a reinforcement learning algorithm of your choice! 
# 
# Try to apply the techniques you have learnt, but also feel free to come up with innovative ideas and test them.

# ## Instructions
# 
# Take a look at the files in the directory to better understand the structure of the project. 
# 
# - `task.py`: Define your task (environment) in this file.
# - `agents/`: Folder containing reinforcement learning agents.
#     - `policy_search.py`: A sample agent has been provided here.
#     - `agent.py`: Develop your agent here.
# - `physics_sim.py`: This file contains the simulator for the quadcopter.  **DO NOT MODIFY THIS FILE**.
# 
# For this project, you will define your own task in `task.py`.  Although we have provided a example task to get you started, you are encouraged to change it.  Later in this notebook, you will learn more about how to amend this file.
# 
# You will also design a reinforcement learning agent in `agent.py` to complete your chosen task.  
# 
# You are welcome to create any additional files to help you to organize your code.  For instance, you may find it useful to define a `model.py` file defining any needed neural network architectures.
# 
# 

# ## Define the Task, Design the Agent, and Train Your Agent!
# 
# Amend `task.py` to specify a task of your choosing.  If you're unsure what kind of task to specify, you may like to teach your quadcopter to takeoff, hover in place, land softly, or reach a target pose.  
# 
# After specifying your task, use the sample agent in `agents/policy_search.py` as a template to define your own agent in `agents/agent.py`.  You can borrow whatever you need from the sample agent, including ideas on how you might modularize your code (using helper methods like `act()`, `learn()`, `reset_episode()`, etc.).
# 
# Note that it is **highly unlikely** that the first agent and task that you specify will learn well.  You will likely have to tweak various hyperparameters and the reward function for your task until you arrive at reasonably good behavior.
# 
# As you develop your agent, it's important to keep an eye on how it's performing. Use the code above as inspiration to build in a mechanism to log/save the total rewards obtained in each episode to file.  If the episode rewards are gradually increasing, this is an indication that your agent is learning.


import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Lambda
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


import ipympl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')

from task import Task


# In[6]:


# Setting inital state of environment
init_pose = np.array([0.,25.,150.,0.,0.,0.])
init_velocities = np.array([0.,0.,0.])
init_angle = np.array([0.,0.,0.])
runtime = 10.
target_pos = np.array([0.,0.,150.])
action_low = 0
action_high = 900
action_range = action_high - action_low


# In[7]:


# Get the environment and extract the number of actions.
task = Task(init_pose, init_velocities, init_angle, runtime, target_pos)
np.random.seed(123)

nb_actions = task.action_size


# In[8]:


# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + task.reset().shape ))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
actor.add(Lambda(lambda x: (x * 450) + 0))
print(actor.summary())


# In[9]:


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,)+task.reset().shape , name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('sigmoid')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


# In[10]:


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])


# In[11]:


agent.load_weights('ddpg_{}_weights.h5f'.format("quad_sim"))


# In[16]:


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

history_1 = agent.fit(task, nb_steps=30000, action_repetition=3, verbose=1, visualize=False)


# In[17]:


# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format("quad_sim"), overwrite=True)


# In[12]:


# Finally, evaluate our algorithm for 5 episodes.
agent.test(task, nb_episodes=5, action_repetition=3, visualize=True)


# In[14]:


print(history.history.keys())


# In[20]:


import pandas as pd

def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.figure(figsize=(10,5))
    plt.plot(scores); plt.title("Scores");
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);
    return rolling_mean

rolling_mean = plot_scores(history.history['episode_reward'] + history_1.history['episode_reward'])


