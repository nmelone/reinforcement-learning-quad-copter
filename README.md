# reinforcement-learning-quad-copter
A reinforcement learning task to have a simulated quad copter learn to achieve a particular goal

This project was part of the Machine Learning Engineer nano-degree from Udacity. The assignment was to create a task and agent to train a quad copter in simulation. The task was left open ended as well was the model but the simulation environment was provided. I chose to have the quad copter try to learn to fly to a designated location and hover there. This is seen in the task.py file under the get_reward method. I employed a DDPG agent through the Keras-RL infrastructure. DDPG is a model that uses two networks that cooperate to achieve a task. One is the Actor and performs a task while the other is a Critic that judges how well the actor did a task. Both networks start out untrained and learn in tandem.
