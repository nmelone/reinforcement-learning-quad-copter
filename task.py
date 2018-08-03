import numpy as np
from physics_sim import PhysicsSim
import ipympl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class Task():
	"""Task (environment) that defines the goal and provides feedback to the agent."""
	def __init__(self, init_pose=None, init_velocities=None, 
		init_angle_velocities=None, runtime=5., target_pos=None):
		"""Initialize a Task object.
		Params
		======
			init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
			init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
			init_angle_velocities: initial radians/second for each of the three Euler angles
			runtime: time limit for each episode
			target_pos: target/goal (x,y,z) position for the agent
		"""
		# Simulation
		self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
		self.action_repeat = 3

		self.state_size = self.action_repeat * 6
		self.action_low = 0
		self.action_high = 900 
		self.action_size = 4

		# Goal
		self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
		
		self.point = {'x':[],'y':[],'z':[]}
		
		self.show_graph=True
		self.do_render=False
		
	def get_reward(self):
		r_min = (((np.array([-150.,-150.,0.]) - self.target_pos)**2).sum())**0.5
		r_max = 0.
		t_min = -1.
		t_max = 1.
		
		# if(np.any(self.sim.pose[:3] <= self.sim.lower_bounds) or np.any(self.sim.pose[:3] >= self.sim.upper_bounds)):
			# reward = -3. 
		# else:
		"""Uses current pose of sim to return reward."""
		reward_raw = (((self.sim.pose[:3] - self.target_pos)**2).sum())**0.5
		reward = (reward_raw-r_min)/(r_max-r_min) * (t_max-t_min) + t_min
		
		
		return reward

	def step(self, rotor_speeds):
		"""Uses action to obtain next state, reward, done."""
		reward = 0
		pose_all = []
		
		done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
		reward += self.get_reward() 
		pose_all.append(self.sim.pose)
		next_state = np.concatenate(pose_all)
		info = dict()
		if(self.do_render):
			self.point['x'].append(self.sim.pose[0])
			self.point['y'].append(self.sim.pose[1])
			self.point['z'].append(self.sim.pose[2])
			self.render(done=done)
		return next_state, reward, done, info

	def reset(self):
		"""Reset the sim to start a new episode."""
		self.sim.reset()
		state = np.concatenate([self.sim.pose] )
		if(self.do_render):
			self.ax.scatter(self.sim.init_pose[0],self.sim.init_pose[1],self.sim.init_pose[2])
		
		return state
	
	def render(self, mode='init',done=False):
		if(mode == 'human'):
			self.do_render = True
			if(self.show_graph):
				self.init_graph()
				self.show_graph=False
		if(done):
			self.line.plot(self.point['x'],self.point['y'], self.point['z'])
			self.point['x'][:] = []
			self.point['y'][:] = []
			self.point['z'][:] = []
	
	def init_graph(self):
		self.fig = plt.figure(figsize=(8,8))
		self.line = self.fig.add_subplot(111, projection='3d')
		self.ax = plt.gca()
		
		self.line.set_xlim(-150,150)
		self.line.set_ylim(-150,150)
		self.line.set_zlim(0,300)
		
		self.ax.scatter(self.target_pos[0], self.target_pos[1], self.target_pos[2], color='green', label='Goal')
		