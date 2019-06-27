import numpy as np
from agents.agent import Agent

class Agent_TakeOff(Agent):
    """Sample agent that searches for optimal policy randomly."""

    def __init__(self, task):
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space
        self.state_size = task.state_size
        self.state_range = task.state_range
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
#         print('Agent takeoff init self.state_size')
#         print(self.state_size)
#         print('Agent takeoff init self.state_range')
#         print(self.state_range)

        # Policy parameters
        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size)))  # start producing actions in a decent range
#         print('self.w')
#         print(self.state_size)
#         print(self.action_size)
#         print(self.action_range)
       
        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode_vars()

    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0

    def step(self, state, reward, done):
        # Transform state vector
        state = (state - self.task.action_high) / self.state_range  # scale to [0.0, 1.0]
        state = state.reshape(1, -1)  # convert to row vector
#         print(' agent step state')
#         print(state)
#         # Choose an action
#         print('calling from agent step action')
        
        action = self.act(state)
        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.total_reward += reward
            self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()
            self.reset_episode_vars()

        self.last_state = state
        self.last_action = action
        return action

    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
#         print(' agent act state self.w')
#         print(state)
#         print(self.w)
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)

        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
#         print("Agent_TakeOff.learn(): t = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
#                 self.count, self.score, self.best_score, self.noise_scale))  # [debug]
#         #print(self.w)  # [debug: policy parameters]