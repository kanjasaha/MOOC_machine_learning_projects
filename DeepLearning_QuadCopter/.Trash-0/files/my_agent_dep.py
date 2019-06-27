import numpy as np
from task import Task
import random
from collections import namedtuple, deque

from keras import layers, models, optimizers
from keras import backend as K

from agents.actor_critic_model import Actor, Critic

class My_Agent():
    """
    Reinforcement Learning agent that learns using DDPG.
    (The actor/critic model is defined in model.py and are imported.)
    """
    def __init__(self, task):
        """
        Initialize parameters and import the actor and critic models.
        """
        # type of task (takeoff, hover or landing etc.)
        self.task = task
        # size of state/action space
        self.state_size = task.state_size # size of state space
        self.action_size = task.action_size # size of action space
        # # range of action space
        # self.action_low = task.action_low # min in the action space
        # self.action_high = task.action_high # max in the action space

        # Algorithm parameters for DDPG
        self.gamma = 0.9  # discount factor
        self.tau = 0.03  # for soft update of target parameters
        self.lr_actor = 0.008 # learning rate for actor model
        self.lr_critic = 0.008 # learning rate for actor model

        # actor (policy) model
        self.actor_local = Actor(self.task, self.lr_actor)
        self.actor_target = Actor(self.task, self.lr_actor)

        # critic (value) model
        self.critic_local = Critic(self.task, self.lr_critic)
        self.critic_target = Critic(self.task, self.lr_critic)S

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # For Ornstein-Uhlenbeck Noise process (see class OUNoise below)
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu,
                                self.exploration_theta, self.exploration_sigma)

        # For replay memory (see class ReplayBuffer below)
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        self.reset_episode()

    def reset_episode(self):
        """
        Reset parameters when a new epsode starts
        """
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.total_reward = 0.0 # total reward

        return state

    def step(self, action, reward, next_state, done):
        """
         Save experience and reward to the memory,
         and learn from the stored experiences.
        """
        # update total reward
        self.total_reward += reward
        # Save experience/reward to memory
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Once enougch samples are stored in the memory, learn to improve the model
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            # print("hello") # for debug

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """
        For a given state, return an action chosen according the current policy.
        Some noise is added in the end for exploration.
        """
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]

        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """
        Update policy and value parameters using given batch of experience tuples.
        """
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """
        Soft update model parameters.
        """
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = []
        for local_weights, target_weights in zip(local_weights, target_weights):
            new_weights.append(self.tau * local_weights + (1 - self.tau) * target_weights)
            target_model.set_weights(new_weights)


class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    """

    def __init__(self, size, mu, theta, sigma):
        """
        Initialize parameters and noise process.
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """
        Reset the internal state (= noise) to mean (mu).
        """
        self.state = self.mu

    def sample(self):
        """
        Update internal state and return it as a noise sample.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, buffer_size, batch_size):
        """
        Initialize a ReplayBuffer object.
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (maximum size is buffer_size)
        self.batch_size = batch_size # size of each training batch
        self.experience = namedtuple("Experience",
                                field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """
        Randomly sample a batch of experiences from memory.
        """
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)
