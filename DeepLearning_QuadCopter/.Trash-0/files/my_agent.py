import numpy as np
from task import Task
import random
from collections import namedtuple, deque

from keras import layers, models, optimizers
from keras import backend as K

from agents.actor_critic_model import Actor, Critic

class My_Agent():
    
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

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
        self.critic_target = Critic(self.task, self.lr_critic)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1
        self.score = 0
        
        # For replay memory (see class ReplayBuffer below)
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)


        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        self.last_state = state
        return state
    
    def step(self, action, next_state,reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1
        
        self.memory.add(self.last_state, action, next_state,reward,  done)
        #print(len(self.memory))
#         print("memory={}".format(memory=len(self.memory)))
        
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            # print("hello") # for debug

        # Roll over last state and action
        self.last_state = next_state
        
     
   
    def act(self, state):
        # Choose action based on given state and policy
        #action = np.dot(state, self.w)  # simple linear policy
        state = np.reshape(state, [-1, self.state_size])
        print(self.actor_local.model.predict(state))
        action = self.actor_local.model.predict(state)[0]
#         print("=====statesize==state==action=========")
#         print(self.state_size)
#         print(state)
#         print(action)
#         print("======================")
        return action


    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        #print("experiences={}".format(experiences),  end="")
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
#         print(next_states)
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
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)


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
                                field_names=["state", "action", "next_state", "reward",  "done"])

    def add(self, state, action, next_state,reward,  done):
        """
        Add a new experience to memory.
        """
        e = self.experience(state, action,next_state, reward,  done)
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