from keras import layers, models, optimizers
from keras import backend as K

class Actor:
    """
    This class defines the actor (policy) model for DDPG.
    """

    def __init__(self, task, lr_actor):
        """
        Initialize parameters and build model.
        """
        # type of task (takeoff, hover or landing etc.)
        self.task = task
        # size of state/action space
        self.state_size = task.state_size # size of state space
        self.action_size = task.action_size # size of action space
        # range of the action space
        self.action_low = self.task.action_low # min in the action space
        self.action_high = self.task.action_high # max in the action space
        self.action_range = self.action_high - self.action_low # range of action space

        self.lr_actor = lr_actor # learning rate for actor model
        # build a model
        self.build_model()

    def build_model(self):
        """
        Define a neural network for an actor (policy) model,
        i.e. the input is states and actions are returned.
        """
        # input layer (input = states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # hidden layers
        net = layers.Dense(units=32, activation='relu')(states)
        net = layers.BatchNormalization()(net)
        net = layers.Dropout(0.5)(net)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.BatchNormalization()(net)
        net = layers.Dropout(0.5)(net)
        net = layers.Dense(units=32, activation='relu')(net)

        # output layer with sigmoid activation function (to be normalized below)
        raw_actions = layers.Dense(units=self.action_size,
                                        activation='sigmoid',name='raw_actions')(net)

        # Rescaling of the output (s.t. the output take the value in the action space)
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                                        name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action-value (Q-value) gradients
        # Note: action_gradients is computed in the class Critic
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.lr_actor)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
                inputs=[self.model.input, action_gradients, K.learning_phase()],
                outputs=[],
                updates=updates_op)

class Critic:
    """
    This class defines the critic (value) model for DDPG.
    """

    def __init__(self, task, lr_critic):
        """
        Initialize parameters and build model.
        """
        # type of task (takeoff, hover or landing etc.)
        self.task = task
        # size of state/action space
        self.state_size = task.state_size # size of state space
        self.action_size = task.action_size # size of action space

        self.lr_critic = lr_critic # learning rate for crtic model

        # build a model
        self.build_model()

    def build_model(self):
        """
        Define a neural network for a critic (value) model,
        i.e. the input is states and actions, and Q-value is returned
        (its gradient is also computed).
        """
        # Input layer (states and actions)
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Hidden layers for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.5)(net_states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.5)(net_states)
        net_states = layers.Dense(units=32, activation='relu')(net_states)

        # Hidden layers for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions= layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.5)(net_actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)
        net_actions= layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.5)(net_actions)
        net_actions = layers.Dense(units=32, activation='relu')(net_actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])

        # # Hidden layers after combining the state and action pathways
        net = layers.Dropout(0.5)(net)
        net = layers.Dense(units=32, activation='relu')(net)

        # Output layer (output = Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.lr_critic)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        # Note: this action_gradients is used in the actor model
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
                        inputs=[*self.model.input, K.learning_phase()],
                        outputs=action_gradients)