"""
DQN Agent - Value-Based Approach
Agent predicts VALUE of states, not Q-values of actions
Simpler and more effective for Tetris
"""
from keras.models import Sequential, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random


class DQNAgent:
    """
    Deep Q-Network Agent with Value-Based Learning
    
    Instead of learning Q(state, action), learns V(state) directly
    This works because we can enumerate all possible next states
    """
    
    def __init__(self, state_size, mem_size=10000, discount=0.95,
                 epsilon=1.0, epsilon_min=0.0, epsilon_stop_episode=500,
                 n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
                 loss='mse', optimizer='adam', replay_start_size=None):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Size of state vector
            mem_size: Size of replay memory
            discount: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_stop_episode: Episode when exploration stops decreasing
            n_neurons: List of neurons per hidden layer
            activations: List of activation functions (layers + output)
            loss: Loss function
            optimizer: Optimizer
            replay_start_size: Min memory size before training starts
        """
        if len(activations) != len(n_neurons) + 1:
            raise ValueError(f"activations must be {len(n_neurons) + 1} long")
        
        self.state_size = state_size
        self.mem_size = mem_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        
        # Exploration settings
        if epsilon_stop_episode > 0:
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = (epsilon - epsilon_min) / epsilon_stop_episode
        else:
            self.epsilon = 0
            self.epsilon_decay = 0
        
        # Network settings
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer
        
        # Replay settings
        self.replay_start_size = replay_start_size if replay_start_size else mem_size // 2
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build neural network"""
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.n_neurons[0], input_dim=self.state_size, 
                       activation=self.activations[0]))
        
        # Hidden layers
        for i in range(1, len(self.n_neurons)):
            model.add(Dense(self.n_neurons[i], activation=self.activations[i]))
        
        # Output layer (single value - state value)
        model.add(Dense(1, activation=self.activations[-1]))
        
        model.compile(loss=self.loss, optimizer=self.optimizer)
        
        return model
    
    def add_to_memory(self, current_state, next_state, reward, done):
        """Add experience to replay memory"""
        self.memory.append((current_state, next_state, reward, done))
    
    def random_value(self):
        """Return random value for exploration"""
        return random.random()
    
    def predict_value(self, state):
        """Predict value of state"""
        return self.model.predict(state, verbose=0)[0]
    
    def best_state(self, states):
        """
        Find best state from given states
        
        Args:
            states: Iterable of state vectors
            
        Returns:
            Best state (highest predicted value)
        """
        # Exploration: random state
        if random.random() <= self.epsilon:
            return random.choice(list(states))
        
        # Exploitation: best predicted value
        max_value = None
        best_state = None
        
        for state in states:
            value = self.predict_value(np.reshape(state, [1, self.state_size]))
            if max_value is None or value > max_value:
                max_value = value
                best_state = state
        
        return best_state
    
    def train(self, batch_size=32, epochs=1):
        """
        Train on batch from replay memory
        
        Args:
            batch_size: Number of samples to train on
            epochs: Number of training epochs
        """
        n = len(self.memory)
        
        # Not enough memory yet
        if n < self.replay_start_size or n < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        
        # Prepare batched predictions for efficiency
        next_states = np.array([x[1] for x in batch])
        next_qs = [x[0] for x in self.model.predict(next_states, verbose=0)]
        
        x_train = []
        y_train = []
        
        # Build training data
        for i, (state, _, reward, done) in enumerate(batch):
            if not done:
                # Q-learning update: Q = reward + gamma * V(next_state)
                new_q = reward + self.discount * next_qs[i]
            else:
                # Terminal state
                new_q = reward
            
            x_train.append(state)
            y_train.append(new_q)
        
        # Train model
        self.model.fit(np.array(x_train), np.array(y_train), 
                      batch_size=batch_size, epochs=epochs, verbose=0)
        
        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save model to file"""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load model from file"""
        self.model = load_model(filepath)
