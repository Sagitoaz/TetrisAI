"""
DQN Agent with Experience Replay and Target Network
Implements the core Q-Learning algorithm
"""

import numpy as np
import random
from collections import deque
import pickle
import os
from .model import DQNModel


class ReplayMemory:
    """
    Experience Replay Memory
    Stores past experiences for training
    """
    
    def __init__(self, capacity=100000):
        """
        Initialize replay memory
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Store an experience
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample random batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Batch of (state, action, reward, next_state, done) tuples
        """
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self):
        """Get current size of memory"""
        return len(self.memory)
    
    def save(self, filepath):
        """
        Save memory to disk
        
        Args:
            filepath: Path to save memory
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.memory), f)
        print(f"Memory saved to {filepath} ({len(self.memory)} experiences)")
    
    def load(self, filepath):
        """
        Load memory from disk
        
        Args:
            filepath: Path to load memory from
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                experiences = pickle.load(f)
                self.memory.clear()
                self.memory.extend(experiences)
            print(f"Memory loaded from {filepath} ({len(self.memory)} experiences)")
        else:
            print(f"Warning: Memory file {filepath} not found")
    
    def merge(self, other_memory):
        """
        Merge another memory into this one
        
        Args:
            other_memory: Another ReplayMemory object
        """
        for experience in other_memory.memory:
            self.push(*experience)


class DQNAgent:
    """
    Deep Q-Network Agent
    
    Implements DQN algorithm with:
    - Experience Replay
    - Target Network
    - Epsilon-greedy exploration
    """
    
    def __init__(self, state_size, action_size, config=None):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            config: Configuration dictionary (optional)
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Default configuration
        default_config = {
            'learning_rate': 0.001,
            'gamma': 0.95,              # Discount factor
            'epsilon': 1.0,              # Exploration rate
            'epsilon_min': 0.01,         # Minimum exploration
            'epsilon_decay': 0.995,      # Exploration decay rate
            'memory_capacity': 100000,   # Replay memory size
            'batch_size': 64,            # Training batch size
            'target_update_freq': 10     # Update target network every N episodes
        }
        
        # Merge with provided config
        self.config = {**default_config, **(config or {})}
        
        # Unpack config for easy access
        self.gamma = self.config['gamma']
        self.epsilon = self.config['epsilon']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay = self.config['epsilon_decay']
        self.batch_size = self.config['batch_size']
        self.target_update_freq = self.config['target_update_freq']
        
        # Initialize replay memory
        self.memory = ReplayMemory(self.config['memory_capacity'])
        
        # Initialize main Q-network
        self.model = DQNModel(
            state_size=state_size,
            action_size=action_size,
            learning_rate=self.config['learning_rate']
        )
        
        # Initialize target Q-network (for stable training)
        self.target_model = DQNModel(
            state_size=state_size,
            action_size=action_size,
            learning_rate=self.config['learning_rate']
        )
        
        # Training statistics
        self.episode_count = 0
        self.training_step = 0
        
        # Copy weights to target network
        self.update_target_model()
    
    def update_target_model(self):
        """
        Update target network weights from main network
        Called periodically for stable Q-learning
        """
        self.target_model.copy_weights(self.model)
        print(f"Target network updated at episode {self.episode_count}")
    
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, always exploit
            
        Returns:
            Selected action (0-6)
        """
        # Exploration: random action
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploitation: best action based on Q-values
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """
        Train on a batch of experiences from replay memory
        
        Implements Q-learning update:
        Q(s,a) = r + γ * max_a' Q_target(s',a')
        
        Returns:
            Average loss for this training step
        """
        # Need enough experiences to sample a batch
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample random minibatch
        minibatch = self.memory.sample(self.batch_size)
        
        # Prepare batch data
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Current Q-values
        current_q_values = self.model.predict(states)
        
        # Target Q-values from target network
        next_q_values = self.target_model.predict(next_states)
        
        # Calculate target Q-values using Bellman equation
        targets = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                # If episode ended, no future reward
                targets[i][actions[i]] = rewards[i]
            else:
                # Q-learning update: Q(s,a) = r + γ * max Q(s',a')
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
        
        return loss
    
    def on_episode_end(self):
        """
        Called at the end of each episode
        Updates target network periodically
        """
        self.episode_count += 1
        
        # Update target network periodically
        if self.episode_count % self.target_update_freq == 0:
            self.update_target_model()
    
    def save(self, filepath_prefix):
        """
        Save agent state (model, target model, memory)
        
        Args:
            filepath_prefix: Prefix for save files (e.g., 'models/checkpoint_100')
        """
        os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
        
        # Save models
        self.model.save(f"{filepath_prefix}_model.h5")
        self.target_model.save(f"{filepath_prefix}_target.h5")
        
        # Save memory
        self.memory.save(f"{filepath_prefix}_memory.pkl")
        
        # Save agent state
        agent_state = {
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'training_step': self.training_step,
            'config': self.config
        }
        
        with open(f"{filepath_prefix}_state.pkl", 'wb') as f:
            pickle.dump(agent_state, f)
        
        print(f"Agent saved to {filepath_prefix}")
    
    def load(self, filepath_prefix):
        """
        Load agent state
        
        Args:
            filepath_prefix: Prefix for load files
        """
        # Load models
        self.model.load(f"{filepath_prefix}_model.h5")
        self.target_model.load(f"{filepath_prefix}_target.h5")
        
        # Load memory
        self.memory.load(f"{filepath_prefix}_memory.pkl")
        
        # Load agent state
        state_file = f"{filepath_prefix}_state.pkl"
        if os.path.exists(state_file):
            with open(state_file, 'rb') as f:
                agent_state = pickle.load(f)
                self.epsilon = agent_state['epsilon']
                self.episode_count = agent_state['episode_count']
                self.training_step = agent_state['training_step']
                # Update config but preserve current settings
                self.config.update(agent_state['config'])
            print(f"Agent loaded from {filepath_prefix}")
        else:
            print(f"Warning: Agent state file {state_file} not found")
    
    def get_stats(self):
        """
        Get current training statistics
        
        Returns:
            Dictionary with training stats
        """
        return {
            'episode': self.episode_count,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'memory_capacity': self.memory.capacity
        }
