"""
Neural Network Model for DQN
Defines the Deep Q-Network architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os


class DQNModel:
    """
    Deep Q-Network Model
    
    Architecture:
        Input: State representation (flattened grid + features)
        Hidden Layer 1: Dense 256 neurons, ReLU
        Hidden Layer 2: Dense 256 neurons, ReLU
        Hidden Layer 3: Dense 128 neurons, ReLU
        Output: 7 Q-values (one for each action)
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        """
        Initialize DQN Model
        
        Args:
            state_size: Dimension of input state
            action_size: Number of possible actions (7)
            learning_rate: Learning rate for optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build neural network architecture
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.state_size,)),
            
            # Hidden layer 1
            layers.Dense(256, activation='relu', 
                        kernel_initializer='he_uniform',
                        name='hidden_1'),
            layers.Dropout(0.2),  # Prevent overfitting
            
            # Hidden layer 2
            layers.Dense(256, activation='relu',
                        kernel_initializer='he_uniform',
                        name='hidden_2'),
            layers.Dropout(0.2),
            
            # Hidden layer 3
            layers.Dense(128, activation='relu',
                        kernel_initializer='he_uniform',
                        name='hidden_3'),
            
            # Output layer - Q-values for each action
            layers.Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform',
                        name='q_values')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',  # Mean Squared Error for Q-learning
            metrics=['mae']  # Mean Absolute Error for monitoring
        )
        
        return model
    
    def predict(self, state):
        """
        Predict Q-values for a given state
        
        Args:
            state: Input state (can be single state or batch)
            
        Returns:
            Q-values for each action
        """
        if len(state.shape) == 1:
            state = np.reshape(state, [1, self.state_size])
        return self.model.predict(state, verbose=0)
    
    def fit(self, states, targets, epochs=1, verbose=0):
        """
        Train the model on a batch of data
        
        Args:
            states: Batch of states
            targets: Target Q-values
            epochs: Number of training epochs
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        return self.model.fit(states, targets, epochs=epochs, 
                             verbose=verbose, batch_size=len(states))
    
    def save(self, filepath):
        """
        Save model weights
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_weights(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model weights
        
        Args:
            filepath: Path to load the model from
        """
        if os.path.exists(filepath):
            self.model.load_weights(filepath)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Warning: Model file {filepath} not found")
    
    def copy_weights(self, other_model):
        """
        Copy weights from another model (for target network)
        
        Args:
            other_model: Source DQNModel to copy weights from
        """
        self.model.set_weights(other_model.model.get_weights())
    
    def get_config(self):
        """
        Get model configuration
        
        Returns:
            Dictionary with model configuration
        """
        return {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'total_params': self.model.count_params()
        }
    
    def summary(self):
        """Print model architecture summary"""
        self.model.summary()


def create_dueling_dqn(state_size, action_size, learning_rate=0.001):
    """
    Create Dueling DQN architecture (Advanced version)
    
    Separates Value and Advantage streams for better learning
    
    Args:
        state_size: Dimension of input state
        action_size: Number of possible actions
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(state_size,))
    
    # Shared feature extraction
    x = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.Dropout(0.2)(x)
    
    # Value stream - estimates state value V(s)
    value_stream = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    value = layers.Dense(1, kernel_initializer='he_uniform', name='value')(value_stream)
    
    # Advantage stream - estimates advantage A(s,a) for each action
    advantage_stream = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    advantage = layers.Dense(action_size, kernel_initializer='he_uniform', name='advantage')(advantage_stream)
    
    # Combine Value and Advantage: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    # This aggregation ensures identifiability
    q_values = layers.Add()([
        value,
        layers.Subtract()([
            advantage,
            layers.Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
        ])
    ])
    
    model = keras.Model(inputs=inputs, outputs=q_values)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model
