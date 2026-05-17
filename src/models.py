import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

class RegressionModelFactory:
    """
    Factory pattern for generating freshly compiled Keras models.
    Essential for K-Fold Cross Validation to ensure each fold starts with random weights.
    """
    
    def __init__(self, input_dim: int, learning_rate: float = 0.001):
        """
        Initializes the factory with hyperparameters.
        
        Args:
            input_dim (int): Number of input features (m, p, n_LU, N_C).
            learning_rate (float): Learning rate for the Adam optimizer.
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        
    def create_model(self) -> Sequential:
        """
        Builds and compiles a Feed-Forward Neural Network (MLP) for regression.
        
        Returns:
            Sequential: A compiled Keras model ready for training.
        """
        model = Sequential([
            # Input layer defining the number of features
            Input(shape=(self.input_dim,)),
            
            # First hidden layer
            Dense(64, activation='relu'),
            
            # Dropout layer to prevent overfitting (randomly turns off 20% of neurons)
            Dropout(0.2),
            
            # Second hidden layer
            Dense(32, activation='relu'),
            
            # Output layer: 1 neuron with linear activation (standard for regression predicting continuous R)
            Dense(1, activation='linear')
        ])
        
        # Compile the model using Mean Squared Error (MSE) as the loss function
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model