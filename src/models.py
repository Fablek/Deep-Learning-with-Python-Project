import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from typing import List


class RegressionModelFactory:
    """
    Factory pattern for generating compiled Keras models dynamically.
    Allows for easy hyperparameter tuning (layers, neurons, optimizers)
    during the K-Fold Cross Validation process.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_layers: List[int] = [64, 32],
                 dropout_rate: float = 0.2,
                 optimizer_name: str = 'adam',
                 learning_rate: float = 0.001):
        """
        Initializes the factory with configurable hyperparameters.

        Args:
            input_dim (int): Number of input features.
            hidden_layers (List[int]): List defining the number of neurons in each hidden layer.
            dropout_rate (float): Dropout probability for regularization.
            optimizer_name (str): String identifier for the optimizer ('adam', 'sgd', 'rmsprop').
            learning_rate (float): Learning rate for the chosen optimizer.
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.optimizer_name = optimizer_name.lower()
        self.learning_rate = learning_rate

    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Private method to resolve the optimizer instance based on the string name."""
        if self.optimizer_name == 'adam':
            return Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            return SGD(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'rmsprop':
            return RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def create_model(self) -> Sequential:
        """
        Builds and compiles a Feed-Forward Neural Network (MLP) for regression
        based on the provided hyperparameters.

        Returns:
            Sequential: A compiled Keras model ready for training.
        """
        model = Sequential()

        model.add(Input(shape=(self.input_dim,)))

        for neurons in self.hidden_layers:
            model.add(Dense(neurons, activation='relu'))
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))

        model.add(Dense(1, activation='linear'))

        optimizer = self._get_optimizer()
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model
