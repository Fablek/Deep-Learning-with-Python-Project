from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ScalingStrategy(ABC):
    """
    Abstract interface for scaling strategies (Strategy Pattern).
    Ensures that any applied scaler provides the exact same interface.
    """

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fits the scaler to the data (computes mean/std etc.)."""
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transforms the data using the fitted scaler."""
        pass

    @abstractmethod
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fits the scaler and transforms the data in one step."""
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverts the scaled data back to its original values."""
        pass


class StandardScalerStrategy(ScalingStrategy):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Highly recommended for Deep Learning regression models.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data: np.ndarray) -> None:
        self.scaler.fit(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)


class MinMaxScalerStrategy(ScalingStrategy):
    """
    Transforms features by scaling each feature to a given range (typically 0 to 1).
    """
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, data: np.ndarray) -> None:
        self.scaler.fit(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)
