import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Dict, Any, Tuple
from .preprocessing import ScalingStrategy
from .models import RegressionModelFactory

class ModelEvaluator:
    """
    Engine for evaluating the Deep Learning model using K-Fold Cross-Validation.
    Ensures zero data leakage and extracts required visualization data from the first fold.
    """
    
    def __init__(self, 
                 model_factory: RegressionModelFactory, 
                 scaler_X: ScalingStrategy, 
                 scaler_y: ScalingStrategy):
        """
        Injects dependencies needed for evaluation.
        
        Args:
            model_factory: Factory to generate fresh neural networks.
            scaler_X: Strategy for scaling input features.
            scaler_y: Strategy for scaling the target variable R.
        """
        self.model_factory = model_factory
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def evaluate(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 sources: pd.Series, 
                 n_splits: int = 5, 
                 epochs: int = 100, 
                 batch_size: int = 32) -> Dict[str, Any]:
        """
        Runs the K-Fold validation loop.
        
        Returns:
            Dictionary containing aggregated metrics and extracted data from Fold 1 
            for charting (history, predictions, true values, and source files).
        """
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_metrics = {'mse': [], 'mae': []}
        fold_1_data = {} # Container for data required by the project instructions
        
        # y needs to be 2D for scikit-learn scalers
        y_2d = y.reshape(-1, 1)
        
        print(f"Starting {n_splits}-Fold Cross-Validation...")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"--- Training Fold {fold + 1}/{n_splits} ---")
            
            # 1. Split the data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_2d[train_idx], y_2d[val_idx]
            sources_val = sources.iloc[val_idx].values
            
            # 2. Prevent Data Leakage: Fit ONLY on training data
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_val_scaled = self.scaler_X.transform(X_val)
            
            y_train_scaled = self.scaler_y.fit_transform(y_train)
            y_val_scaled = self.scaler_y.transform(y_val)
            
            # 3. Request a fresh, compiled model from the factory
            model = self.model_factory.create_model()
            
            # 4. Train the model
            history = model.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0 # Set to 1 if you want to see the progress bar
            )
            
            # 5. Evaluate and predict
            y_pred_scaled = model.predict(X_val_scaled, verbose=0)
            
            # Inverse transform to calculate real-world metrics
            y_pred_real = self.scaler_y.inverse_transform(y_pred_scaled)
            y_val_real = self.scaler_y.inverse_transform(y_val_scaled)
            
            # Calculate metrics manually on unscaled data to reflect true business error
            mse = np.mean(np.square(y_val_real - y_pred_real))
            mae = np.mean(np.abs(y_val_real - y_pred_real))
            
            fold_metrics['mse'].append(mse)
            fold_metrics['mae'].append(mae)
            
            print(f"Fold {fold + 1} - Real MSE: {mse:.4f}, Real MAE: {mae:.4f}")
            
            # 6. Task 5.2: Extract specific data from Fold 1 for the 32 charts
            if fold == 0:
                fold_1_data = {
                    'history': history.history,
                    'y_true': y_val_real.flatten(),
                    'y_pred': y_pred_real.flatten(),
                    'sources': sources_val
                }
                
        # Aggregate final results
        results = {
            'mean_mse': np.mean(fold_metrics['mse']),
            'std_mse': np.std(fold_metrics['mse']),
            'mean_mae': np.mean(fold_metrics['mae']),
            'std_mae': np.std(fold_metrics['mae']),
            'fold_1_data': fold_1_data
        }
        
        print("\nCross-Validation Complete!")
        print(f"Average Real MSE: {results['mean_mse']:.4f} (+/- {results['std_mse']:.4f})")
        print(f"Average Real MAE: {results['mean_mae']:.4f} (+/- {results['std_mae']:.4f})")
        
        return results