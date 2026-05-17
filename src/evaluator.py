import numpy as np
from sklearn.model_selection import KFold
from typing import Dict, Any
from .preprocessing import ScalingStrategy
from .models import RegressionModelFactory


class ModelEvaluator:

    def __init__(self,
                 model_factory: RegressionModelFactory,
                 scaler_X: ScalingStrategy,
                 scaler_y: ScalingStrategy):
        self.model_factory = model_factory
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def evaluate(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 n_splits: int = 5,
                 epochs: int = 100,
                 batch_size: int = 32) -> Dict[str, Any]:

        n_samples = len(X)
        actual_splits = min(n_splits, n_samples)

        kfold = KFold(n_splits=actual_splits, shuffle=True, random_state=42)

        fold_metrics = {'mse': [], 'mae': []}
        fold_1_data = {}

        y_2d = y.reshape(-1, 1)

        print(f"  Running {actual_splits}-Fold Cross-Validation ({n_samples} samples)...")

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_2d[train_idx], y_2d[val_idx]

            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_val_scaled = self.scaler_X.transform(X_val)

            y_train_scaled = self.scaler_y.fit_transform(y_train)
            y_val_scaled = self.scaler_y.transform(y_val)

            model = self.model_factory.create_model()

            actual_batch_size = min(batch_size, len(X_train))

            history = model.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                epochs=epochs,
                batch_size=actual_batch_size,
                verbose=0
            )

            y_pred_scaled = model.predict(X_val_scaled, verbose=0)

            y_pred_real = self.scaler_y.inverse_transform(y_pred_scaled)
            y_val_real = self.scaler_y.inverse_transform(y_val_scaled)

            mse = np.mean(np.square(y_val_real - y_pred_real))
            mae = np.mean(np.abs(y_val_real - y_pred_real))

            fold_metrics['mse'].append(mse)
            fold_metrics['mae'].append(mae)

            print(f"    Fold {fold + 1}/{actual_splits} - MSE: {mse:.4f}, MAE: {mae:.4f}")

            if fold == 0:
                fold_1_data = {
                    'history': history.history,
                    'y_true': y_val_real.flatten(),
                    'y_pred': y_pred_real.flatten(),
                }

        return {
            'mean_mse': np.mean(fold_metrics['mse']),
            'std_mse': np.std(fold_metrics['mse']),
            'mean_mae': np.mean(fold_metrics['mae']),
            'std_mae': np.std(fold_metrics['mae']),
            'fold_1_data': fold_1_data,
        }
