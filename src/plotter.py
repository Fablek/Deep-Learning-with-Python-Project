import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any


class ResultPlotter:

    def __init__(self, save_dir: str = 'results'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_learning_curve(self, history: Dict[str, Any], dataset_name: str) -> None:
        plt.figure(figsize=(10, 6))

        train_loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(train_loss) + 1)

        plt.plot(epochs, train_loss, label='Training Loss', color='blue', linewidth=2)
        plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=2)

        plt.title(f'Learning Curve - {dataset_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(fontsize=12)

        clean_name = dataset_name.replace('.csv', '')
        filepath = os.path.join(self.save_dir, f'learning_curve_{clean_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str) -> None:
        plt.figure(figsize=(8, 8))

        sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, color='purple', edgecolor='w', s=60)

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

        plt.title(f'True vs Predicted R\nDataset: {dataset_name}', fontsize=14, fontweight='bold')
        plt.xlabel('True R', fontsize=12)
        plt.ylabel('Predicted R', fontsize=12)
        plt.legend()

        clean_name = dataset_name.replace('.csv', '')
        filepath = os.path.join(self.save_dir, f'scatter_{clean_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")
