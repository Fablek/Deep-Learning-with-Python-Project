import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any

class ResultPlotter:
    """
    Class responsible for generating and saving visualization charts.
    Handles Learning Curves and Scatter Plots grouped by source files.
    """
    
    def __init__(self, save_dir: str = 'results'):
        """
        Args:
            save_dir (str): Directory where the charts will be saved.
        """
        self.save_dir = save_dir
        # Ensure the directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        # Set a professional visual theme
        sns.set_theme(style="whitegrid")

    def plot_learning_curve(self, history: Dict[str, Any], filename: str = 'global_learning_curve.png') -> None:
        """
        Plots the training and validation loss over epochs.
        """
        plt.figure(figsize=(10, 6))
        
        # Extract loss arrays
        train_loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(train_loss) + 1)
        
        plt.plot(epochs, train_loss, label='Training Loss (MSE)', color='blue', linewidth=2)
        plt.plot(epochs, val_loss, label='Validation Loss (MSE)', color='orange', linewidth=2)
        
        plt.title('Global Model Learning Curve', fontsize=16, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Mean Squared Error (Scaled)', fontsize=12)
        plt.legend(fontsize=12)
        
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved learning curve to: {filepath}")

    def plot_predictions_per_file(self, y_true: np.ndarray, y_pred: np.ndarray, sources: np.ndarray) -> None:
        """
        Generates individual Scatter Plots (True vs Predicted) for each unique source file.
        """
        unique_files = np.unique(sources)
        
        for file in unique_files:
            # Filter data for this specific file
            mask = (sources == file)
            y_t = y_true[mask]
            y_p = y_pred[mask]
            
            plt.figure(figsize=(8, 8))
            
            # Scatter plot of predictions
            sns.scatterplot(x=y_t, y=y_p, alpha=0.7, color='purple', edgecolor='w', s=60)
            
            # Ideal prediction line (y = x)
            min_val = min(np.min(y_t), np.min(y_p))
            max_val = max(np.max(y_t), np.max(y_p))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
            
            plt.title(f'True vs Predicted R\nDataset: {file}', fontsize=14, fontweight='bold')
            plt.xlabel('True Relative Efficiency (R)', fontsize=12)
            plt.ylabel('Predicted Relative Efficiency (R)', fontsize=12)
            plt.legend()
            
            # Create a safe filename (remove .csv extension if present)
            clean_name = file.replace('.csv', '')
            filepath = os.path.join(self.save_dir, f'scatter_{clean_name}.png')
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
        print(f"Saved {len(unique_files)} scatter plots to the '{self.save_dir}' directory.")