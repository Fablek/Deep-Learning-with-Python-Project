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
        epochs = range(1, len(train_loss) + 1)

        plt.plot(epochs, train_loss, label='Training Loss', color='blue', linewidth=2)

        if 'val_loss' in history:
            val_loss = history['val_loss']
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

    def plot_nc_predictions(self, nc_values: list, mean_r: list, dataset_name: str) -> None:
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette('viridis', len(nc_values))
        bars = plt.bar([str(nc) for nc in nc_values], mean_r, color=colors, edgecolor='white', linewidth=1.5)

        for bar, val in zip(bars, mean_r):
            plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        plt.title(f'Predicted R for different NC\nDataset: {dataset_name}', fontsize=14, fontweight='bold')
        plt.xlabel('NC (logical processors)', fontsize=12)
        plt.ylabel('Predicted R', fontsize=12)

        clean_name = dataset_name.replace('.csv', '')
        filepath = os.path.join(self.save_dir, f'nc_predictions_{clean_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")

    def plot_global_summary(self, summary_df) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        name_col = summary_df.columns[0]
        sorted_df = summary_df.sort_values(name_col)

        mse_col = 'CV MSE' if 'CV MSE' in sorted_df.columns else 'Test MSE'
        mae_col = 'CV MAE' if 'CV MAE' in sorted_df.columns else 'Test MAE'
        name_col = sorted_df.columns[0]

        axes[0].barh(sorted_df[name_col], sorted_df[mse_col], color='coral', edgecolor='white')
        axes[0].set_xlabel('MSE')
        axes[0].set_title(f'{mse_col} per Dataset', fontweight='bold')
        axes[0].tick_params(axis='y', labelsize=8)

        axes[1].barh(sorted_df[name_col], sorted_df[mae_col], color='steelblue', edgecolor='white')
        axes[1].set_xlabel('MAE')
        axes[1].set_title(f'{mae_col} per Dataset', fontweight='bold')
        axes[1].tick_params(axis='y', labelsize=8)

        plt.tight_layout()
        filepath = os.path.join(self.save_dir, 'global_summary.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")
