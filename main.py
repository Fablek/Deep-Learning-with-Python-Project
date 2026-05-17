import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from src.data_loader import DataLoader
from src.preprocessing import StandardScalerStrategy
from src.models import RegressionModelFactory
from src.evaluator import ModelEvaluator
from src.plotter import ResultPlotter


def get_hyperparams(n_samples):
    if n_samples < 5:
        return {
            'hidden_layers': [8, 4],
            'dropout_rate': 0.0,
            'learning_rate': 0.005,
            'patience': 50,
        }
    elif n_samples < 15:
        return {
            'hidden_layers': [32, 16],
            'dropout_rate': 0.1,
            'learning_rate': 0.002,
            'patience': 30,
        }
    else:
        return {
            'hidden_layers': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'patience': 20,
        }


def main():
    print("=" * 60)
    print("  Deep Learning Regression - 16 Datasets Pipeline")
    print("=" * 60)

    loader = DataLoader(data_dir='data', p_threshold=5)
    datasets = loader.load_all_datasets()

    print(f"\nLoaded {len(datasets)} datasets successfully.\n")

    plotter = ResultPlotter(save_dir='results')
    results_rows = []

    for idx, (dataset_name, (X, y)) in enumerate(datasets.items(), 1):
        n = len(X)
        hp = get_hyperparams(n)
        n_splits = min(5, n)

        print(f"[{idx}/{len(datasets)}] {dataset_name} ({n} samples, {n_splits}-fold, layers={hp['hidden_layers']})")
        print("-" * 50)

        scaler_X = StandardScalerStrategy()
        scaler_y = StandardScalerStrategy()

        model_factory = RegressionModelFactory(
            input_dim=X.shape[1],
            hidden_layers=hp['hidden_layers'],
            dropout_rate=hp['dropout_rate'],
            learning_rate=hp['learning_rate'],
        )

        evaluator = ModelEvaluator(
            model_factory, scaler_X, scaler_y,
            patience=hp['patience'],
        )

        results = evaluator.evaluate(
            X, y,
            n_splits=n_splits,
            epochs=200,
            batch_size=32,
        )

        fold_data = results['fold_1_data']

        plotter.plot_learning_curve(fold_data['history'], dataset_name)
        plotter.plot_scatter(fold_data['y_true'], fold_data['y_pred'], dataset_name)

        results_rows.append({
            'Dataset': dataset_name,
            'Samples': n,
            'K-Folds': n_splits,
            'Layers': str(hp['hidden_layers']),
            'MSE': round(results['mean_mse'], 4),
            'MAE': round(results['mean_mae'], 4),
            'Std_MSE': round(results['std_mse'], 4),
            'Std_MAE': round(results['std_mae'], 4),
        })

        print(f"  => MSE: {results['mean_mse']:.4f} (+/- {results['std_mse']:.4f})")
        print(f"  => MAE: {results['mean_mae']:.4f} (+/- {results['std_mae']:.4f})")
        print()

    report_df = pd.DataFrame(results_rows)
    report_path = os.path.join('results', 'evaluation_report.csv')
    report_df.to_csv(report_path, index=False)

    print("=" * 60)
    print("  FINAL REPORT")
    print("=" * 60)
    print(report_df.to_string(index=False))
    print(f"\nReport saved to: {report_path}")
    print(f"Charts saved to: results/")


if __name__ == '__main__':
    main()
