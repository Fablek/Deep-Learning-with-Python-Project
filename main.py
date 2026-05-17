import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
from src.data_loader import DataLoader
from src.preprocessing import StandardScalerStrategy
from src.models import RegressionModelFactory
from src.evaluator import ModelEvaluator
from src.plotter import ResultPlotter


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
        print(f"[{idx}/{len(datasets)}] Processing: {dataset_name} ({len(X)} samples)")
        print("-" * 50)

        scaler_X = StandardScalerStrategy()
        scaler_y = StandardScalerStrategy()

        input_dim = X.shape[1]
        model_factory = RegressionModelFactory(
            input_dim=input_dim,
            hidden_layers=[64, 32],
            learning_rate=0.001,
        )

        evaluator = ModelEvaluator(model_factory, scaler_X, scaler_y)

        n_splits = min(5, len(X))
        results = evaluator.evaluate(
            X, y,
            n_splits=n_splits,
            epochs=100,
            batch_size=32,
        )

        fold_data = results['fold_1_data']

        plotter.plot_learning_curve(fold_data['history'], dataset_name)
        plotter.plot_scatter(fold_data['y_true'], fold_data['y_pred'], dataset_name)

        results_rows.append({
            'Dataset': dataset_name,
            'Samples': len(X),
            'K-Folds': n_splits,
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
