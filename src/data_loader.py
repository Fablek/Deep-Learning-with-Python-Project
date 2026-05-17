import os
import re
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class DataLoader:

    def __init__(self, data_dir: str = 'data', p_threshold: int = 5):
        self.data_dir = data_dir
        self.p_threshold = p_threshold
        self.m_mapping = {1: 10, 2: 21, 3: 30, 4: 20}

    def _get_network_number(self, filename: str) -> int:
        match = re.search(r'(?:par_|ext_)?Ex[t]?_(\d+)', filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Cannot determine network for: {filename}")

    def _get_default_N_C(self, filename: str):
        if filename.startswith('Ex_'):
            return 32
        elif filename.startswith('ext_Ex_'):
            return 96
        elif filename.startswith('Ext_'):
            return 192
        elif filename.startswith('par_Ex_'):
            return None
        return None

    def load_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)

        df = df.rename(columns={
            '#d-MPs': 'p',
            'tim_K2': 'tim_algo1',
            'tim_K': 'tim_algo1',
            'tim_Lp': 'tim_algo2',
        })

        network_num = self._get_network_number(filename)
        df['m'] = self.m_mapping[network_num]

        default_nc = self._get_default_N_C(filename)
        if default_nc is not None:
            df['N_C'] = default_nc
        else:
            df['N_C'] = df['Nc']

        df['R'] = df['tim_algo1'] / df['tim_algo2']

        df = df[df['p'] >= self.p_threshold]
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        X = df[['m', 'p', 'n_LU', 'N_C']].values
        y = df['R'].values

        X = np.log1p(X)
        y = np.log1p(np.clip(y, 0, None))

        return X, y

    def load_all_datasets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        datasets = {}
        for filename in sorted(os.listdir(self.data_dir)):
            if not filename.endswith('.csv'):
                continue
            filepath = os.path.join(self.data_dir, filename)
            try:
                X, y = self.load_dataset(filepath)
                if len(X) >= 2:
                    datasets[filename] = (X, y)
                else:
                    print(f"Skipping {filename}: too few samples ({len(X)}) after filtering.")
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
        return datasets
