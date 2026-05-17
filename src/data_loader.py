import os
import pandas as pd
import numpy as np
from typing import Tuple

class DataLoader:
    """
    Class responsible for loading the 16 specific datasets.
    Automatically extracts parameters (m, N_C) from filenames, calculates the target (R),
    and filters out noise (low values of p).
    """
    
    def __init__(self, data_dir: str = 'data', p_threshold: int = 5):
        self.data_dir = data_dir
        self.p_threshold = p_threshold
        # Mapping network number to parameter 'm' based on project hints
        self.m_mapping = {1: 10, 2: 21, 3: 30, 4: 20}
        
    def _parse_filename_params(self, filename: str) -> Tuple[int, int]:
        """Extracts 'm' and default 'N_C' from the filename."""
        # Determine 'm' (network number)
        if 'Ex_1' in filename: network_num = 1
        elif 'Ex_2' in filename: network_num = 2
        elif 'Ex_3' in filename: network_num = 3
        elif 'Ex_4' in filename: network_num = 4
        else: raise ValueError(f"Cannot match network for file: {filename}")
        
        m_val = self.m_mapping[network_num]
        
        # Determine 'N_C' (number of logical processors)
        if filename.startswith('Ex_'): n_c = 32
        elif filename.startswith('ext_Ex_'): n_c = 96
        elif filename.startswith('Ext_'): n_c = 192
        else: n_c = None # For 'par_Ex' files, N_C column is inside the file
        
        return m_val, n_c

    def load_all_data(self) -> pd.DataFrame:
        """Merges all files into a single DataFrame."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory '{self.data_dir}' does not exist!")
            
        all_data = []
        
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.csv') and not filename.startswith('par_Ex'):
                continue
                
            filepath = os.path.join(self.data_dir, filename)
            df = pd.read_csv(filepath)
            
            # Standardize column names based on project hints
            df = df.rename(columns={
                '#d-MPs': 'p', 'n_LU': 'n_LU', 'tim_K2': 'tim_algo1', 
                'tim_K': 'tim_algo1', 'tim_Lp': 'tim_algo2'
            })
            
            # Extract hidden variables from filename
            try:
                m_val, default_nc = self._parse_filename_params(filename)
                df['m'] = m_val
                if default_nc is not None:
                    df['N_C'] = default_nc
            except ValueError:
                continue
                
            # Calculate target variable R
            df['R'] = df['tim_algo1'] / df['tim_algo2']
            df['source_file'] = filename
            
            required_cols = ['m', 'p', 'n_LU', 'N_C', 'R', 'source_file']
            if not all(col in df.columns for col in required_cols):
                continue
                
            all_data.append(df[required_cols])
            
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Filter out noise (small p values)
        final_df = final_df[final_df['p'] >= self.p_threshold]
        # Remove infinite values (e.g., division by zero) and NaNs
        final_df = final_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        return final_df

    def get_X_y(self) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """Returns matrices ready for the Neural Network."""
        df = self.load_all_data()
        X = df[['m', 'p', 'n_LU', 'N_C']].values
        y = df['R'].values
        sources = df['source_file']
        return X, y, sources