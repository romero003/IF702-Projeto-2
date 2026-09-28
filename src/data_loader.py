import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    """Dataset customizado para criar janelas temporais X (sequência) e y (alvo)."""
    def __init__(self, data, seq_length=30, target_col_idx=-2):
        """
        data: numpy array com as features normalizadas
        seq_length: quantidade de dias/passos passados na sequência
        target_col_idx: índice da coluna alvo (ex: 'close' costuma ser a penúltima)
        """
        self.seq_length = seq_length
        self.X = []
        self.y = []
        
        for i in range(len(data) - seq_length):
            self.X.append(data[i : i + seq_length, :])
            self.y.append(data[i + seq_length, target_col_idx])
            
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(
    filepath='data/data-bitcoin_timedata-2023_v2.csv',
    batch_size=32,
    seq_length=30,
    target_col='close',
    train_ratio=0.65,
    val_ratio=0.15):

    # 1. Carrega o dataset e ordena por data
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Seleciona as colunas numéricas (open, high, low, close, number_of_trades)
    feature_cols = ['open', 'high', 'low', 'close', 'number_of_trades']
    data = df[feature_cols].values
    target_col_idx = feature_cols.index(target_col)

    # 2. Divisão Temporal (sem shuffle para manter a ordem do tempo)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 3. Normalização (Ajusta o scaler APENAS no treino para evitar leakage)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # 4. Criação dos Datasets com janelas de sequência
    train_dataset = TimeSeriesDataset(train_scaled, seq_length=seq_length, target_col_idx=target_col_idx)
    val_dataset   = TimeSeriesDataset(val_scaled, seq_length=seq_length, target_col_idx=target_col_idx)
    test_dataset  = TimeSeriesDataset(test_scaled, seq_length=seq_length, target_col_idx=target_col_idx)

    # 5. Criação dos DataLoaders
    # Nota: shuffle=False é essencial para séries temporais manterem a coerência dos batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler