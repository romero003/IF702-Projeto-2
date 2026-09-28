import time
import torch
import numpy as np


class TimedModel:
    def __init__(self, model):
        self.model = model
        self.total_time = 0.0

    def __call__(self, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        outputs = self.model(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.total_time += (time.perf_counter() - start)
        
        return outputs
        
    def reset(self):
        self.total_time = 0.0


def desnormalizar_target(y_scaled, scaler, target_col_idx=3, num_features=5):
    """
    Reverte a escala normalizada para o preço original em USD.
    """
    if y_scaled is None or len(y_scaled) == 0:
        return y_scaled
        
    y_scaled = np.array(y_scaled)
    dummy = np.zeros((len(y_scaled), num_features))
    dummy[:, target_col_idx] = y_scaled.squeeze()
    
    unscaled_data = scaler.inverse_transform(dummy)
    return unscaled_data[:, target_col_idx]