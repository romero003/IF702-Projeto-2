import numpy as np
import torch
import torch.nn as nn
from src.utils import TimedModel
from tqdm import tqdm

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout_rate=0.0):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Capa LSTM que implementa as compuertas (Forget, Input, Candidate e Output Gates)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        # Camada totalmente conectada para mapear o estado oculto ao valor final do preço
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Selecionamos o último passo de tempo da sequência para gerar a previsão
        last_time_step = lstm_out[:, -1, :]
        
        # Mapeamento para o preço predito
        out = self.fc(last_time_step)
        return out

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)


def build_lstm(input_size=5, hidden_size=64, num_layers=2, output_size=1, dropout_rate=0.2):
    return LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout_rate=dropout_rate
    )

#### -------------------------

def train_epoch(model, dataloader, optimizer, criterion, epoch=0, total_epochs=10):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

    for sequences, targets in pbar:
        optimizer.zero_grad()
        
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate_epoch(model, dataloader, criterion, epoch=0, total_epochs=10):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    timed_model = TimedModel(model)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Valid]", leave=False)
    
    with torch.no_grad():
        for sequences, targets in pbar:
            outputs = timed_model(sequences)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(dataloader)
    avg_time_per_batch = timed_model.total_time / len(dataloader)
            
    return avg_loss, avg_time_per_batch, all_targets, all_preds