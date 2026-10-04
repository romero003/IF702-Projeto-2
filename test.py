import os
import optuna
import wandb
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import get_dataloaders
from src.model_lstm import build_lstm
#from src.metrics import calculate_metrics, plot_confusion_matrix_figure
from src.utils import TimedModel
from src.utils import desnormalizar_target

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ENTITY = "Proj-IF702"
PROJECT = "miniprojeto2-bitcoin-lstm"

def fetch_best_run_from_wandb(group_name, metric_name="val_loss", mode="min"):
    """
    Busca todas as runs do grupo no WandB e retorna a melhor run com base na métrica informada.
    
    :param group_name: Nome do grupo no WandB.
    :param metric_name: Nome da métrica registrada no summary (ex: 'val_mse', 'val_loss', 'val_mape').
    :param mode: 'min' se menor é melhor (perdas/erros), 'max' se maior é melhor (acurácia).
    """
    api = wandb.Api()
    
    # Filtra apenas pelo grupo diretamente no servidor do WandB
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": group_name})
    
    best_run = None
    best_value = float("inf") if mode == "min" else float("-inf")
    
    for run in runs:
        # Pega o valor da métrica registrada na run
        metric_value = run.summary.get(metric_name)
        
        if metric_value is None:
            continue
            
        # Atualiza a melhor run dependendo do modo ('min' ou 'max')
        if mode == "min" and metric_value < best_value:
            best_value = metric_value
            best_run = run
        elif mode == "max" and metric_value > best_value:
            best_value = metric_value
            best_run = run
            
    return best_run

def download_checkpoint_from_wandb(run, preferred_prefix="best_loss_"):
    """
    Faz o download do arquivo .pth associado à run diretamente do servidor do WandB.
    """
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Procura o arquivo .pth salvo nos arquivos da run
    target_file = None
    fallback_file = None
    for file in run.files():
        if file.name.endswith(".pth"):
            fallback_file = fallback_file or file
            if os.path.basename(file.name).startswith(preferred_prefix):
                target_file = file
                break

    target_file = target_file or fallback_file
            
    if target_file is None:
        raise FileNotFoundError(f"Nenhum arquivo .pth encontrado na run {run.id} do WandB.")

    download_path = os.path.join("./checkpoints", os.path.basename(target_file.name))
    print(f"📥 Baixando '{target_file.name}' da run {run.id}...")
    target_file.download(root="./checkpoints", replace=True)
    print(f"✅ Checkpoint salvo localmente em: {download_path}")
    
    return download_path


def evaluate_on_test(model, test_loader, scaler, criterion=None):
    if criterion is None:
        criterion = torch.nn.MSELoss()

    model.eval()
    all_preds, all_targets = [], []
    timed_model = TimedModel(model)
    running_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = timed_model(x_batch)

            loss = criterion(outputs.view_as(y_batch), y_batch)
            running_loss += loss.item()

            all_preds.extend(outputs.view(-1).cpu().numpy())
            all_targets.extend(y_batch.view(-1).cpu().numpy())

    avg_loss = running_loss / len(test_loader)
    avg_inference_time = timed_model.total_time / len(test_loader)

    # Desnormaliza para ter os valores reais em dólares (USD)
    targets_usd = desnormalizar_target(np.array(all_targets), scaler)
    preds_usd = desnormalizar_target(np.array(all_preds), scaler)

    mape = np.mean(np.abs((targets_usd - preds_usd) / targets_usd)) * 100
    mae = np.mean(np.abs(targets_usd - preds_usd))
    rmse = np.sqrt(np.mean((targets_usd - preds_usd) ** 2))

    actual_diff = np.diff(targets_usd)
    pred_diff = np.diff(preds_usd)
    pocid = np.mean((actual_diff * pred_diff) > 0) * 100

    metrics = {
        "test_mape": mape,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_pocid": pocid
    }

    return metrics, avg_loss, avg_inference_time

def evaluate_best_model_wandb(group_name, model_type="LSTM"):   
    """
    Obtém a melhor run do WandB Online, baixa o checkpoint e avalia no Teste (Regressão LSTM).
    """
    print(f"\n================ AVALIANDO MELHOR MODELO ONLINE ({model_type.upper()}) ================")
    
    try:
        run = fetch_best_run_from_wandb(group_name, metric_name="best_val_mse", mode="min")
        if not run:
            print(f"❌ Nenhuma run encontrada para o grupo '{group_name}'.")
            return None
    except Exception as e:
        print(f"❌ Erro ao conectar com a API do WandB: {e}")
        return None

    os.makedirs("test_results", exist_ok=True)
    config = run.config
    val_loss = run.summary.get("best_val_mse", run.summary.get("val_loss", None))
    
    user_info = run.user.username if hasattr(run, "user") and run.user else "N/A"
    print(f"Run Campeã WandB: {run.name} (ID: {run.id}) | Criador: {user_info}")
    if val_loss is not None:
        print(f"Val Loss (WandB Summary): {val_loss:.6f}")
    print("Config do WandB Online:", config)

    # 1. Baixa o checkpoint diretamente do WandB
    try:
        checkpoint_path = download_checkpoint_from_wandb(run, preferred_prefix="best_loss_")
    except Exception as e:
        print(f"❌ Erro no download do modelo: {e}")
        return None

    batch_size = config.get("batch_size", 32)
    seq_length = config.get("seq_length", 30)

    # 2. Carrega o DataLoader de Teste e o Scaler (4 retornos)
    _, _, test_loader, scaler = get_dataloaders(batch_size=batch_size, seq_length=seq_length)

    # 3. Instancia o modelo LSTM usando a configuração da nuvem
    hidden_size = config.get("hidden_size", 64)
    num_layers = config.get("num_layers", 2)
    dropout_rate = config.get("dropout_rate", 0.2)

    model = build_lstm(
        input_size=5,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout_rate=dropout_rate
    ).to(device)

    # 4. Carrega os pesos salvos no modelo
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 5. Avalia no conjunto de Teste (Regressão)
    metrics, test_loss, inf_time = evaluate_on_test(model, test_loader, scaler)

    run.summary.update({
        "test_loss": float(test_loss),
        "test_mape_percent": float(metrics["test_mape"]),
        "test_mae_usd": float(metrics["test_mae"]),
        "test_rmse_usd": float(metrics["test_rmse"]),
        "test_pocid_percent": float(metrics["test_pocid"]),
        "test_inference_time_batch": float(inf_time),
        "test_checkpoint": os.path.basename(checkpoint_path)
    })
    run.update()

    print(f"-> TEST Loss: {test_loss:.6f}")
    print(f"-> TEST MAPE: {metrics['test_mape']:.2f}%")
    print(f"-> TEST MAE: {metrics['test_mae']:.4f}")
    print(f"-> TEST RMSE: {metrics['test_rmse']:.4f}")
    print(f"-> TEST POCID (Acurácia Direcional): {metrics['test_pocid']:.2f}%")
    print(f"-> Tempo de Inferência/Batch: {inf_time:.6f} s")

    return {
        "Model": model_type.upper(),
        "WandB Run ID": run.id,
        "Val Loss (WandB)": round(val_loss, 6) if val_loss is not None else None,
        "Test Loss": round(test_loss, 6),
        "Test MAPE (%)": round(metrics["test_mape"], 2),
        "Test MAE": round(metrics["test_mae"], 4),
        "Test RMSE": round(metrics["test_rmse"], 4),
        "Test POCID (%)": round(metrics["test_pocid"], 2),
        "Inf Time (s/batch)": round(inf_time, 6),
        "Batch Size": batch_size,
        "Seq Length": seq_length,
        "Hidden Size": hidden_size,
        "Num Layers": num_layers,
        "LR": config.get("lr")
    }
    


if __name__ == "__main__":
    # Define the WandB group name used during training/optuna tuning
    GROUP_LSTM = "lstm_optimization" 

    # Evaluate the best LSTM model
    lstm_best = evaluate_best_model_wandb(GROUP_LSTM, "LSTM")

    # Store results (supports adding more groups later if you compare baselines vs tuned models)
    results = [res for res in [lstm_best] if res is not None]
    
    if results:
        df = pd.DataFrame(results)
        print("\n================ COMPARAÇÃO DOS MELHORES MODELOS (TESTE) ================\n")
        print(df.to_string(index=False))

        # Save results to CSV
        df.to_csv("test_results/best_models_comparison.csv", index=False)
        print("\n✅ Resultados salvos com sucesso em 'test_results/best_models_comparison.csv'.")
