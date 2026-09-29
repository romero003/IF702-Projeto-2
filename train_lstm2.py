import os
import optuna
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.data_loader import get_dataloaders
from src.model_lstm import build_lstm, train_epoch, validate_epoch
from src.metrics import calculate_metrics, plot_predictions_figure
from src.utils import desnormalizar_target


def objective(trial):
    # --- Hiperparâmetros a serem otimizados ---
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    seq_length = trial.suggest_categorical("seq_length", [15, 30, 60, 70, 90]) 
    criterion_name = trial.suggest_categorical("criterion", ["MSELoss", "L1Loss"]) # ls1=mae = mean absolute error

    run_name = f"LSTM_trial-{trial.number}_lr-{lr:.4f}_seq-{seq_length}_bs-{batch_size}"

    run = wandb.init(
        entity="Proj-IF702",
        project="miniprojeto2-bitcoin-lstm",
        name=run_name,
        group="lstm_optimization",
        config={
            "lr": lr,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "weight_decay": weight_decay,
            "seq_length": seq_length,
            "criterion": criterion_name
        },
        reinit=True
    )

    # 1. Carrega os DataLoaders com a janela de sequência atual do trial
    train_loader, val_loader, _, scaler = get_dataloaders(
        batch_size=batch_size,
        seq_length=seq_length
    )

    # 2. Constrói a arquitetura da LSTM (5 colunas de entrada: open, high, low, close, number_of_trades)
    model = build_lstm(
        input_size=5,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout_rate=dropout_rate
    )

    criterion = nn.MSELoss() if criterion_name == "MSELoss" else nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model_path = os.path.join(wandb.run.dir, f"best_lstm_trial_{trial.number}.pth")

    epochs = 20
    inference_times = []
    best_val_loss = float("inf")
    best_pocid_at_best_loss = 0.0
    best_targets_usd, best_preds_usd = None, None

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch + 1, epochs)
        val_loss, inference_time, targets, preds = validate_epoch(model, val_loader, criterion, epoch + 1, epochs)
        
        inference_times.append(inference_time)

        # Desnormaliza para calcular métricas em dólares (USD)
        targets_usd = desnormalizar_target(targets, scaler)
        preds_usd = desnormalizar_target(preds, scaler)

        metrics = calculate_metrics(targets_usd, preds_usd)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pocid_at_best_loss = metrics["pocid_percent"]
            best_targets_usd, best_preds_usd = targets_usd, preds_usd
            torch.save(model.state_dict(), model_path)

        # Log de métricas no W&B
        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae_usd": metrics["mae"],
            "val_rmse_usd": metrics["rmse"],
            "val_mape_percent": metrics["mape_percent"],
            "val_pocid_percent": metrics["pocid_percent"],
            "val_inference_time_batch": inference_time
        }

        wandb.log(log_data)

        trial.report(val_loss, epoch)

        if trial.should_prune():
            wandb.run.summary["pruned"] = True
            wandb.finish()
            raise optuna.exceptions.TrialPruned()


    # 3. Plota e envia o gráfico comparativo do melhor momento para o W&B
    fig = plot_predictions_figure(best_targets_usd, best_preds_usd)
    wandb.log({"predictions_plot": wandb.Image(fig)})
    plt.close(fig)

    if os.path.exists(model_path):
        wandb.save(model_path, base_path=run.dir)

    average_inference_time = sum(inference_times) / len(inference_times)
    trial.set_user_attr("avg_inference_time", average_inference_time)
    wandb.finish()

    return best_val_loss, best_pocid_at_best_loss


if __name__ == "__main__":
    pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name="bitcoin-lstm-multiobjective",
        storage="sqlite:///bitcoin_lstm_optuna.db",
        directions=["minimize", "maximize"], # Loss e POCID        sampler=optuna.samplers.TPESampler(),
        pruner=pruner,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=30,n_jobs=2)

    print("\n================ MODELOS DA FRONTEIRA DE PARETO ================\n")

    best_trials = study.best_trials  # Retorna todos os trials não-dominados (ótimos em pelo menos 1 aspecto)

    for rank, trial in enumerate(best_trials, 1):
        loss_val, pocid_val = trial.values
        inference_time = trial.user_attrs.get("avg_inference_time")

        print(f"--- Modelo Pareto #{rank} (Trial #{trial.number}) ---")
        print(f"  - Validation Loss (Minimizar): {loss_val:.6f}")
        print(f"  - Validation POCID (Maximizar): {pocid_val:.2f}%")
        if inference_time is not None:
            print(f"  - Avg Inference Time: {inference_time:.6f} sec/batch")
        print("  Hiperparâmetros:")
        for param, val in trial.params.items():
            print(f"    * {param}: {val}")
        print("-" * 50 + "\n")