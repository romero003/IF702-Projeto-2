import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_pocid(y_true, y_pred):
    """
    Calcula a POCID (Percentage of Correct Directional Predictions).
    Retorna a taxa de acerto da direção do movimento do preço em %.
    """
    y_true = np.array(y_true).squeeze()
    y_pred = np.array(y_pred).squeeze()

    # Variação em relação ao dia anterior (t - t-1)
    diff_true = np.diff(y_true)
    diff_pred = np.diff(y_pred)

    # Verifica se os sinais das variações são iguais (ambos positivos ou ambos negativos)
    correct_directions = (diff_true * diff_pred) > 0

    # POCID em porcentagem
    pocid = np.mean(correct_directions) * 100.0
    return float(pocid)



def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de regressão para preços de Bitcoin,
    incluindo a acurácia da direção (subida/descida).
    """
    y_true = np.array(y_true).squeeze()
    y_pred = np.array(y_pred).squeeze()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Erro Percentual Médio Absoluto)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    pocid = calculate_pocid(y_true, y_pred)


    # Directional Accuracy (Acurácia de Tendência: Subiu ou Caiu?)
    # Compara a variação do dia atual em relação ao dia anterior
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_acc = np.mean(direction_true == direction_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2),
        "mape_percent": float(mape),
        "pocid_percent": float(pocid),
        "directional_accuracy": float(directional_acc)
    }

def plot_predictions_figure(y_true, y_pred, dates=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x_axis = dates if dates is not None else range(len(y_true))
    
    ax.plot(x_axis, y_true, label='Preço Real (USD)', color='blue', alpha=0.8)
    ax.plot(x_axis, y_pred, label='Previsão LSTM (USD)', color='orange', linestyle='--', alpha=0.9)
    
    ax.set_ylabel('Preço (USD)')
    ax.set_xlabel('Tempo / Dias')
    ax.set_title('Comparativo: Preço Real vs Previsão LSTM')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    return fig
