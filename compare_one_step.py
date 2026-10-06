"""Compare one-step LSTM forecasts with the previous-close baseline.

Each test-day forecast uses the real observations available through the
previous day. The two LSTM checkpoints were selected on validation data;
validation residuals also set the width of the approximate error bands.

Run from the project root with::

    python compare_one_step.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data_loader import get_dataloaders
from src.model_lstm import build_lstm


MODELS = (
    {
        "trial": 32,
        "label": "Best validation MSE",
        "checkpoint": "best_loss_lstm_trial_32.pth",
        "seq_length": 70,
        "hidden_size": 128,
        "num_layers": 1,
        "dropout_rate": 0.2266711029553587,
        "filename": "prediction_intervals_best_mse_03_10.png",
    },
    {
        "trial": 46,
        "label": "Best validation POCID",
        "checkpoint": "best_pocid_lstm_trial_46.pth",
        "seq_length": 60,
        "hidden_size": 256,
        "num_layers": 2,
        "dropout_rate": 0.30421815675522534,
        "filename": "prediction_intervals_best_pocid_03_10.png",
    },
)

DEFAULT_START = "2023-01-29"
DEFAULT_STEPS = 100
Z_VALUE = 1.96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(min(4, torch.get_num_threads()))


def predict_scaled(model, windows, batch_size=16):
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in torch.split(windows, batch_size):
            predictions.append(model(batch.to(DEVICE)).view(-1).cpu().numpy())
    return np.concatenate(predictions)


def test_target_dates(filepath, seq_length):
    dates = pd.read_csv(filepath, usecols=["date"])
    dates["date"] = pd.to_datetime(dates["date"])
    dates = dates.sort_values("date").reset_index(drop=True)
    val_end = int(len(dates) * (0.65 + 0.15))
    return dates["date"].iloc[val_end + seq_length:].reset_index(drop=True)


def evaluate_model(config, checkpoint_dir, filepath, start_date, steps):
    _, val_loader, test_loader, scaler = get_dataloaders(
        filepath=filepath,
        batch_size=16,
        seq_length=config["seq_length"],
    )
    checkpoint = checkpoint_dir / config["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    model = build_lstm(
        input_size=5,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=1,
        dropout_rate=config["dropout_rate"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))

    dates = test_target_dates(filepath, config["seq_length"])
    if len(dates) != len(test_loader.dataset):
        raise ValueError("Test dates and model windows do not align.")
    matches = np.flatnonzero(
        dates.dt.normalize().to_numpy() == pd.Timestamp(start_date).normalize()
    )
    if len(matches) != 1:
        raise ValueError(
            f"{start_date} is not a test target date for trial {config['trial']} "
            f"({dates.iloc[0].date()} to {dates.iloc[-1].date()})."
        )
    start = int(matches[0])
    end = start + steps
    if end > len(dates):
        raise ValueError(f"Only {len(dates) - start} test steps remain after {start_date}.")

    close_mean = scaler.mean_[3]
    close_scale = scaler.scale_[3]
    val = val_loader.dataset
    val_targets = val.y[:, 0].numpy() * close_scale + close_mean
    val_lstm = predict_scaled(model, val.X) * close_scale + close_mean
    val_rw = val.X[:, -1, 3].numpy() * close_scale + close_mean

    test = test_loader.dataset
    windows = test.X[start:end]
    targets = test.y[start:end, 0].numpy() * close_scale + close_mean
    lstm = predict_scaled(model, windows) * close_scale + close_mean
    # For target t, X[t]'s final row is the real observation at t-1.
    random_walk = windows[:, -1, 3].numpy() * close_scale + close_mean

    context_length = max(1, int(np.ceil(config["seq_length"] * 0.25)))
    context = test.X[start, -context_length:, 3].numpy() * close_scale + close_mean

    return {
        "config": config,
        "dates": dates.iloc[start:end],
        "context": context,
        "targets": targets,
        "lstm": lstm,
        "random_walk": random_walk,
        "lstm_sigma": float(np.std(val_targets - val_lstm, ddof=1)),
        "rw_sigma": float(np.std(val_targets - val_rw, ddof=1)),
    }


def plot_comparison(result, output_path):
    config = result["config"]
    targets = result["targets"]
    lstm = result["lstm"]
    random_walk = result["random_walk"]
    context = result["context"]
    x = np.arange(1, len(targets) + 1)
    context_x = np.arange(1 - len(context), 1)
    lstm_error = np.abs(targets - lstm)
    rw_error = np.abs(targets - random_walk)
    lstm_rmse = float(np.sqrt(np.mean((targets - lstm) ** 2)))
    rw_rmse = float(np.sqrt(np.mean((targets - random_walk) ** 2)))

    fig, (price_ax, error_ax) = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.13},
    )

    for ax in (price_ax, error_ax):
        ax.axvspan(
            context_x[0], 0, color="0.9", alpha=0.55,
            label=(
                f"Shown context ({len(context)} of {config['seq_length']} input steps)"
                if ax is price_ax else None
            ),
        )
        ax.axvline(0, color="0.35", linestyle=":", linewidth=1)
        ax.grid(True, linestyle=":", alpha=0.55)

    price_ax.plot(
        np.r_[context_x, x], np.r_[context, targets],
        color="black", linewidth=1.6, label="Actual close",
    )
    price_ax.plot(x, lstm, color="tab:blue", linewidth=1.3, label="LSTM one-step")
    price_ax.fill_between(
        x, lstm - Z_VALUE * result["lstm_sigma"],
        lstm + Z_VALUE * result["lstm_sigma"],
        color="tab:blue", alpha=0.16, label="LSTM residual band (±1.96σ)",
    )
    price_ax.plot(
        x, random_walk, color="tab:orange", linewidth=1.2,
        linestyle="--", label="Random walk (previous close)",
    )
    price_ax.fill_between(
        x, random_walk - Z_VALUE * result["rw_sigma"],
        random_walk + Z_VALUE * result["rw_sigma"],
        color="tab:orange", alpha=0.12,
        label="Random-walk residual band (±1.96σ)",
    )
    price_ax.set_title(
        f"{config['label']} — Trial {config['trial']} | "
        f"{result['dates'].iloc[0].date()} to {result['dates'].iloc[-1].date()}\n"
        f"One-step test RMSE: LSTM USD {lstm_rmse:,.0f}, "
        f"random walk USD {rw_rmse:,.0f}",
        fontweight="bold",
    )
    price_ax.set_ylabel("Close price (USD)")
    price_ax.legend(loc="upper left", ncol=2, fontsize=9)

    error_ax.plot(
        x, lstm_error, color="tab:blue", linewidth=1.4,
        label=f"LSTM absolute error (MAE USD {np.mean(lstm_error):,.0f})",
    )
    error_ax.plot(
        x, rw_error, color="tab:orange", linewidth=1.4, linestyle="--",
        label=f"Random-walk absolute error (MAE USD {np.mean(rw_error):,.0f})",
    )
    error_ax.set_xlim(context_x[0], len(targets))
    error_ax.set_ylim(bottom=0)
    error_ax.set_ylabel("Absolute error (USD)")
    error_ax.set_xlabel("Test step relative to first forecast")
    error_ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Rolling one-step forecasts and per-step errors", fontsize=15, fontweight="bold")
    fig.text(
        0.5, 0.015,
        "Each forecast uses real observations through the previous day. "
        "Bands use validation residual σ; coverage is not calibrated.",
        ha="center", fontsize=9, color="0.35",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.12, hspace=0.13)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START, help="First test date (YYYY-MM-DD)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--data", type=Path, default=Path("data/data-bitcoin_timedata-2023_v2.csv"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/files"))
    parser.add_argument("--output-dir", type=Path, default=Path("test_results"))
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in MODELS:
        result = evaluate_model(config, args.checkpoint_dir, args.data, args.start, args.steps)
        output_path = args.output_dir / config["filename"]
        plot_comparison(result, output_path)
        for name, predictions, sigma in (
            ("LSTM", result["lstm"], result["lstm_sigma"]),
            ("Random walk", result["random_walk"], result["rw_sigma"]),
        ):
            errors = result["targets"] - predictions
            rows.append({
                "trial": config["trial"],
                "selection": config["label"],
                "model": name,
                "evaluation": "rolling one-step test",
                "start_date": str(result["dates"].iloc[0].date()),
                "end_date": str(result["dates"].iloc[-1].date()),
                "rmse_usd": float(np.sqrt(np.mean(errors ** 2))),
                "mae_usd": float(np.mean(np.abs(errors))),
                "validation_residual_std_usd": sigma,
                "band_half_width_usd": Z_VALUE * sigma,
            })
        print(f"Saved {output_path}")

    metrics_path = args.output_dir / "prediction_intervals_metrics_03_10.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
