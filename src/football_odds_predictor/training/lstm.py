"""Train the V1 LSTM on one chronological fold."""

import math

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..data.dataset import TARGET_COLUMNS
from ..data.features import HISTORY_FIELDS, HISTORY_LENGTH, SIDES
from ..models.lstm import OddsLSTM

BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 12
SEED = 42
MIN_LOG_ODDS = math.log(1.01)


def _histories(rows: pd.DataFrame) -> np.ndarray:
    """Read the CSV columns as fixtures × teams × ordered matches × stats."""
    columns = [
        f"{side}_match_{step}_{field}"
        for side in SIDES
        for step in range(1, HISTORY_LENGTH + 1)
        for field in HISTORY_FIELDS
    ]
    return rows[columns].to_numpy().reshape(
        len(rows), len(SIDES), HISTORY_LENGTH, len(HISTORY_FIELDS)
    )


def lstm_predictions(
    train: pd.DataFrame, validation: pd.DataFrame,
) -> tuple[np.ndarray, int, float]:
    """Return validation odds, the best epoch, and its training log-odds MAE."""
    train_history = _histories(train)
    validation_history = _histories(validation)
    elo_columns = ["home_elo", "away_elo"]

    # Fit scaling only on training data and share the history scale across teams.
    history_scaler = StandardScaler().fit(train_history.reshape(-1, len(HISTORY_FIELDS)))
    elo_scaler = StandardScaler().fit(train[elo_columns])
    train_history, validation_history = (
        history_scaler.transform(values.reshape(-1, len(HISTORY_FIELDS))).reshape(values.shape)
        for values in (train_history, validation_history)
    )
    train_elo = elo_scaler.transform(train[elo_columns])
    validation_elo = elo_scaler.transform(validation[elo_columns])
    train_inputs = tuple(torch.as_tensor(x, dtype=torch.float32) for x in (train_history, train_elo))
    validation_inputs = tuple(
        torch.as_tensor(x, dtype=torch.float32) for x in (validation_history, validation_elo)
    )
    train_targets = torch.as_tensor(np.log(train[TARGET_COLUMNS].to_numpy()), dtype=torch.float32)
    validation_targets = torch.as_tensor(
        np.log(validation[TARGET_COLUMNS].to_numpy()), dtype=torch.float32,
    )

    torch.manual_seed(SEED)
    model = OddsLSTM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.MSELoss()
    batches = DataLoader(
        TensorDataset(*train_inputs, train_targets),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    best_error = math.inf
    best_epoch = 0
    best_predictions = None
    best_train_error = math.inf

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for history, elo, targets in batches:
            optimizer.zero_grad()
            loss = loss_fn(model(history, elo), targets)
            loss.backward()
            optimizer.step()

        # Validation selects the epoch. Its labels never enter backpropagation.
        model.eval()
        with torch.inference_mode():
            log_predictions = model(*validation_inputs).clamp_min(MIN_LOG_ODDS)
            error = (log_predictions - validation_targets).abs().mean().item()
            if error < best_error:
                best_error = error
                best_epoch = epoch
                best_predictions = log_predictions.exp().numpy()
                train_logs = model(*train_inputs).clamp_min(MIN_LOG_ODDS)
                best_train_error = (train_logs - train_targets).abs().mean().item()
        if epoch - best_epoch >= PATIENCE:
            break

    if best_predictions is None:
        raise ValueError("LSTM training produced no finite validation score")
    return best_predictions, best_epoch, best_train_error
