"""Predict three log closing odds from two recent match histories."""

import torch
from torch import Tensor, nn

from ..data.features import HISTORY_FIELDS


class OddsLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_size = 16
        self.encoder = nn.LSTM(len(HISTORY_FIELDS), hidden_size, batch_first=True)
        self.odds = nn.Linear(2 * hidden_size + 2, 3)

    def forward(self, history: Tensor, elo: Tensor) -> Tensor:
        """Read both team sequences with the same encoder and independent memory."""
        fixtures, teams, steps, fields = history.shape
        # Treat each team's history as an independent sequence in the batch.
        _, (hidden, _) = self.encoder(history.reshape(fixtures * teams, steps, fields))
        team_vectors = hidden[-1].reshape(fixtures, -1)
        return self.odds(torch.cat((team_vectors, elo), dim=1))
