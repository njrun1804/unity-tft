import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl

class SimpleLSTM(pl.LightningModule):
    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True, num_layers=num_layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])

    def training_step(self, batch, _):
        x, y = batch
        pred = self(x).squeeze(-1)
        loss = F.huber_loss(pred, y, delta=5.0)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        pred = self(x).squeeze(-1)
        loss = F.huber_loss(pred, y, delta=5.0)
        self.log("val_loss", loss, prog_bar=True)
        # hit-rate: did we predict the right sign?
        hits = (torch.sign(pred) == torch.sign(y)).float().mean()
        self.log("val_hit", hits, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
