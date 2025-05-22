import torch
from unity_gpu.model import SimpleLSTM

def test_lstm_forward():
    """Pass a fake batch through the net and check output shape."""
    batch_size, seq_len, n_features = 8, 20, 6
    x = torch.randn(batch_size, seq_len, n_features)
    model = SimpleLSTM(n_features=n_features, hidden=16)
    y = model(x)
    assert y.shape == (batch_size, 1)
