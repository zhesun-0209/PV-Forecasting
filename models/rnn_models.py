"""
RNN forecasting models: LSTM & GRU.
"""
import torch
import torch.nn as nn

class RNNBase(nn.Module):
    def __init__(self, hist_dim: int, fcst_dim: int, config: dict, rnn_type: str = 'LSTM'):                                                                             
        super().__init__()
        self.cfg = config
        hidden = config['hidden_dim']
        layers = config['num_layers']

        self.hist_proj = nn.Linear(hist_dim, hidden) if hist_dim > 0 else None
        self.fcst_proj = nn.Linear(fcst_dim, hidden) if config.get('use_forecast', False) and fcst_dim > 0 else None                                                    

        # 简化的RNN架构，与ML模型保持一致 | Simplified RNN architecture, consistent with ML models
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden, hidden, num_layers=layers,
                               batch_first=True, dropout=config['dropout'])
        else:
            self.rnn = nn.GRU(hidden, hidden, num_layers=layers,
                              batch_first=True, dropout=config['dropout'])

        self.head = nn.Sequential(
            nn.Linear(hidden, config['future_hours']),
            nn.Softplus()
        )

    def forward(self, hist: torch.Tensor, fcst: torch.Tensor = None) -> torch.Tensor:                                                                                   
        seqs = []

        if self.hist_proj is not None and hist.shape[-1] > 0:
            h_proj = self.hist_proj(hist)
            seqs.append(h_proj)

        if self.fcst_proj is not None and fcst is not None and fcst.shape[-1] > 0:
            f_proj = self.fcst_proj(fcst)
            seqs.append(f_proj)

        if not seqs:
            raise ValueError("Both hist and forecast inputs are missing or zero-dimensional.")                                                                          

        seq = torch.cat(seqs, dim=1)  # (B, past+future, hidden)
        out, _ = self.rnn(seq)        # (B, seq_len, hidden)
        return self.head(out[:, -1, :])  # (B, future_hours)


class LSTM(nn.Module):
    """简化的LSTM forecasting model - 与其他模型保持一致的复杂度 | Simplified LSTM forecasting model - consistent complexity with other models"""
    def __init__(self, hist_dim: int, fcst_dim: int, config: dict):
        super().__init__()
        self.cfg = config
        hidden = config['hidden_dim']
        layers = config['num_layers']

        self.hist_proj = nn.Linear(hist_dim, hidden) if hist_dim > 0 else None
        self.fcst_proj = nn.Linear(fcst_dim, hidden) if config.get('use_forecast', False) and fcst_dim > 0 else None                                                    

        # 简化的LSTM层 | Simplified LSTM layers
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers,
                           batch_first=True, dropout=config['dropout'])

        # 统一的简单输出层 | Unified simple output layer
        self.head = nn.Linear(hidden, config['future_hours'])

    def forward(self, hist: torch.Tensor, fcst: torch.Tensor = None) -> torch.Tensor:                                                                                   
        seqs = []

        if self.hist_proj is not None and hist.shape[-1] > 0:
            h_proj = self.hist_proj(hist)
            seqs.append(h_proj)

        if self.fcst_proj is not None and fcst is not None and fcst.shape[-1] > 0:
            f_proj = self.fcst_proj(fcst)
            seqs.append(f_proj)

        if not seqs:
            raise ValueError("Both hist and forecast inputs are missing or zero-dimensional.")                                                                          

        seq = torch.cat(seqs, dim=1)  # (B, past+future, hidden)
        out, _ = self.lstm(seq)
        
        # 使用最后一个时间步的输出 | Use output from last time step
        return self.head(out[:, -1, :])

class GRU(nn.Module):
    """简化的GRU forecasting model - 与其他模型保持一致的复杂度 | Simplified GRU forecasting model - consistent complexity with other models"""
    def __init__(self, hist_dim: int, fcst_dim: int, config: dict):
        super().__init__()
        self.cfg = config
        hidden = config['hidden_dim']
        layers = config['num_layers']

        self.hist_proj = nn.Linear(hist_dim, hidden) if hist_dim > 0 else None
        self.fcst_proj = nn.Linear(fcst_dim, hidden) if config.get('use_forecast', False) and fcst_dim > 0 else None                                                    

        # 简化的GRU层 | Simplified GRU layers
        self.gru = nn.GRU(hidden, hidden, num_layers=layers,
                          batch_first=True, dropout=config['dropout'])

        # 统一的简单输出层 | Unified simple output layer
        self.head = nn.Linear(hidden, config['future_hours'])

    def forward(self, hist: torch.Tensor, fcst: torch.Tensor = None) -> torch.Tensor:                                                                                   
        seqs = []

        if self.hist_proj is not None and hist.shape[-1] > 0:
            h_proj = self.hist_proj(hist)
            seqs.append(h_proj)

        if self.fcst_proj is not None and fcst is not None and fcst.shape[-1] > 0:
            f_proj = self.fcst_proj(fcst)
            seqs.append(f_proj)

        if not seqs:
            raise ValueError("Both hist and forecast inputs are missing or zero-dimensional.")                                                                          

        seq = torch.cat(seqs, dim=1)
        out, _ = self.gru(seq)
        
        # 使用最后一个时间步的输出 | Use output from last time step
        return self.head(out[:, -1, :])
