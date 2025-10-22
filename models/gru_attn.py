# -*- coding: utf-8 -*-
import torch, torch.nn as nn

class MeanAttention(nn.Module):
    def __init__(self, d): 
        super().__init__()
        self.w = nn.Linear(d, 1)
    def forward(self, z):  # z: [B,T,D]
        a = torch.softmax(self.w(z).squeeze(-1), dim=1)  # [B,T]
        return (z * a.unsqueeze(-1)).sum(dim=1)          # [B,D]

class MultiTaskGRUAttn(nn.Module):
    """
    GRU bidirecional com atenção + cabeças condicionais por motor.
    """
    def __init__(self, n_mels=64, hidden=128, layers=2, dropout=0.2, state_meta=None):
        super().__init__()
        self.state_meta = state_meta or {k: {"codes":[0,1]} for k in range(1,6)}
        self.gru = nn.GRU(n_mels, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = MeanAttention(2*hidden)
        self.drop = nn.Dropout(dropout)
        self.motor_head = nn.Linear(2*hidden, 5)
        self.state_heads = nn.ModuleDict({str(k): nn.Linear(2*hidden, len(v["codes"])) for k,v in self.state_meta.items()})

    def forward(self, x, motor_k=None):
        z,_ = self.gru(x)                    # [B,T,2H]
        z  = self.attn(z)                    # [B,2H]
        z  = self.drop(z)
        logits_motor = self.motor_head(z)    # [B,5]
        if motor_k is None: 
            return logits_motor
        B = x.size(0)
        Cmax = max(len(self.state_meta[int(k)]["codes"]) for k in motor_k.tolist())
        logits_state = x.new_full((B, Cmax), fill_value=-1e9)
        for i, k in enumerate(motor_k.tolist()):
            head = self.state_heads[str(int(k))]
            out  = head(z[i:i+1])           # [1,Ck]
            logits_state[i, :out.size(1)] = out.squeeze(0)
        return logits_motor, logits_state
