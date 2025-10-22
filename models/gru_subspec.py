# -*- coding: utf-8 -*-
import torch, torch.nn as nn

class BandProjection(nn.Module):
    def __init__(self, in_dim, h):
        super().__init__()
        self.proj = nn.Linear(in_dim, h)
    def forward(self, x):  # [B,T,Fb]
        return self.proj(x)

class MeanAttention(nn.Module):
    def __init__(self, d): 
        super().__init__()
        self.w = nn.Linear(d, 1)
    def forward(self, z):  # z: [B,T,D]
        a = torch.softmax(self.w(z).squeeze(-1), dim=1)  # [B,T]
        return (z * a.unsqueeze(-1)).sum(dim=1)          # [B,D]

class GRUSubSpec(nn.Module):
    """
    Subespectrogramas: split -> proj por banda -> GRU compartilhada -> atenção por banda -> concat/fusão -> cabeças.
    """
    def __init__(self, bands_cfg, hidden=128, layers=2, dropout=0.2, state_meta=None):
        super().__init__()
        self.state_meta = state_meta or {k: {"codes":[0,1]} for k in range(1,6)}
        self.h = hidden
        # Projeções específicas por banda
        self.band_projs = nn.ModuleList([BandProjection(in_dim=b["Fb"], h=hidden) for b in bands_cfg])
        # GRU compartilhada (opera no tempo)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout)
        # Atenção por banda (não compartilhada)
        self.band_attn = nn.ModuleList([MeanAttention(2*hidden) for _ in bands_cfg])
        # Fusão linear pós-concatenação
        self.fuse = nn.Linear(2*hidden*len(bands_cfg), 2*hidden)
        self.drop = nn.Dropout(dropout)
        self.motor_head = nn.Linear(2*hidden, 5)
        self.state_heads = nn.ModuleDict({str(k): nn.Linear(2*hidden, len(v["codes"])) for k,v in self.state_meta.items()})

    def forward(self, xb, motor_k=None, bands_slices=None):
        # xb: [B,T,64]; bands_slices: lista de slices (freq) para cada banda
        Hs = []
        for i, slc in enumerate(bands_slices):
            x_band = xb[:, :, slc]                  # [B,T,Fb]
            z_proj = self.band_projs[i](x_band)     # [B,T,h]
            z_enc,_= self.gru(z_proj)               # [B,T,2h] (compartilhada)
            h_b    = self.band_attn[i](z_enc)       # [B,2h]
            Hs.append(h_b)
        z = torch.cat(Hs, dim=1)                    # [B, 2h*B]
        z = self.fuse(z)                            # [B, 2h]
        z = self.drop(z)
        logits_motor = self.motor_head(z)
        if motor_k is None:
            return logits_motor
        B = xb.size(0)
        Cmax = max(len(self.state_meta[int(k)]["codes"]) for k in motor_k.tolist())
        logits_state = xb.new_full((B, Cmax), fill_value=-1e9)
        for i, k in enumerate(motor_k.tolist()):
            head = self.state_heads[str(int(k))]
            out  = head(z[i:i+1])
            logits_state[i, :out.size(1)] = out.squeeze(0)
        return logits_motor, logits_state
