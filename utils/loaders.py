# -*- coding: utf-8 -*-
import torch, torch.utils.data as tud
import soundfile as sf, librosa, numpy as np, pandas as pd
from pathlib import Path

class WindowedMelDataset(torch.utils.data.Dataset):
    def __init__(self, df, sr=44100, seg_dur=0.5, seg_hop=0.25, n_mels=64, n_fft=1024, hop=256, fmin=20, fmax=8000):
        self.df = df.reset_index(drop=True).copy()
        self.sr = sr
        self.seg = int(seg_dur*sr)
        self.hop = int(seg_hop*sr)
        self.n_mels=n_mels; self.n_fft=n_fft; self.hop_length=hop; self.fmin=fmin; self.fmax=fmax
        # indexa (arquivo, start)
        self.idx = []
        for i, row in self.df.iterrows():
            path = Path(row["wav_path"])
            if not path.exists(): continue
            try:
                f = sf.SoundFile(str(path)); dur_s = len(f)/f.samplerate
            except:
                dur_s = 60.0
            total = int(dur_s*self.sr)
            starts = [0] if total<self.seg else list(range(0, total-self.seg+1, self.hop))
            for s in starts:
                self.idx.append((i,s))

    def __len__(self): return len(self.idx)

    def _mel(self, y):
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
                                           n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power=2.0)
        Sdb = librosa.power_to_db(S, ref=np.max).astype(np.float32)
        mu, sd = Sdb.mean(), Sdb.std()+1e-6
        return ((Sdb-mu)/sd).T  # [T, n_mels]

    def __getitem__(self, i):
        row_i, start = self.idx[i]
        row = self.df.iloc[row_i]
        y, sr = sf.read(str(row["wav_path"]), always_2d=False)
        if y.ndim>1: y=y.mean(axis=1)
        if sr!=self.sr: y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        end = start + self.seg
        seg = y[start:end] if end<=len(y) else np.pad(y[start:], (0, end-len(y)))
        x = torch.from_numpy(self._mel(seg))
        return x, int(row["motor_idx"]), int(row["state_idx"]), int(row["motor_idx"])+1

def make_dataloaders(cfg):
    # Placeholder: carregar índice pré-formatado (CSV) com colunas: wav_path, motor_idx, state_idx
    df = pd.read_csv(cfg["data"]["index_csv"])
    # split simples por arquivo (já fornecido pelo CSV, por exemplo "split" em {train,test})
    df_tr = df[df["split"]=="train"].copy()
    df_va = df[df["split"]=="valid"].copy() if "valid" in df["split"].unique() else df[df["split"]=="test"].copy()

    ds_tr = WindowedMelDataset(df_tr,
                               sr=cfg["features"]["sr"],
                               seg_dur=cfg["features"]["seg_dur"],
                               seg_hop=cfg["features"]["seg_hop"],
                               n_mels=cfg["features"]["n_mels"],
                               n_fft=cfg["features"]["n_fft"],
                               hop=cfg["features"]["hop"],
                               fmin=cfg["features"]["fmin"],
                               fmax=cfg["features"]["fmax"])
    ds_va = WindowedMelDataset(df_va,
                               sr=cfg["features"]["sr"],
                               seg_dur=cfg["features"]["seg_dur"],
                               seg_hop=cfg["features"]["seg_hop"],
                               n_mels=cfg["features"]["n_mels"],
                               n_fft=cfg["features"]["n_fft"],
                               hop=cfg["features"]["hop"],
                               fmin=cfg["features"]["fmin"],
                               fmax=cfg["features"]["fmax"])
    dl_tr = tud.DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0)
    dl_va = tud.DataLoader(ds_va, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)

    # meta mínimo (exemplo)
    meta = {"state_meta": {k: {"codes":[0,1,2,3,4]} for k in range(1,6)}}
    return dl_tr, dl_va, meta
