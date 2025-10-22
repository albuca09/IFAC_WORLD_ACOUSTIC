#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal de treino/avaliação multitarefa (motor + estado).
"""
import argparse, yaml, os
from utils.loaders import make_dataloaders
from models.gru_attn import MultiTaskGRUAttn
from utils.train import Trainer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Caminho para o arquivo de configuração YAML")
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dl_train, dl_valid, meta = make_dataloaders(cfg)

    model = MultiTaskGRUAttn(
        n_mels=cfg["features"]["n_mels"],
        hidden=cfg["model"]["hidden"],
        layers=cfg["model"]["layers"],
        dropout=cfg["model"]["dropout"],
        state_meta=meta["state_meta"],
    )

    trainer = Trainer(cfg)
    trainer.fit(model, dl_train, dl_valid, meta)

if __name__ == "__main__":
    main()
