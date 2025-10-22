# Reconhecimento Multitarefa de Motores Subaquáticos com GRU e Atenção
Multitask Recognition of Underwater Engine Types and Operating States via Attention-Based GRU and Sub-Spectral Decomposition

## Descrição
Este repositório contém código, modelos e scripts do artigo:

**“Reconhecimento Multitarefa de Tipos e Estados Operacionais de Motores Subaquáticos a partir da Decomposição Subespectral em Modelo GRU com Atenção.”**

O estudo propõe uma arquitetura multitarefa baseada em GRU bidirecional com mecanismo de atenção aplicada sobre subespectrogramas (a partir de representações log-mel). O modelo realiza identificação do tipo de motor e estimativa do estado operacional usando o dataset Wolfset, com bom desempenho e custo computacional reduzido.

## Estrutura do repositório
```
root/
├─ data/            # Pré-processamento e split do Wolfset
├─ models/          # GRU, GRU-subspec, BiLSTM, Transformer, ResNet-50
├─ experiments/     # Configurações e logs (M1–M7)
├─ utils/           # Métricas, plots, loaders, etc.
├─ notebooks/       # Exemplos reprodutíveis (Jupyter/Colab)
├─ figures/         # Diagramas e resultados
└─ main.py          # Treino e avaliação principais
```

## Modelos implementados
| Modelo | Descrição | Avaliação |
|-------|-----------|-----------|
| M1 | GRU bidirecional com atenção (baseline) | Segmento / Arquivo |
| M2–M6 | GRU com subespectrogramas (B = 6–12 bandas) | Segmento / Arquivo |
| M7 | GRU sem aumento de dados (ablation) | Segmento / Arquivo |
| Baselines | BiLSTM + Atenção, Transformer, ResNet-50 | Comparativo |

## Métricas
- Macro-F1 por tarefa (motor e estado).  
- Score médio multitarefa: média simples entre F1 de motor e F1 de estado.  
- Avaliação em dois níveis:
  - Por segmento (sensibilidade local).
  - Por arquivo (estabilidade global via agregação de logits).

## Dataset: Wolfset
- **Wolfset: A High-Quality Underwater Acoustic Dataset for Algorithm Development and Analysis** (Pessanha Santos et al., 2025).  
- URL: https://zenodo.org/record/11449792  
- Licença: CC-BY 4.0  
- Autores: Nuno Pessanha Santos, Ricardo Moura, Victor Lobo.

## Requisitos
- Python >= 3.9  
- torch >= 2.0  
- torchaudio >= 2.0  
- numpy, scikit-learn, matplotlib, tqdm, pandas

Instalação:
```bash
pip install -r requirements.txt
```

## Como treinar e avaliar
Treinamento do modelo principal (ex.: M4, B = 8):
```bash
python main.py --model rnn_subspec --bands 8 --epochs 50 --batch_size 32
```

Avaliação em nível de arquivo:
```bash
python evaluate.py --input checkpoints/M4_best.pt --level file
```

## Resultados principais (exemplo)
| Modelo | F1 (motor) | F1 (estado) | F1 médio | Tempo (s) | Memória (MB) |
|--------|------------|-------------|----------|-----------|--------------|
| M4: GRU-subspec (8 bandas) | 1.00 | 0.74 | 0.80 | 96 | 216 |
| M1: GRU baseline | 1.00 | 0.70 | 0.78 | 88 | 211 |
| M7: sem aumento de dados | 0.90 | 0.47 | 0.68 | 83 | 187 |

## Citação
Se este código for útil, por favor cite:
```bibtex
@article{Guedes2025WolfsetGRU,
  title   = {Multitask Recognition of Underwater Engine Types and Operating States via an Attention-Based GRU with Sub-Spectral Decomposition},
  author  = {Guedes, Luis P. A. and Guedes, Pedro. H.},
  year    = {2025},
  journal = {IFAC World Congress / arXiv Preprint}
}
```

## Licença
MIT License. Veja `LICENSE`.

## Contato
Luis Paulo Albuquerque Guedes  
Centro de Guerra Acústica e Eletrônica da Marinha (CGAEM)  
E-mail: luis.guedes@marinha.mil.br
