# Reconhecimento Multitarefa de Motores Subaquáticos (GRU + Atenção)

Este repositório contém o código-base e a estrutura de diretórios para experimentos de reconhecimento multitarefa de **tipo de motor** e **estado operacional** a partir de **log-mel espectrogramas**, com **decomposição subespectral** e **GRU com atenção**.

## Estrutura
```
├─ data/            # Pré-processamento e split do Wolfset
├─ models/          # GRU, GRU-subspec, BiLSTM, Transformer, ResNet-50
├─ experiments/     # Configurações e logs (M1–M7)
├─ utils/           # Métricas, plots, loaders, etc.
├─ notebooks/       # Exemplos reprodutíveis (Jupyter/Colab)
├─ figures/         # Diagramas e resultados
└─ main.py          # Treino e avaliação principais
```

### Requisitos (sugestão)
- Python 3.10+
- PyTorch >= 2.2
- librosa, soundfile, numpy, pandas, scikit-learn, matplotlib, tqdm, pyyaml

### Uso rápido
1. Ajuste caminhos de dados no arquivo `experiments/config.yaml` ou crie um arquivo próprio.
2. Rode o script principal:
   ```bash
   python main.py --config experiments/config.yaml
   ```

> Observação: substitua os **placeholders** pelos caminhos reais do conjunto Wolfset e ajuste hiperparâmetros conforme necessário.
