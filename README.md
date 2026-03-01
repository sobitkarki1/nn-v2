> **[ARCHIVED]** Intermediate experiment (~6.8K parameters, PyTorch bigram model).
> Current project: **[nn-v4](https://github.com/sobitkarki1/nn-v4)** — 1.5B-parameter GPT transformer.

---

# Simple Text Inference Model

A minimal character-level bigram language model using PyTorch.
Part of a learning progression: **nn-v1 → nn-v2 → nn-v4**.

| Version | Params | Architecture | Notes |
|---------|--------|--------------|-------|
| nn-v1 | ~1.4K | Embedding + MLP (NumPy) | Ultra-lightweight starter |
| **nn-v2** (this) | ~6.8K | Bigram + transformer block | 8-char context window |
| nn-v4 | ~1.45B | GPT decoder, 24 layers | Mixed precision, Flash Attention 2 |

## Usage
```bash
pip install -r requirements.txt
python train.py    # 5000 iterations on Shakespeare
python generate.py # interactive generation
```