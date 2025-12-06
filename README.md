# Simple Text Inference Model

A minimal character-level bigram language model using PyTorch for GPU acceleration.

## Files

- `model.py` - BigramModel class definition
- `train.py` - Training script
- `generate.py` - Interactive text generation
- `data.txt` - Training data (Shakespeare excerpt)
- `test_generate.py` - Quick generation test

## Setup

Requires PyTorch with CUDA support (or CPU):
```bash
pip install torch
```

## Usage

### 1. Train the model
```bash
python train.py
```
This creates `bigram_model.pth` with the trained model.

### 2. Generate text (interactive)
```bash
python generate.py
```

Commands in interactive mode:
- Type a prompt to generate text
- `temp:X` - Change temperature (e.g., `temp:0.8`)
- `len:X` - Change generation length (e.g., `len:1000`)
- `quit` - Exit

### 3. Quick test
```bash
python test_generate.py
```

## How It Works

**Architecture**: Simple block-based model
- Character embedding (64 dimensions)
- Single linear layer for prediction
- Context window of 8 characters
- ~6.8K parameters

**Training**: 
- Character-level encoding
- Cross-entropy loss
- AdamW optimizer
- 5000 iterations on Shakespeare text

**Inference**:
- Temperature-based sampling
- GPU-accelerated generation
- Generates one character at a time based on 8-character context window

## Customization

Edit hyperparameters in `train.py`:
- `BATCH_SIZE` - Training batch size
- `BLOCK_SIZE` - Context window size
- `EMBEDDING_DIM` - Embedding dimensions
- `LEARNING_RATE` - Optimizer learning rate
- `MAX_ITERS` - Training iterations

Replace `data.txt` with your own text data.
