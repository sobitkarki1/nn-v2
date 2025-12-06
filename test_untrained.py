import torch
from model import BlockModel

# Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8
EMBEDDING_DIM = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

# Load and prepare data
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Loaded {len(text)} characters")

# Create character-to-integer mapping
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

# Encoding and decoding functions
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda l: ''.join([idx_to_char[i] for i in l])

# Initialize UNTRAINED model
model = BlockModel(vocab_size, EMBEDDING_DIM)
model = model.to(DEVICE)
model.eval()

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())/1e3:.2f}K")
print("\nGenerating from UNTRAINED model...")

# Generate sample text
print("\n" + "="*60)
print("Zero-training generation (random weights):")
print("="*60)

# Test with prompt
prompt = "First Citizen:"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

with torch.no_grad():
    generated = model.generate(context, max_new_tokens=300, temperature=1.0)

print(decode(generated[0].tolist()))
print("="*60)
