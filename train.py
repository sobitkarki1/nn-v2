import torch
from model import BlockModel

# Hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 8
EMBEDDING_DIM = 64
LEARNING_RATE = 1e-3
MAX_ITERS = 5000
EVAL_INTERVAL = 500
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
print(f"Characters: {''.join(chars)}")

# Encoding and decoding functions
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda l: ''.join([idx_to_char[i] for i in l])

# Prepare training data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading function
def get_batch(split):
    """Generate a batch of data"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# Evaluation function
@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Initialize model
model = BlockModel(vocab_size, EMBEDDING_DIM)
model = model.to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())/1e3:.2f}K")

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
print("\nStarting training...")
for iter in range(MAX_ITERS):
    # Evaluate loss periodically
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Sample a batch
    xb, yb = get_batch('train')
    
    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nTraining complete!")

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'embedding_dim': EMBEDDING_DIM,
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
}, 'bigram_model.pth')

print("Model saved to 'bigram_model.pth'")

# Generate sample text
print("\n" + "="*50)
print("Sample generation:")
print("="*50)
print("\nNote: Training teaches the model character patterns and relationships.")
print("Without training (random weights), output is pure gibberish like '!?UBFRahnp,SSfmykRj'.")
print("After training, the model learns:")
print("  - Character distributions (common vs rare characters)")
print("  - N-gram patterns (character sequences from context window)")
print("  - Text structure (spaces, capitalization, punctuation)")
print("\nGenerated text:")
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
generated = model.generate(context, max_new_tokens=500, temperature=1.0)
print(decode(generated[0].tolist()))
