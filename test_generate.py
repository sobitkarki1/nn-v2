import torch
from model import BlockModel

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
print("Loading model...")
checkpoint = torch.load('bigram_model.pth', map_location=DEVICE)

vocab_size = checkpoint['vocab_size']
embedding_dim = checkpoint['embedding_dim']
char_to_idx = checkpoint['char_to_idx']
idx_to_char = checkpoint['idx_to_char']

# Initialize model and load weights
model = BlockModel(vocab_size, embedding_dim)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

print(f"Model loaded successfully (device: {DEVICE})")
print(f"Vocabulary size: {vocab_size}")

# Encoding and decoding functions
encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda l: ''.join([idx_to_char[i] for i in l])

# Test generation
print("\n" + "="*60)
print("Test Generation (temp=0.8):")
print("="*60)

prompt = "First Citizen:"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)

with torch.no_grad():
    generated = model.generate(context, max_new_tokens=300, temperature=0.8)

result = decode(generated[0].tolist())
print(result)
print("="*60)
