import torch
from model import BlockModel
import sys

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

def generate_text(prompt="", max_new_tokens=500, temperature=1.0):
    """
    Generate text from a prompt.
    
    Args:
        prompt: Starting text (empty string starts from scratch)
        max_new_tokens: Number of characters to generate
        temperature: Controls randomness (0.5=safe, 1.0=balanced, 1.5=creative)
    
    Returns:
        Generated text string
    """
    # Encode the prompt
    if prompt:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    else:
        # Start with a random character
        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature)
    
    # Decode and return
    return decode(generated[0].tolist())

if __name__ == "__main__":
    # Interactive mode
    print("\n" + "="*60)
    print("Text Generation (Block-based Model)")
    print("="*60)
    print("Commands:")
    print("  - Type a prompt and press Enter to generate")
    print("  - Type 'temp:X' to change temperature (e.g., 'temp:0.8')")
    print("  - Type 'len:X' to change length (e.g., 'len:1000')")
    print("  - Type 'quit' to exit")
    print("="*60)
    
    temperature = 1.0
    max_tokens = 500
    
    while True:
        user_input = input(f"\nPrompt (temp={temperature}, len={max_tokens}): ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.startswith('temp:'):
            try:
                temperature = float(user_input.split(':')[1])
                print(f"Temperature set to {temperature}")
            except:
                print("Invalid temperature. Use format: temp:1.0")
        elif user_input.startswith('len:'):
            try:
                max_tokens = int(user_input.split(':')[1])
                print(f"Generation length set to {max_tokens}")
            except:
                print("Invalid length. Use format: len:500")
        else:
            print("\nGenerating...\n")
            print("-" * 60)
            result = generate_text(user_input, max_tokens, temperature)
            print(result)
            print("-" * 60)
