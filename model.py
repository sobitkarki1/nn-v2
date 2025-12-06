import torch
import torch.nn as nn

class BlockModel(nn.Module):
    """
    Simple block-based language model that predicts the next character
    based on a context window of previous characters.
    """
    
    def __init__(self, vocab_size, embedding_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Character embedding table
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Linear layer to predict next character
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, idx, targets=None):
        """
        Args:
            idx: (batch_size, block_size) tensor of input indices
            targets: (batch_size, block_size) tensor of target indices
        
        Returns:
            logits: (batch_size, block_size, vocab_size)
            loss: scalar if targets provided, else None
        """
        # Get embeddings
        embeddings = self.embedding(idx)  # (B, T, C)
        
        # Get logits
        logits = self.linear(embeddings)  # (B, T, vocab_size)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits_reshaped, targets_reshaped)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Generate new tokens given a context.
        
        Args:
            idx: (batch_size, block_size) tensor of context indices
            max_new_tokens: number of tokens to generate
            temperature: controls randomness (higher = more random)
        
        Returns:
            idx: (batch_size, block_size + max_new_tokens) extended sequence
        """
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = self(idx)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Apply softmax to get probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)
        
        return idx
