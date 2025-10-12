import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Head(nn.Module):
    """Single attention head without caching."""
    def __init__(self, head_size, embed_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # Compute attention scores
        wei = q @ k.transpose(1, 2) / math.sqrt(self.head_size)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Apply attention to values
        out = wei @ v
        return out, wei

class CachedHead(nn.Module):
    """Single attention head with KV-caching."""
    def __init__(self, head_size, embed_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size
        self.block_size = block_size
        self.k_cache = None
        self.v_cache = None
        self.cache_index = 0

    def reset_cache(self):
        """Reset the KV cache."""
        self.k_cache = None
        self.v_cache = None
        self.cache_index = 0

    def forward(self, x, caching=False):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if not caching:
            # Standard attention without caching
            wei = q @ k.transpose(1, 2) / math.sqrt(self.head_size)
            tril = torch.tril(torch.ones(T, T, device=x.device))
            wei = wei.masked_fill(tril == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out, wei

        # Cached path - store and reuse K,V
        if self.k_cache is None:
            self.k_cache = torch.zeros(B, self.block_size, self.head_size, device=x.device)
            self.v_cache = torch.zeros(B, self.block_size, self.head_size, device=x.device)
            self.cache_index = 0

        # Update cache
        if self.cache_index + T <= self.block_size:
            self.k_cache[:, self.cache_index:self.cache_index+T, :] = k
            self.v_cache[:, self.cache_index:self.cache_index+T, :] = v
        else:
            # Rolling cache when full
            shift = self.cache_index + T - self.block_size
            self.k_cache[:, :-shift, :] = self.k_cache[:, shift:, :].clone()
            self.v_cache[:, :-shift, :] = self.v_cache[:, shift:, :].clone()
            self.k_cache[:, -T:, :] = k
            self.v_cache[:, -T:, :] = v

        self.cache_index = min(self.cache_index + T, self.block_size)

        # Compute attention with cached keys/values
        wei = q @ self.k_cache.transpose(1, 2) / math.sqrt(self.head_size)
        wei = F.softmax(wei[:, :, :self.cache_index], dim=-1)
        wei = self.dropout(wei)
        out = wei @ self.v_cache[:, :self.cache_index, :]
        return out, wei

class MultiHeadAttention(nn.Module):
    """Multi-head attention without caching."""
    def __init__(self, num_heads, embed_size, block_size, dropout):
        super().__init__()
        head_size = embed_size // num_heads
        self.heads = nn.ModuleList([
            Head(head_size, embed_size, block_size, dropout) 
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = []
        attention_weights = []
        for head in self.heads:
            out, wei = head(x)
            head_outputs.append(out)
            attention_weights.append(wei)
        
        x = torch.cat(head_outputs, dim=-1)
        x = self.dropout(self.proj(x))
        return x, attention_weights

class CachedMultiHeadAttention(nn.Module):
    """Multi-head attention with KV-caching."""
    def __init__(self, num_heads, embed_size, block_size, dropout):
        super().__init__()
        head_size = embed_size // num_heads
        self.heads = nn.ModuleList([
            CachedHead(head_size, embed_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def reset_cache(self):
        """Reset cache for all heads."""
        for head in self.heads:
            head.reset_cache()

    def forward(self, x, caching=False):
        head_outputs = []
        attention_weights = []
        for head in self.heads:
            out, wei = head(x, caching=caching)
            head_outputs.append(out)
            attention_weights.append(wei)
        
        x = torch.cat(head_outputs, dim=-1)
        x = self.dropout(self.proj(x))
        return x, attention_weights

class FeedForward(nn.Module):
    """Feed-forward network."""
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer block without caching."""
    def __init__(self, embed_size, num_heads, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, embed_size, block_size, dropout)
        self.ff = FeedForward(embed_size, dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attn_out, weights = self.sa(self.ln1(x))
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, weights

class CachedTransformerBlock(nn.Module):
    """Transformer block with KV-caching."""
    def __init__(self, embed_size, num_heads, block_size, dropout):
        super().__init__()
        self.sa = CachedMultiHeadAttention(num_heads, embed_size, block_size, dropout)
        self.ff = FeedForward(embed_size, dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def reset_cache(self):
        """Reset the attention cache."""
        self.sa.reset_cache()

    def forward(self, x, caching=False):
        attn_out, weights = self.sa(self.ln1(x), caching=caching)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, weights

class BaselineModel(nn.Module):
    """Transformer model without KV-cache."""
    def __init__(self, vocab_size, block_size, embed_size, num_heads, num_layers, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(block_size, embed_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        
        # Token and position embeddings
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        
        # Apply transformer blocks
        all_weights = []
        for block in self.blocks:
            x, weights = block(x)
            all_weights.append(weights)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, all_weights

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Generate tokens without caching (recomputes entire sequence each step)."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

class KVCacheModel(nn.Module):
    """Transformer model with KV-cache."""
    def __init__(self, vocab_size, block_size, embed_size, num_heads, num_layers, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(block_size, embed_size)
        self.blocks = nn.ModuleList([
            CachedTransformerBlock(embed_size, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)
        self.block_size = block_size

    def reset_cache(self):
        """Reset cache for all blocks."""
        for block in self.blocks:
            block.reset_cache()

    def forward(self, idx, caching=False):
        B, T = idx.shape
        
        # Token and position embeddings
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        
        # Apply transformer blocks
        all_weights = []
        for block in self.blocks:
            x, weights = block(x, caching=caching)
            all_weights.append(weights)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, all_weights

    @torch.no_grad()
    def generate_cached(self, idx, max_new_tokens):
        """Generate tokens with KV-caching (only processes new token each step)."""
        self.eval()
        self.reset_cache()
        
        for _ in range(max_new_tokens):
            # Only process the last token (caching handles the rest)
            x = idx[:, -1:]
            
            # Get predictions with caching
            logits, _ = self(x, caching=True)
            logits = logits[:, -1, :]
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat([idx, next_idx], dim=1)
        return idx
