import numpy as np
from .layers import (
    multi_head_attention,
    feed_forward_network,
    positional_encoding
)

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

class Transformer:
    def __init__(self, vocab_size, embed_dim, max_len, num_heads, num_layers, hidden_dim):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.01
        self.positional_encodings = positional_encoding(max_len, embed_dim)
        self.attention_layers = [
            multi_head_attention for _ in range(num_layers)
        ]
        self.ffn_layers = [
            feed_forward_network(embed_dim, hidden_dim) for _ in range(num_layers)
        ]
        self.projection = np.random.randn(embed_dim, vocab_size) * 0.01
        self.cache = [{} for _ in range(num_layers)]

    def forward(self, token_ids, use_cache=False):
        embedded = self.embeddings[token_ids] + self.positional_encodings[:token_ids.shape[1]]
        x = embedded
        for i in range(self.num_layers):
            if use_cache and 'k' in self.cache[i] and 'v' in self.cache[i]:
                cached_k = self.cache[i]['k']
                cached_v = self.cache[i]['v']
                attention_output, new_k, new_v = self.attention_layers[i](
                    x, x, x, num_heads=self.num_heads, embed_dim=self.embed_dim, 
                    cached_k=cached_k, cached_v=cached_v
                )
                self.cache[i]['k'] = new_k
                self.cache[i]['v'] = new_v
            else:
                attention_output, new_k, new_v = self.attention_layers[i](
                    x, x, x, num_heads=self.num_heads, embed_dim=self.embed_dim
                )
                if use_cache:
                    self.cache[i]['k'] = new_k
                    self.cache[i]['v'] = new_v
            x = attention_output + x
            ff_output = self.ffn_layers[i](x)
            x = ff_output + x
        logits = np.dot(x, self.projection)
        return logits

    def generate(self, prompt_ids, max_len, tokenizer, temperature=1.0, repetition_penalty=1.2):
        if isinstance(prompt_ids, np.ndarray):
            generated = prompt_ids.tolist()
        else:
            generated = prompt_ids.copy()
        
        for _ in range(max_len - len(generated)):
            token_ids = np.array([generated], dtype=np.int32)
            logits = self.forward(token_ids, use_cache=True)
            last_logits = logits[0, -1, :] / temperature
            
            for t in set(generated):
                last_logits[t] /= repetition_penalty
            
            probabilities = softmax(last_logits)
            
            next_token = np.random.choice(len(probabilities), p=probabilities)
            generated.append(next_token)
            
            if next_token == tokenizer.vocab.get("<EOS>", next_token):
                break
            
        return generated

