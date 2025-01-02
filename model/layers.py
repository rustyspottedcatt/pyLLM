import numpy as np

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def embedding_layer(vocab_size, embed_dim):
    return np.random.randn(vocab_size, embed_dim)

def positional_encoding(max_len, embed_dim):
    pe = np.zeros((max_len, embed_dim))
    for pos in range(max_len):
        for i in range(0, embed_dim, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (2 * i / embed_dim)))
            pe[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / embed_dim)))
    return pe

def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = np.dot(query, key.T)
    dk = key.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = softmax(scaled_attention_logits)
    output = np.dot(attention_weights, value)
    return output

def multi_head_attention(query, key, value, num_heads, embed_dim, cached_k=None, cached_v=None):
    head_dim = embed_dim // num_heads
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    
    batch_size, seq_len, _ = query.shape
    
    query = query.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    key = key.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    value = value.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    if cached_k is not None and cached_v is not None:
        key = np.concatenate((cached_k, key), axis=2)
        value = np.concatenate((cached_v, value), axis=2)
    
    scores = np.matmul(query, key.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    attention_weights = softmax(scores)
    attention_output = np.matmul(attention_weights, value)
    
    attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
    
    new_k = key
    new_v = value
    
    return attention_output, new_k, new_v


def feed_forward_network(embed_dim, hidden_dim):
    w1 = np.random.randn(embed_dim, hidden_dim)
    w2 = np.random.randn(hidden_dim, embed_dim)
    return lambda x: np.maximum(0, np.dot(x, w1)).dot(w2)
