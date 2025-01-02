from tokenizer import Tokenizer, build_vocab
from model import Transformer, train_model
from datasets import load_dataset
import numpy as np

print("Step 1: Loading Dataset...")
dataset = load_dataset("wikitext", "wikitext-103-v1")
train_texts = dataset['train']['text']
print(f"Dataset loaded. Number of samples: {len(train_texts)}")

print("\nStep 2: Preprocessing Text...")
subset_size = 1000
train_texts = train_texts[:subset_size]
corpus = " ".join([text for text in train_texts if text.strip()])
print(f"Corpus length (characters): {len(corpus)}")

print("\nStep 3: Building Vocabulary and Initializing Tokenizer...")
vocab = build_vocab(corpus, vocab_size=5000)
print(f"Vocabulary size: {len(vocab)}")
tokenizer = Tokenizer(vocab)

print("\nStep 4: Preparing Training Data...")
sentences = corpus.split(". ")
print(f"Number of sentences: {len(sentences)}")

data = []
for sentence in sentences:
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.encode(tokens)
    if len(token_ids) > 1:
        data.append((np.array(token_ids[:-1]), np.array(token_ids[1:])))
print(f"Number of training samples: {len(data)}")

print("\nStep 5: Defining Model Parameters...")
vocab_size = len(vocab)
max_len = max(len(pair[0]) for pair in data) + 1
print(f"Maximum sequence length: {max_len}")

embed_dim = 512
num_heads = 16
num_layers = 8
hidden_dim = 512

print(f"Model parameters set. "
      f"Vocab size: {vocab_size}, Embed Dim: {embed_dim}, "
      f"Num Heads: {num_heads}, Num Layers: {num_layers}, Hidden Dim: {hidden_dim}")

print("\nStep 6: Initializing Transformer...")
model = Transformer(vocab_size, embed_dim, max_len, num_heads, num_layers, hidden_dim)
print("Transformer initialized.")

print("\nStep 7: Training the Model...")
epochs = 50
learning_rate = 0.001
train_model(model, data, vocab_size, epochs, learning_rate, debug=False)
print("Training completed.")

print("\nStep 8: Performing Inference...")
prompt = "This is"
prompt_tokens = tokenizer.tokenize(prompt)
prompt_ids = tokenizer.encode(prompt_tokens)
print(f"Prompt token IDs: {prompt_ids}")

generated_ids = model.generate(prompt_ids, max_len=20, tokenizer=tokenizer, temperature=1.0)
generated_text = tokenizer.decode(generated_ids)
print("Generated Text:", generated_text)
