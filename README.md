# pyLLM

[![Maintainer](https://img.shields.io/badge/maintainer-rustyspottedcatt-blue)](https://github.com/YourGitHubUsername)  
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB.svg)](https://www.python.org/)  
[![License](https://img.shields.io/badge/License-MIT-blue)](https://choosealicense.com/licenses/mit/)  

> [!NOTE]  
> pyLLM is an experimental Large Language Model (LLM) framework implemented from scratch using NumPy. Expect bugs.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Tokenization](#tokenization)
  - [Building Vocabulary](#building-vocabulary)
  - [Training](#training)
  - [Inference](#inference)
- [Modules](#modules)
- [Dependencies](#dependencies)
- [License](#license)

---

## Features

- **Custom Tokenizer**  
  Implements word-based tokenization with `<PAD>`, `<UNK>`, and `<EOS>` tokens.
  
- **Transformer Model**  
  Supports multi-head attention, feed-forward networks, and position encoding.

- **Efficient Training Pipeline**  
  Uses `numba`-optimized softmax and cross-entropy loss for speed improvements.

- **Text Generation**  
  Implements token sampling with temperature scaling and repetition penalty.

- **Minimal Dependencies**  
  Uses `numpy` and `numba` for efficient numerical computations.

---

## Installation

### Prerequisites

- Python 3.8 or higher  
- pip package manager  

### Clone the Repository

```sh
git clone https://github.com/rustyspottedcatt/pyLLM.git
cd pyLLM
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

---

## Usage

### Tokenization

```python
from tokenizer import Tokenizer

vocab = {"hello": 0, "world": 1, "<UNK>": 2, "<PAD>": 3, "<EOS>": 4}
tokenizer = Tokenizer(vocab)

text = "hello world!"
tokens = tokenizer.tokenize(text)
print(tokens)  # Output: ['hello', 'world', '!']

token_ids = tokenizer.encode(tokens)
print(token_ids)  # Output: [0, 1, 2]

decoded_text = tokenizer.decode(token_ids)
print(decoded_text)  # Output: "hello world <UNK>"
```

---

### Building Vocabulary

```python
from vocab import build_vocab

corpus = "Hello world! This is a simple LLM implementation."
vocab = build_vocab(corpus, vocab_size=5000)
print(f"Vocabulary Size: {len(vocab)}")
```

---

### Training

```python
from main import train_model, Transformer
import numpy as np

# Define Model Parameters
vocab_size = 5000
embed_dim = 512
max_len = 128
num_heads = 8
num_layers = 6
hidden_dim = 1024

# Initialize Model
model = Transformer(vocab_size, embed_dim, max_len, num_heads, num_layers, hidden_dim)

# Dummy Training Data
data = [(np.array([1, 2, 3]), np.array([2, 3, 4]))]

# Train the Model
train_model(model, data, vocab_size, epochs=10, lr=0.001, debug=True)
```

---

### Inference

```python
from tokenizer import Tokenizer
from transformer import Transformer

# Load Tokenizer and Model
tokenizer = Tokenizer(vocab)
model = Transformer(vocab_size, embed_dim, max_len, num_heads, num_layers, hidden_dim)

# Generate Text
prompt = "Hello world"
prompt_tokens = tokenizer.tokenize(prompt)
prompt_ids = tokenizer.encode(prompt_tokens)

generated_ids = model.generate(prompt_ids, max_len=20, tokenizer=tokenizer, temperature=1.0)
generated_text = tokenizer.decode(generated_ids)
print("Generated Text:", generated_text)
```

---

## Modules

- **Tokenizer (`tokenizer.py`)**  
  - Implements text tokenization, encoding, and decoding.

- **Vocabulary Builder (`vocab.py`)**  
  - Creates a vocabulary from a given corpus.

- **Transformer Model (`transformer.py`)**  
  - Implements a Transformer with multi-head attention and feed-forward networks.

- **Training Pipeline (`training.py`)**  
  - Uses `numba` to optimize softmax and cross-entropy loss calculations.

- **Main Script (`main.py`)**  
  - Loads dataset, preprocesses text, initializes the model, and runs training.

---

## Dependencies

```toml
numpy = "^1.21.0"
numba = "^0.54.0"
datasets = "^2.0.0"
```

---

## License

Distributed under the [MIT License](https://choosealicense.com/licenses/mit/).