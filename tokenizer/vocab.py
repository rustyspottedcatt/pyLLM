from collections import Counter
import re

def build_vocab(corpus, vocab_size=10000, special_tokens=["<PAD>", "<UNK>", "<EOS>"], include_punctuation=True):
    tokens = re.findall(r"\w+|[^\w\s]", corpus)
    freqs = Counter(tokens)
    most_common_tokens = [token for token, _ in freqs.most_common(vocab_size)]
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for token in most_common_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    if include_punctuation:
        punctuation = [".", ",", "!", "?", ":", ";"]
        for punct in punctuation:
            if punct not in vocab:
                vocab[punct] = len(vocab)
    return vocab
