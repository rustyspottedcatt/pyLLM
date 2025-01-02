import re

class Tokenizer:
    def __init__(self, vocab, unk_token="<UNK>", pad_token="<PAD>", eos_token="<EOS>"):
        self.vocab = vocab
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.id_to_token = {id: token for token, id in vocab.items()}

    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text)

    def encode(self, tokens):
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def decode(self, token_ids):
        tokens = [self.id_to_token.get(id, self.unk_token) for id in token_ids]
        return " ".join(tokens)
