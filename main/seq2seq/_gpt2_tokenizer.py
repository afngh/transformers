import tiktoken

class TokenCodec:
    def __init__(self):
        self.codec = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.codec.n_vocab
        self.EOS_IDX = self.codec.eot_token

    def encode(self, text :str) -> list:
        return self.codec.encode_ordinary(text)
    
    def decode(self, ids: list) -> str:
        filtered = [i for i in ids if i < self.codec.n_vocab]
        return self.codec.decode(filtered)