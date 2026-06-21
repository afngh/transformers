import tiktoken

class TokenCodec:
    def __init__(self):
        self.codec = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.codec.n_vocab
        self.EOS_IDX = self.codec.encode_ordinary("<|endoftext|>")[0]

    def encode(self, text :str) -> list:
        return self.codec.encode_ordinary(text)
    
    def decode(self, ids :list) -> str:
        return self.codec.decode(ids)