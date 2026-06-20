class ByteCodec:
    SPECIALS = ['<BOS>', '<EOS>', '<PAD>', '<UNK>']
    BYTE_VOCAB_SIZE = 256

    def __init__(self):
        self.stoi = {tok: self.BYTE_VOCAB_SIZE + i for i, tok in enumerate(self.SPECIALS)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.vocab_size = self.BYTE_VOCAB_SIZE + len(self.SPECIALS)

        self.BOS_IDX = self.stoi['<BOS>']
        self.EOS_IDX = self.stoi['<EOS>']
        self.PAD_IDX = self.stoi['<PAD>']
        self.UNK_IDX = self.stoi['<UNK>']

    def encode(self, text: str) -> list:
        return list(text.encode('utf-8'))

    def decode(self, ids) -> str:
        chunks, buf = [], bytearray()
        for i in ids:
            if i < self.BYTE_VOCAB_SIZE:
                buf.append(i)
            else:
                if buf:
                    chunks.append(bytes(buf).decode('utf-8', errors='replace'))
                    buf = bytearray()
                chunks.append(self.itos.get(i, ''))
        if buf:
            chunks.append(bytes(buf).decode('utf-8', errors='replace'))
        return ''.join(chunks)