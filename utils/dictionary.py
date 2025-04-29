class Dictionary:
    def __init__(self, word2idx, idx2word):
        self.word2idx = word2idx
        self.idx2word = idx2word

    @classmethod
    def load(cls, vocab_file):
        word2idx = {}
        idx2word = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                word = line.strip()
                word2idx[word] = idx
                idx2word[idx] = word
        return cls(word2idx, idx2word)