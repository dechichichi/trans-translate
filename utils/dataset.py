import torch
import numpy as np

class Dataset:
    def __init__(self, X, Y, src_seq_size, tgt_seq_size):
        self.X = X
        self.Y = Y
        self.src_seq_size = src_seq_size
        self.tgt_seq_size = tgt_seq_size
        self.cn2idx = self.load_vocab("cn")
        self.en2idx = self.load_vocab("en")

    def load_vocab(self, language):
        assert language in ["cn", "en"]
        vocab = [
            line.split()[0]
            for line in open(f"preprocessed/{language}.txt.vocab.tsv", "r", encoding="utf-8")
            .read()
            .splitlines()
        ]
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        return word2idx

    def get_batch(self, batch_size):
        indices = torch.randperm(len(self.X))[:batch_size]
        src_batch = self.X[indices]
        tgt_in_batch = self.Y[indices]
        
        # 将 numpy.ndarray 转换为 torch.Tensor
        src_batch = torch.tensor(src_batch, dtype=torch.long)
        tgt_in_batch = torch.tensor(tgt_in_batch, dtype=torch.long)
        
        # 使用 torch.roll
        tgt_out_batch = torch.roll(tgt_in_batch, shifts=-1, dims=1)
        
        return src_batch, tgt_in_batch, tgt_out_batch, None, None