from torch.nn import Module, Linear, Dropout, LayerNorm, ModuleList
from model.multi_head_attention import MultiheadAttention
from typing import Optional
from torch import Tensor
import torch
import copy

#编码时只对src进行编码
#TransformerEncoder由n个TransformerEncoderLayer组成
#重复n次TransformerEncoderLayer后得到TransformerEncoder的输出
class TransformerEncoderLayer(Module):

    def __init__(self, word_emb_dim, nhead, dim_feedforward=2048, dropout_prob=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(word_emb_dim, nhead, dropout_prob=dropout_prob)

        self.linear1 = Linear(word_emb_dim, dim_feedforward)
        # dropout的作用是防止过拟合
        # 随机丢弃一部分神经元
        self.dropout = Dropout(dropout_prob)
        # 线性变换
        # Linear将输入张量通过一个线性变换映射到一个新的空间
        # 第一个参数是输入纬度 第二个是输出纬度
        self.linear2 = Linear(dim_feedforward, word_emb_dim)
        # 归一化
        # LayerNorm在单个样本的特征维度上进行归一化
        # 参数为归一化的特征纬度
        self.norm1 = LayerNorm(word_emb_dim)
        self.norm2 = LayerNorm(word_emb_dim)
        self.dropout1 = Dropout(dropout_prob)
        self.dropout2 = Dropout(dropout_prob)
        # 在神经网络的某个层中使用 ReLU 激活函数
        self.activation = torch.relu

    def forward(self, src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        :param src: Tensor, shape: [src_sequence_size, batch_size, word_emb_dim]
        :param src_mask: Tensor, shape: [src_sequence_size, src_sequence_size]
        :param src_key_padding_mask: Tensor, shape: [batch_size, src_sequence_size]
        :return: Tensor, shape: [src_sequence_size, batch_size, word_emb_dim]
        """
        # self attention
        src2 = self.self_attn(src, src, src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 两层全连接
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(Module):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm):
        super(TransformerEncoder, self).__init__()
        # 将同一个encoder_layer进行deepcopy n次
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src: Tensor,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        :param src: Tensor, shape: [src_sequence_size, batch_size, word_emb_dim]
        :param mask: Tensor, shape: [src_sequence_size, src_sequence_size]
        :param src_key_padding_mask: Tensor, shape: [batch_size, src_sequence_size]
        :return: Tensor, shape: [src_sequence_size, batch_size, word_emb_dim]
        """
        output = src
        # 串行n个encoder_layer
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        output = self.norm(output)
        return output


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])

