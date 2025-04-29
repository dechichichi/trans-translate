import os
import torch
import numpy as np
from data_load import load_test_data
from torch.nn import Transformer
from utils.dictionary import Dictionary  # 假设你有一个 Dictionary 类

save_file = "weights/grad_model.pkl"
output_file = "output/mix_translations.txt"  # 输出文件路径
tgt_sequence_size = 8  # 假设目标序列的最大长度为 8
d_model = 8  # 假设嵌入维度为 8

if __name__ == "__main__":
    # 加载测试数据
    X_test, Sources, Targets = load_test_data()
    
    # 加载预训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查是否有可用的 GPU
    transformer = torch.load(save_file, map_location=device, weights_only=False)  # 加载模型到指定设备
    print(f"Loaded transformer from: {save_file}")
    
    # 加载词汇表
    cn_dict = Dictionary.load("preprocessed/cn.txt.vocab.tsv")  # 假设词汇表文件路径
    en_dict = Dictionary.load("preprocessed/en.txt.vocab.tsv")  # 假设词汇表文件路径
    
    # 定义嵌入层
    src_embedding = torch.nn.Embedding(len(cn_dict.word2idx), d_model).to(device)
    tgt_embedding = torch.nn.Embedding(len(en_dict.word2idx), d_model).to(device)

    # 将模型设置为评估模式
    transformer.eval()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 推理过程
    with torch.no_grad():  # 禁用梯度计算
        with open(output_file, 'w', encoding='utf-8') as f:  # 打开输出文件
            for i in range(len(X_test)):
                # 将 NumPy 数组转换为 PyTorch 张量
                src = torch.tensor(X_test[i], dtype=torch.long).unsqueeze(0).to(device)  # 增加批次维度并移动到指定设备
                tgt_in = torch.zeros(1, 1, dtype=torch.long).to(device)  # 初始化目标输入并移动到指定设备
                tgt_out = []
            
                # 通过嵌入层
                src_emb = src_embedding(src)
                tgt_emb = tgt_embedding(tgt_in)
            
                for _ in range(tgt_sequence_size):
                    output = transformer(src_emb, tgt_emb)
                    next_word = output.argmax(dim=-1)[:, -1]  # 获取下一个单词
                    tgt_out.append(next_word.item())
                    tgt_in = torch.cat([tgt_in, next_word.unsqueeze(0)], dim=1)
                    tgt_emb = tgt_embedding(tgt_in)
            
                # 将索引转换为单词
                translated_sentence = ' '.join([en_dict.idx2word[idx] for idx in tgt_out])
            
                # 写入文件
                f.write(f"Source: {Sources[i]}\n")
                f.write(f"Target: {Targets[i]}\n")
                f.write(f"Translated: {translated_sentence}\n")
                f.write("-" * 50 + "\n")
    
    print(f"Translations saved to: {output_file}")