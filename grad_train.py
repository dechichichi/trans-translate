import os
import torch
import torch.nn as nn
from torch.nn import Transformer
from data_load import load_train_data, load_cn_vocab, load_en_vocab
from utils.dataset import Dataset
from hyperparams import Hyperparams as hp

save_folder = 'weights'
save_file = "grad_model.pkl"
steps = 1000
gradient_accumulation_steps = 4  # 定义梯度累积步数
total_loss = -1.
src_sequence_size = 8
tgt_sequence_size = 8

if __name__ == "__main__":
    # 加载训练数据
    X_train, Y_train = load_train_data()
    
    # 初始化数据集
    dataset = Dataset(X_train, Y_train, src_sequence_size, tgt_sequence_size)
    
    # 加载词汇表
    cn2idx, _ = load_cn_vocab()
    en2idx, _ = load_en_vocab()
    
    # 模型初始化
    model = Transformer(
        d_model=8,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        batch_first=True
    )
    
    # 定义源语言和目标语言的嵌入层
    src_embedding = nn.Embedding(len(cn2idx), 8)
    tgt_embedding = nn.Embedding(len(en2idx), 8)
    
    # 定义输出层
    fc_out = nn.Linear(8, len(en2idx))
    
    # 将嵌入层和输出层加入模型
    model.src_embedding = src_embedding
    model.tgt_embedding = tgt_embedding
    model.fc_out = fc_out
    
    model.train()
    loss_f = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 循环开始
    for i in range(steps):
        # 准备输入和目标数据
        src, tgt_in, tgt_out, _, _ = dataset.get_batch(batch_size=hp.batch_size)
        
        # 嵌入层
        src_emb = model.src_embedding(src)
        tgt_emb = model.tgt_embedding(tgt_in)
        
        # 前向传播
        output = model(src_emb, tgt_emb)
        
        # 通过输出层
        output = model.fc_out(output)
        
        # 调整输出形状以匹配损失函数的要求
        output = output.view(-1, output.size(-1))  # [batch_size * seq_length, vocab_size]
        tgt_out = tgt_out.view(-1)  # [batch_size * seq_length]
        
        # 计算损失
        loss = loss_f(output, tgt_out)
        
        # 缩放损失
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 梯度累积
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.zero_grad()
        
        # 打印损失
        if (i + 1) % 100 == 0:
            print("step: ", i + 1, "loss:", loss.item())
    
    # 保存模型
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, save_file)
    torch.save(model, save_path)
    print(f"finished train ! model saved file: {save_path}")