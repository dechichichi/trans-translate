import os
import torch
import torch.nn as nn
from torch.nn import Transformer
from data_load import load_train_data, load_cn_vocab, load_en_vocab
from utils.dataset import Dataset
from hyperparams import Hyperparams as hp

save_folder = 'weights'
save_file = "freeze_model.pkl"
step = 1000
total_loss = -1.
src_sequence_size = 8
tgt_sequence_size = 8

def freeze(module):
    """
    Freezes module's parameters.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False

def get_freezed_parameters(module):
    """
    Returns names of freezed parameters of the given module.
    """
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
    return freezed_parameters

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
        d_model=8,  # 假设 word_emb_dim 为 8
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        batch_first=True  # 确保输入数据的形状为 (batch_size, seq_length, embedding_dim)
    )
    
    # 定义源语言和目标语言的嵌入层
    src_embedding = nn.Embedding(len(cn2idx), 8)
    tgt_embedding = nn.Embedding(len(en2idx), 8)
    
    # 定义输出层
    fc_out = nn.Linear(8, len(en2idx))
    
    # 冻结部分参数（可选）
    freeze(src_embedding)
    freeze(tgt_embedding)
    
    # 获取并打印冻结的参数名称
    freezed_parameters = get_freezed_parameters(model)
    print(f"Freezed parameters: {freezed_parameters}")
    
    # 选择需要梯度更新的参数并初始化优化器
    model_parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params=model_parameters, lr=2e-5, weight_decay=0.0)
    
    # 损失函数
    loss_f = torch.nn.NLLLoss()
    
    # 确保保存目录存在
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # 训练循环
    model.train()
    for i in range(step):
        # 准备输入和目标数据
        src, tgt_in, tgt_out, _, _ = dataset.get_batch(batch_size=hp.batch_size)
        
        # 嵌入层
        src_emb = src_embedding(src)
        tgt_emb = tgt_embedding(tgt_in)
        
        # 前向传播
        output = model(src_emb, tgt_emb)
        
        # 通过输出层
        output = fc_out(output)
        
        # 调整输出形状以匹配损失函数的要求
        output = output.view(-1, output.size(-1))  # [batch_size * seq_length, vocab_size]
        tgt_out = tgt_out.view(-1)  # [batch_size * seq_length]
        
        # 计算损失
        loss = loss_f(torch.log(output), tgt_out)
        if total_loss < 0:
            total_loss = loss.detach().item()
        else:
            total_loss = total_loss * 0.95 + loss.detach().item() * 0.05
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 优化器步骤
        optimizer.step()
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 打印损失
        if (i + 1) % 100 == 0:
            print("step: ", i+1, "loss:", total_loss)
    
    # 保存模型
    save_path = os.path.join(save_folder, save_file)
    torch.save(model.state_dict(), save_path)
    print(f"Finished training! Model saved to file: {save_path}")