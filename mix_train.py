import os
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.cuda.amp import autocast, GradScaler
from data_load import load_train_data, load_cn_vocab, load_en_vocab
from utils.dataset import Dataset
from hyperparams import Hyperparams as hp

save_folder = 'weights'
save_file = "mix_model.pkl"
step = 1000
total_loss = -1.
src_sequence_size = 8
tgt_sequence_size = 8
max_norm = 1.0  # 定义梯度裁剪的最大范数

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
    
    model.train()
    loss_f = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 使用自动混合精度训练
    scaler = GradScaler()

    # 循环开始
    for i in range(step):
        # 准备输入和目标数据
        src, tgt_in, tgt_out, _, _ = dataset.get_batch(batch_size=hp.batch_size)
        
        # 嵌入层
        src_emb = src_embedding(src)
        tgt_emb = tgt_embedding(tgt_in)
        
        # 使用 autocast 自动混合精度
        with autocast():
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
        scaler.scale(loss).backward()
        
        # 将优化器中的梯度取消缩放
        scaler.unscale_(optimizer)
        
        # 对模型的梯度进行裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # 执行优化器的一步更新
        scaler.step(optimizer)
        
        # 更新缩放因子
        scaler.update()
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 打印损失
        if (i + 1) % 100 == 0:
            print("step: ", i+1, "loss:", total_loss)
    
    # 保存模型
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(save_folder, save_file)
    torch.save(model.state_dict(), save_path)  # 保存模型的 state_dict
    print(f"Finished training! Model saved to file: {save_path}")