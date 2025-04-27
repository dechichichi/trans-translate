from model.transformer import Transformer
import torch
import data.train.base_data as bd
from utils.dataset import Dataset
import os

save_folder = 'weights'
save_file = "base_model.pkl"
step = 1000
total_loss = -1.
src_sequence_size = 8
tgt_sequence_size = 8

if __name__ == "__main__":
    # 数据初始化
    dataset = Dataset(bd.en_dict, bd.cn_dict, bd.sentence_pair_demo, src_sequence_size, tgt_sequence_size)
    # 模型初始化
    model=Transformer(src_vocab_size=len(bd.en_dict),
                      tgt_vocab_size=len(bd.cn_dict),
                      word_emb_dim=8,
                      tgt_sequence_size=8)
    loss_f = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    # 循环开始
    for i in range(step):
        # 准备输入和目标数据
        src, tgt_in, tgt_out, _, _ = dataset.get_batch(batch_size=1)
        # 前向传播
        output = model(src, tgt_in)
        # 计算损失
        loss = loss_f(torch.log(output), tgt_out)
        if total_loss < 0:
            total_loss = loss.detach().numpy()
        else:
            total_loss = total_loss * 0.95 + loss.detach().numpy() * 0.05
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
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(save_folder, save_file)
    torch.save(model, save_path)
    print(f"finished train ! model saved file: {save_path}")