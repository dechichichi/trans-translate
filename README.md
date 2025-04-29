本项目是为了记录transformer的几种调优方法



# 普通训练流程

```text
for step, batch in enumerate(loader, 1):
    
    # prepare inputs and targets for the model and loss function respectively.
    
    # forward pass
    outputs = model(inputs)
    
    # computing loss
    loss = loss_fn(outputs, targets)
    
    # backward pass
    loss.backward()
    
    # perform optimization step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    model.zero_grad()
```

## 梯度累加（Gradient Accumulation）

Gradient Accumulation 背后的想法非常简单——模拟更大的批量。有时为了更好地收敛或提高性能，需要使用大批量大小，但是，它通常需要大量内存。这种问题的一种可能的解决方案是使用较小的批大小，但是，一方面，小批大小会导致训练或推理时间增加，另一方面，梯度下降算法对批量大小，并可能导致不稳定的收敛和性能下降。相反，我们可以运行乘法步骤（累积步骤）并累积（计算平均）梯度一定数量的累积步骤，然后当我们有足够的计算梯度时执行优化步骤。

```
steps = len(loader)
for step, batch in enumerate(loader, 1):
    
    # prepare inputs and targets for the model and loss function respectively.
    
    # forward pass
    outputs = model(inputs)
    
    # computing loss
    loss = loss_fn(outputs, targets)
    
    # accumulating gradients over steps
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps
    
    # backward pass
    loss.backward()
    
    # perform optimization step after certain number of accumulating steps and at the end of epoch
    if step & gradient_accumulation_steps == 0 and step == steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        model.zero_grad()
```

## 冻结（freezing）

冻结是通过切换模型某些层中的计算梯度来加速训练和降低内存利用率的有效方法，几乎不会损失最终质量。

深度学习中的一个众所周知的事实是，低层学习输入数据模式，同时顶层学习特定于目标任务的高级特征。当使用某种优化算法（例如 SGD、AdamW 或 RMSprop）执行优化步骤时，低层接收到小的梯度，因此参数几乎保持不变，这称为梯度消失，因此不是计算“无用”梯度和执行这种低梯度参数的优化，有时需要大量的时间和计算能力，我们可以冻结它们。

PyTorch 为切换计算梯度提供了一个舒适的 API。这种行为可以通过 torch.Tensor 的 requires_grad 属性来设置。

```
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
```

```
import torch
from transformers import AutoConfig, AutoModel


# initializing model
model_path = "microsoft/deberta-v3-base"
config = AutoConfig.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, config=config)


# freezing embeddings and first 2 layers of encoder
freeze(model.embeddings)
freeze(model.encoder.layer[:2])

freezed_parameters = get_freezed_parameters(model)
print(f"Freezed parameters: {freezed_parameters}")

# selecting parameters, which requires gradients and initializing optimizer
model_parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())
optimizer = torch.optim.AdamW(params=model_parameters, lr=2e-5, weight_decay=0.0)
```

## 自动混合精度训练（Automatic Mixed Precision）

自动混合精度（AMP）是另一种在不损失最终质量的情况下减少内存消耗和训练时间的非常简单的方法，在 [文章](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1710.03740) 'Mixed Precision Training' 中进行了介绍NVIDIA 和百度研究人员在 2017 年发表的论文。该方法背后的关键思想是使用较低的精度来将模型的梯度和参数保存在内存中，即建议的方法不是使用全精度（例如 float32），而是使用半精度（例如float16) 用于将张量保存在内存中。然而，当以较低的精度计算梯度时，一些值可能太小以至于它们被视为零，这种现象称为[overflow](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Integer_overflow) 。为了防止“溢出”，原论文的作者提出了一种梯度缩放方法。

PyTorch 为使用自动混合精度提供了一个具有必要功能（从降低精度到梯度缩放）的包，称为 [torch.cuda.amp](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/amp.html) 。自动混合精度作为上下文管理器实现，因此可以轻松插入到训练和推理脚本中。

### 普通训练流程

```text
for step, batch in enumerate(loader, 1):
    
    # prepare inputs and targets for the model and loss function respectively.
    
    # forward pass
    outputs = model(inputs)
    
    # computing loss
    loss = loss_fn(outputs, targets)
    
    # backward pass
    loss.backward()
    
    # perform optimization step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    model.zero_grad()
```

### 使用混合精度训练流程

```text
from torch.cuda.amp import autocast, GradScaler


scaler = GradScaler()

for step, batch in enumerate(loader, 1):
    
    # prepare inputs and targets for the model and loss function respectively.

    # forward pass with `autocast` context manager
    with autocast(enabled=True):
        outputs = model(inputs)
    
    # computing loss
    loss = loss_fn(outputs, targets)
    
    # scale gradint and perform backward pass
    scaler.scale(loss).backward()
    
    # before gradient clipping the optimizer parameters must be unscaled.
    scaler.unscale_(optimizer)
    
    # perform optimization step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    scaler.step(optimizer)
    scaler.update()
```