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