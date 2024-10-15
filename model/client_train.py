import torch
from torch import nn

from utils.metrics import accuracy


def client_train(train_loader, model, global_epoch, local_epoch, client_index, client_optimizer_model, device, logger):

    epoch_loss = []         # 初始化, 用于存储每个epoch的平均损失
    epoch_acc= []           # 初始化, 用于存储每个epoch的平均精度

    criterion = nn.CrossEntropyLoss().to(device)

    logger.info(f'\t\t Client_idx: {client_index}')

    model = model.to(device)
    model.train()

    # print("运行client_train: len(train_loader)", len(train_loader.dataset))

    # 记录每个样本的loss
    samples_loss = torch.ones(0).to(device)


    for iter in range(local_epoch):
        batch_loss = []           # 用于存储每个 batch 的loss
        train_acc = []            # 创建一个空列表, 用于存储每个 batch 的准确率


        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = model(inputs)

            train_acc.append(accuracy(outputs.data, targets.data, topk=(1,))[0])

            loss = criterion(outputs, targets)

            # 记录每个样本的loss
            #       确保将 loss 转换为一个张量，并添加一个新的维度
            samples_loss = torch.cat((samples_loss, loss.unsqueeze(0)), dim=-1)

            loss_regular = 0.0
            loss = loss + loss_regular

            client_optimizer_model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)     # 对模型参数的梯度进行裁剪, 防止梯度爆炸
            client_optimizer_model.step()                                                # 使用优化器更新模型参数

            batch_loss.append(loss.item())                                        # .item() 将一个 Tensor 变量转换为 python 标量

        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        epoch_acc.append(sum(train_acc) / len(train_acc))

        # logger.info(f'\t\t\t local_epoch: {iter + 1} \t\t'
        #             f'epoch_loss: {epoch_loss[iter]} \t\t'
        #             f'epoch_acc: {epoch_acc[iter].item()}'
        #             )

    logger.info(f'\t\t Average_loss: {sum(epoch_loss) / len(epoch_loss)} \t\t'
                f' Average_acc: {sum(epoch_acc) / len(epoch_acc)}'
                )

    loss = sum(epoch_loss) / len(epoch_loss)

    return model.state_dict(), loss, samples_loss



