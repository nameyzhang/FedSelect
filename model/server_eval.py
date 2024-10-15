import torch
import torch.nn.functional as F
import wandb
import os
import time

from sklearn.metrics import f1_score

from utils.metrics import accuracy


def server_eval(train_loader, model, epoch, args, logger, best_acc, device, epoch_start_time):
    server_loss = 0
    server_acc = []

    current_time = time.time()

    model.eval()

    for batch_idx, (inputs_val, target_val) in enumerate(train_loader):
        with torch.no_grad():
            inputs_val, target_val = inputs_val.to(device), target_val.to(device)
            y_g_hat = model(inputs_val)

            l_g_meta = F.cross_entropy(y_g_hat, target_val)
            server_acc.append(accuracy(y_g_hat.data, target_val.data, topk=(1,))[0])

            server_loss += l_g_meta.item()      # 累积损失

            # f1_score
            # y_g_hat_label = torch.argmax(y_g_hat, dim=1)
            # server_f1_score.append(f1_score(y_g_hat_label.cpu(), target_val.cpu(), average='macro'))

            # 在最后一个批次时记录日志和保存结果
            if (batch_idx + 1) == len(train_loader):
                logger.info(f'\t Eval: \t\t'
                            f'Epoch: {epoch + 1}\t'
                            f'Loss: %.4f\t'
                            f'Prec@1: %.4f' % ((server_loss / (batch_idx + 1)), sum(server_acc) / len(server_acc))
                            )

                with open(f"./logs/{args.dataset_name}/{args.algorithm}/{args.algorithm}.txt", "a") as f:            # 使用 with open 语句打开文件, 确保文件在写入后自动关闭
                    f.write(f'{epoch + 1},'
                            f'{current_time - epoch_start_time},'
                            f'%.4f,'
                            f'%.4f\n' % ((server_loss / (batch_idx + 1)), sum(server_acc) / len(server_acc))
                            )


                wandb.log({
                    'epoch': epoch,
                    'test_avg_loss': (server_loss / (batch_idx + 1)),
                    'test_avg_acc': (sum(server_acc) / len(server_acc)),
                })

    return sum(server_acc) / len(server_acc)
