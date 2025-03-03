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
            server_acc.append(accuracy(y_g_hat.data, target_val.data, topk=(1,))[0].to(device))

            server_loss += l_g_meta.item()


            if (batch_idx + 1) == len(train_loader):
                elapsed_time_hours = (current_time - epoch_start_time) / 60

                logger.info(f'\t Eval: \t\t'
                            f'Epoch: {epoch + 1}\t'
                            f'Loss: %.4f\t'
                            f'Prec@1: %.4f' % ((server_loss / (batch_idx + 1)), sum(server_acc) / len(server_acc))
                            )

                txt_path = os.path.join(args.main_path, f'{args.dataset_name}.txt')
                with open(txt_path, "a") as f:
                    f.write(f'{epoch + 1},'
                            f'{elapsed_time_hours},'
                            f'%.4f,'
                            f'%.4f\n' % ((server_loss / (batch_idx + 1)), sum(server_acc) / len(server_acc))
                            )


                # if bool(args.is_wandb) is True:
                #     wandb.log({
                #         'epoch': epoch,
                #         'time': elapsed_time_hours,
                #         'test_avg_loss': (server_loss / (batch_idx + 1)),
                #         'test_avg_acc': (sum(server_acc) / len(server_acc))
                #     })



    return sum(server_acc) / len(server_acc)
