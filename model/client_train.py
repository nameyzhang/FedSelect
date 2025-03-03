import torch
from torch import nn

from utils.metrics import accuracy


def client_train(train_loader, model, global_epoch, local_epoch, client_index, client_optimizer_model, device, logger):

    epoch_loss = []
    epoch_acc= []

    criterion = nn.CrossEntropyLoss().to(device)

    logger.info(f'\t\t Client_idx: {client_index}')

    model = model.to(device)
    model.train()


    samples_loss = torch.ones(0).to(device)


    for iter in range(local_epoch):
        batch_loss = []
        train_acc = []


        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = model(inputs)

            train_acc.append(accuracy(outputs.data, targets.data, topk=(1,))[0])

            loss = criterion(outputs, targets)


            samples_loss = torch.cat((samples_loss, loss.unsqueeze(0)), dim=-1)

            loss_regular = 0.0
            loss = loss + loss_regular

            client_optimizer_model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            client_optimizer_model.step()

            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        epoch_acc.append(sum(train_acc) / len(train_acc))


    logger.info(f'\t\t Average_loss: {sum(epoch_loss) / len(epoch_loss)} \t\t'
                f' Average_acc: {sum(epoch_acc) / len(epoch_acc)}'
                )

    loss = sum(epoch_loss) / len(epoch_loss)

    return model.state_dict(), loss, samples_loss



