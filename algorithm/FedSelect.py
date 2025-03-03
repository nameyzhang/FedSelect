import copy
import random
import time
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os

from torch import nn
from torch.utils.data import WeightedRandomSampler, DataLoader

from model.server_eval import server_eval
from model.client_train import client_train
from model.build_model import build_model
from model.models_meta import MetaSGD, load_VNet

from utils.utils import get_device, normalize_tensor
from utils.MyDataloader import build_dataset, load_corrupt_client_data, load_non_iid_data, load_non_iid_class_data, load_client_weight_data, load_client_reward_data, load_server_data, CustomDataset
from utils.metrics import accuracy





def FedSelect(args, logger):

    device = get_device(args)
    criterion = nn.CrossEntropyLoss()

    # data
    # data_train, data_test, num_class = build_dataset(args.dataset_name)

    if args.noniid_ratio == -1 and args.noniid_class_ratio == -1:
        client_data_loader = load_corrupt_client_data(args,
                                                      args.client_num,
                                                      corruption_type=args.corruption_type
                                                      )

    elif args.noniid_ratio != -1:
        client_data_loader = load_non_iid_data(args,
                                               args.client_num,
                                               corruption_type=args.corruption_type
                                               )

    elif args.noniid_class_ratio != -1:
        client_data_loader = load_non_iid_class_data(args,
                                                     args.client_num,
                                                     corruption_type=args.corruption_type
                                                     )

    server_data_loader = load_server_data(args)

    # model
    client_model_list = [build_model(args) for i in range(0, args.client_num)]
    server_model = build_model(args)

    server_meta_model = load_VNet().to(device)
    client_meta_model = load_VNet().to(device)



    # optimizer
    client_optimizer_list = [torch.optim.SGD(client_model_list[i].parameters(), args.lr, momentum = args.momentum)
                        for i in range(0, args.client_num)]
    client_meta_optimizer =  torch.optim.Adam(client_meta_model.parameters(), 1e-3, weight_decay=1e-4)

    weight_accumulator = {}

    client_select_num = int(args.client_num * args.select_ratio)



    # sample_total_loss_pre = []
    meta_margin_pre = []
    weight_list = []


    total_value_list = torch.zeros(args.client_num, device=device)

    print("total_value_list: ", total_value_list)

    sample_total_loss_pre = [None] * args.client_num


    # for early stop
    best_acc = 0
    patience = 10
    epochs_without_improvement = 0


    client_weight = torch.ones(args.client_num)

    for j in range(args.client_num):

        meta_margin_pre.append(torch.zeros(len(client_data_loader[j]['train'].dataset), device=device))

        sample_total_loss_pre[j] = torch.ones(0, device=device)

        # get global_model from server
        client_model_list[j].load_state_dict(server_model.state_dict())
        model = client_model_list[j]
        model.eval()

        # print("client_data_loader[j]['train']", len(client_data_loader[j]['train'].dataset))
        for batch_idx, (inputs, targets) in enumerate(client_data_loader[j]['train']):
            with torch.no_grad():
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                loss = F.cross_entropy(outputs, targets.long(), reduction = 'none')

                sample_total_loss_pre[j] = torch.cat((sample_total_loss_pre[j], loss), -1)


    start_time = time.time()

    ''' --- begin training process --- '''
    for epoch in range(0, args.epochs):
        logger.info(f"\t\t*Epoch*: {epoch + 1}")

        weight_sum = torch.zeros(args.client_num, device=device)


        ### step 1: select client
        if bool(args.select_client) is True and epoch >= args.warmup_epochs and epoch != 0:

            client_select_list = list(WeightedRandomSampler(client_weight, client_select_num, replacement=False))
        else:
            client_select_list = random.sample(range(0, args.client_num), client_select_num)

        print("len_client_select_list: ", len(client_select_list))
        print("client_select_list: ", client_select_list)



        ### step2: selected client training
        for j in client_select_list:

            # client_model
            client_model_list[j].load_state_dict(server_model.state_dict())
            client_model = client_model_list[j]
            # client_model_optimizer
            local_optimizer_model = client_optimizer_list[j]

            # client_meta_model
            client_meta_model.load_state_dict(server_meta_model.state_dict())

            client_train_set = client_data_loader[j]['train'].dataset

            client_train_weight = torch.ones(0, device=device)



            if bool(args.select_samples) is True and epoch >= args.warmup_epochs and epoch != 0:
                client_model.eval()
                client_meta_model.eval()

                for index, (data, label) in enumerate(client_data_loader[j]['train']):
                    with torch.no_grad():
                        data, label = data.to(device), label.to(device).long()
                        y_pred = client_model(data)

                        loss = F.cross_entropy(y_pred, label, reduce=False)

                        # loss -> meta_model -> weight

                        weight = client_meta_model(torch.reshape(loss, (len(loss), -1)))
                        weight = torch.reshape(weight, loss.shape)
                        client_train_weight = torch.cat((client_train_weight, weight.to(device)), -1)

                print("len(client_train_weight)", len(client_train_weight))
                print("len(client_data_loader[j]['train'].dataset)", len(client_data_loader[j]['train'].dataset))


            else:
                client_train_weight = torch.ones(len(client_train_set))



            client_weight[j] = client_train_weight.sum() / (len(client_data_loader[j]['train'].dataset))



            client_data_loader[j]['meta_train'] = load_client_weight_data(args.batch_size,
                                                                          args.select_samples_ratio,
                                                                          client_train_weight,
                                                                          client_data_loader[j]['train']
                                                                          )


            client_train(client_data_loader[j]['meta_train'], client_model, epoch, args.local_epochs, j, local_optimizer_model, device, logger)



            client_model.eval()
            sample_total_loss = torch.ones(0)
            for idx, (inputs, targets) in enumerate(client_data_loader[j]['train']):
                with torch.no_grad():
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = client_model(inputs)
                    loss = F.cross_entropy(outputs, targets.long(), reduction='none')

                    sample_total_loss = torch.cat((sample_total_loss, loss.cpu()), -1)

                    # sample_total_loss = normalize_tensor(sample_total_loss)

            meta_margin = sample_total_loss_pre[j].to(device) - sample_total_loss.to(device)
            meta_margin = normalize_tensor(meta_margin)



            meta_margin_ = args.meta_momentum * meta_margin_pre[j].to(device) + (1 - args.meta_momentum) * meta_margin
            meta_margin_ = normalize_tensor(meta_margin_)


            sample_total_loss_pre[j] = sample_total_loss
            meta_margin_pre[j] = meta_margin




            client_data_loader[j]['reward_data'], total_value = load_client_reward_data(args.batch_size,
                                                                           args.reward_data_size,
                                                                           meta_margin_,
                                                                           client_data_loader[j]['train']
                                                                           )

            print("len(client_data_loader[j]['reward_data']): ", len(client_data_loader[j]['reward_data'].dataset))
            print("--")

            total_value_list[j] = total_value

        for name, params in server_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params, device=device)
        for j in client_select_list:
            for name, params in client_model_list[j].state_dict().items():
                weight_accumulator[name] += params
        for name in weight_accumulator.keys():
            weight_accumulator[name] = weight_accumulator[name] / client_select_num

        server_model.load_state_dict(weight_accumulator)


        max_index = torch.argmax(client_weight)

        client_i = max_index.item()


        print("client_i: ", client_i)

        client_meta_model.load_state_dict(server_meta_model.state_dict())
        client_model_list[client_i].load_state_dict(server_model.state_dict())

        meta_model_train(client_data_loader[client_i]['train'], client_data_loader[client_i]['reward_data'], client_model_list[client_i], client_meta_model, epoch, args, client_meta_optimizer)

        server_meta_model.load_state_dict(client_meta_model.state_dict())

        accuracy_current = server_eval(server_data_loader['val'], server_model.to(device), epoch, args, logger, best_acc, device, start_time)

        if accuracy_current > best_acc:
            best_acc_epoch = epoch
            best_acc = accuracy_current
            epochs_without_improvement = 0

            print("current_best_acc_epoch: ", best_acc_epoch)
            print("current_best_acc: ", best_acc)

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print('Early stopping')
                break

        print(f"Epoch {epoch}: accuracy_current = {accuracy_current}, best_acc = {best_acc}, best_acc_epoch = {best_acc_epoch}")


    end_time = time.time()
    computation_time = end_time - start_time 



    print("--------------End----------------")
    print("final_best_acc_epoch: ", best_acc_epoch)
    print("final_best_acc: ", best_acc)
    print("Total computation time (s): ", computation_time)


    final_txt_path = os.path.join(args.main_path, f'{args.dataset_name}_final.txt')
    txt_path = os.path.join(args.main_path, f'{args.dataset_name}.txt')


    with open(final_txt_path, "a") as f:
        f.write(
            f"Seed: {args.seed}\n"
            f"local_epoch: {args.local_epochs}\n"
            f"validation_size: {args.reward_data_size}\n"
            f"Final best accuracy epoch: {best_acc_epoch}\n"
            f"Final best accuracy: {best_acc}\n"
            f"Total computation time (s): {computation_time:.4f}\n"
        )
    with open(txt_path, "a") as f:
        f.write(
            f"Seed:{args.seed}\t"
            f"local_epoch: {args.local_epochs}\n"
            f"Final best accuracy epoch: {best_acc_epoch}\t"
            f"Final best accuracy: {best_acc}\n"
            f"Total computation time (s): {computation_time:.4f}\n"
        )



def meta_model_train(client_train_loader, reward_data_loader, local_model, client_meta_model, epoch, args, meta_model_optimizer):

    meta_loss = 0
    meta_acc = []

    device = get_device(args)

    client_meta_model.train()

    for batch_idx, (inputs_val, targets_val, _) in enumerate(reward_data_loader):

        client_local_model = build_model(args)
        client_local_model.load_state_dict(local_model.state_dict())

        for (inputs, targets) in client_train_loader:
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = client_local_model(inputs)

            cost = F.cross_entropy(outputs, targets, reduce=False)
            cost_v = torch.reshape(cost, (len(cost), 1))

            v_lambda = client_meta_model(cost_v.data)
            l_f_meta = torch.sum(cost_v * v_lambda) / len(cost_v)

            grads = torch.autograd.grad(l_f_meta, client_local_model.parameters(), create_graph=True)
            meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))

            pseudo_optimizer = MetaSGD(client_local_model, client_local_model.parameters(), lr=meta_lr,
                                       momentum=args.momentum)
            pseudo_optimizer.meta_step(grads)
            del grads
            break

        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        y_g_hat = client_local_model(inputs_val)
        l_g_meta = F.cross_entropy(y_g_hat, targets_val.long())

        meta_acc.append(accuracy(y_g_hat.to(device).data, targets_val.to(device).data, topk=(1,))[0])

        meta_model_optimizer.zero_grad()
        l_g_meta.backward()

        meta_model_optimizer.step()

        meta_loss += l_g_meta.item()

