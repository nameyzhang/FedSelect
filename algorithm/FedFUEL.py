import copy
import random
import time
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.utils.data import WeightedRandomSampler, DataLoader

from model.server_eval import server_eval
from model.client_train import client_train
from model.build_model import build_model
from model.models_meta import MetaSGD, load_VNet

from utils.utils import get_device, normalize_tensor
from utils.MyDataloader import build_dataset, load_corrupt_client_data, load_non_iid_data, load_non_iid_class_data, load_client_weight_data, load_client_reward_data, load_server_data, CustomDataset
from utils.metrics import accuracy

def FedFUEL(args, logger):

    device = get_device(args)
    criterion = nn.CrossEntropyLoss()     # 用 F.cross_entropy

    # data
    # data_train, data_test, num_class = build_dataset(args.dataset_name)

    if args.noniid_ratio == -1 and args.noniid_class_ratio == -1:  # 不存在 NonIID 的情况
        # args, client_num, corruption_type='uniform', corruption_ratio=args.corruption_prob, corrupt_num=args.corrupt_num, imbalanced_factor=args.imbalanced_factor
        client_data_loader = load_corrupt_client_data(args,
                                                      args.client_num,
                                                      corruption_type=args.corruption_type
                                                      )

    elif args.noniid_ratio != -1:  # 是 noniid_ratio 的情况
        client_data_loader = load_non_iid_data(args,
                                               args.client_num,
                                               corruption_type=args.corruption_type
                                               )

    elif args.noniid_class_ratio != -1:  # 是 noniid_class_ratio 的情况
        client_data_loader = load_non_iid_class_data(args,
                                                     args.client_num,
                                                     corruption_type=args.corruption_type
                                                     )

    # load_server_data 加载的是 meta data 和 val data
    server_data_loader = load_server_data(args)

    # model
    client_model_list = [build_model(args) for i in range(0, args.client_num)]  # 创建一个包含 client_num 个客户端模型的列表
    server_model = build_model(args)

    server_meta_model = load_VNet().to(device)  # 其中一个 R 最大的 client 训练的
    client_meta_model = load_VNet().to(device)



    # optimizer
    client_optimizer_list = [torch.optim.SGD(client_model_list[i].parameters(), args.lr, momentum = args.momentum)
                        for i in range(0, args.client_num)]
    server_meta_optimizer =  torch.optim.Adam(server_meta_model.parameters(), 1e-3, weight_decay=1e-4)

    weight_accumulator = {}

    client_select_num = int(args.client_num * args.select_ratio)



    ###  初始化
    # sample_total_loss_pre = []                            # 用来存储每个样本的 loss
    meta_margin_pre = []                                    # 用来存储每个 client 的本轮的 meta-margin, 初始的时候为 0
    weight_list = []                                        # 用于存储每个 client 中所有 sample 的 weight (为每个 client 创建一个列表用来存储每个 sample 的 weight), 初始值不应该为 1 吧, 应该为1/样本数

    # total_list 用来存储每个 client 的 Dictionary 的 R 的值
    # total_value_list = np.zeros(args.client_num, dtype=float)
    total_value_list = torch.zeros(args.client_num)

    print("total_value_list: ", total_value_list)

    sample_total_loss_pre = [None] * args.client_num        # 变量初始化; 确保 sample_total_loss_pre 在每个客户端的循环开始时被正确初始化一个张量, 并在后续计算中使用

    # for early stop
    best_acc = 0
    patience = 10
    epochs_without_improvement = 0

    # 01 Tracking total training time
    total_training_time = 0
    time_to_accuracy = None  # 初始化用于记录达到目标精度所需的时间
    # 连续达到目标精度的次数
    successive_accuracy_count = 0
    required_successive_count = 10

    # 用于记录每个阶段的时间
    total_client_training_time = 0
    total_aggregation_time = 0


    for j in range(args.client_num):
        # sample_total_loss_pre[j] = []             # 用来存储 client j 中所有样本初始的 loss 值
        meta_margin_pre.append(torch.zeros(len(client_data_loader[j]['train'].dataset)))     # 用来存储 client j 对应的所有样本的 margin-score: P, 初始化为 0


        # 第 0 轮 (初始化模型) 每个样本的loss
        sample_total_loss_pre[j] = torch.ones(0).to(device)      # 用来存储每个样本的 loss

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

                # 每个样本的初始化的 loss, 因为 model 是初始化的模型
                sample_total_loss_pre[j] = torch.cat((sample_total_loss_pre[j], loss), -1)


    strat_time = time.time()

    ### training process
    for epoch in range(0, args.epochs):
        logger.info(f"\t\t*Epoch*: {epoch + 1}")

        # 02 开始每轮训练的计时
        epoch_start_time = time.time()


        weight_sum = [0.0 for i in range(args.client_num)]         # 记录每一个 client 的权重 (表示的是所有样本权重的总和)

        print("-------------------------------- step:1 --------------------------------")

        ### step 1: select client
        if bool(args.select_client) is True and epoch >= args.warmup_epochs and epoch != 0:
            for idx in range(args.client_num):
                server_model.eval()          # 将服务器模型和服务器元模型设置为评估模式
                server_meta_model.eval()
                # cal loss_sum
                for (inputs, targets) in client_data_loader[idx]['train']:  # 遍历指定客户端 (idx表示) 的训练数据加载器, 每次迭代返回一批输入数据: inputs 和对应的目标标签: targets
                    with torch.no_grad():  # 不计算梯度, 因为只需要进行前向传播来计算损失, 而不需要进行反向传播来更新权重
                        inputs, targets = inputs.to(device), targets.to(device).long()  # 将 inputs 和 targets 移动到指定的设备上 (通常是 GPU), 以便加速计算; 同时将 target 转换为 long 类型, 这是计算交叉熵损失时需要的整数类型
                        outputs = server_model(inputs)  # 模型的预测结果

                        loss = F.cross_entropy(outputs, targets, reduce=False)  # 计算交叉熵损失; reduce = False 表示返回每个样本的损失值, 而不是将损失值求和或求平均
                        # meta network: w = f(loss, 参数); loss 作为输入, 计算每个损失值对应的权重
                        weight_sum[idx] += sum(server_meta_model(torch.reshape(loss, (len(loss), 1)))).item()  # 计算并累计损失的权重; torch.reshape(loss, (len(loss), 1)) 将 loss 张量重塑为二维张量, 形状为 (len(loss), 1), 即每个样本的损失值占一行, 变成一个列向量; server_meta_model() 将重塑后的损失张量输入到服务器元模型; .item() 将张量转换为 Python 标量, 这个方法用于从包含单个值的张量中提取数值
                        # server_meta_model(torch.reshape(loss, (len(loss), 1))) 返回的是一个张量，包含每个损失值对应的权重; sum() 对这个张量中的所有元素求和, 得到一个标量值, 表示当前批次所有样本的权重之和

            client_weight = torch.ones(args.client_num)  # 存储每个客户端的权重; 初始化张量, 初始值为1

            for k in range(args.client_num):
                client_weight[k] = weight_sum[k] / (len(client_data_loader[k]['train'].dataset))  # 计算第 k 个客户端的平均权重 (每个样本的平均权重); len(client_data_loader[k]['train'].dataset 表示样本数量
            print(f"client_weight: {client_weight}")

            #       WeightedRandomSampler 根据每个客户端的权重随机选择客户端, 权重越高的客户端被选择的概率越大;
            #       转换后的列表 client_select_list 包含了被选中的客户端索引
            client_select_list = list(WeightedRandomSampler(client_weight, client_select_num, replacement=False))  # client_weight 是一个一维张量, 长度等于客户端数量, 每个元素表示对应客户端的权重; replace=False 表示在采样时不进行替换, 即一个客户端一旦被选中, 就不会在本次采样中再次被选中
        else:
            client_select_list = random.sample(range(0, args.client_num), client_select_num)  # 返回随机选择的客户端索引列表

        print("len_client_select_list: ", len(client_select_list))


        print("-------------------------------- step:2 --------------------------------")

        # 03 记录每个客户端的训练时间
        client_start_time = time.time()  # 客户端训练开始时间

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

            client_train_weight = torch.ones(0)    # 存储客户端训练样本的权重

            # total_value_list = torch.ones()   # 用来存储每个 client 中 total value 的值 (top k 个 meta_margin 最大的值的和)


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
                        client_train_weight = torch.cat((client_train_weight, weight.cpu()), -1)


            else:
                client_train_weight = torch.ones(len(client_train_set))


            # if args.gaussian != 0:


            client_data_loader[j]['meta_train'] = load_client_weight_data(args.dataset_name,
                                                                          args.batch_size,
                                                                          args.select_samples_ratio,
                                                                          client_train_weight,
                                                                          j,
                                                                          client_data_loader[j]['train']
                                                                          )

            # 进行本地训练
            client_train(client_data_loader[j]['meta_train'], client_model, epoch, args.local_epochs, j, local_optimizer_model, device, logger)



            # 每个 client 还要计算更新之后的 loss 的变化 (即, P), 上一轮的 loss - 这一轮的 loss (更新之后的)

            #   这一轮的 loss
            client_model.eval()
            sample_total_loss = torch.ones(0)  # 存储每个样本的 loss; tensor 空向量
            print("client_data_loader[j]['train']", len(client_data_loader[j]['train'].dataset))
            for idx, (inputs, targets) in enumerate(client_data_loader[j]['train']):
                with torch.no_grad():
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = client_model(inputs)
                    loss = F.cross_entropy(outputs, targets.long(),
                                           reduction='none')  # loss 是一个标量 (0维张量), 不能直接与其他张量拼接; reduction = 'none' 表示禁用任何缩减操作, 返回每个样本的独立损失值.

                    sample_total_loss = torch.cat((sample_total_loss, loss.cpu()), -1)  # 拼接的时候

            # 根据 loss 的差值计算 scores (meta-margin)
            meta_margin = np.array(sample_total_loss_pre[j].cpu()) - np.array(sample_total_loss.cpu())
            meta_margin_ = args.meta_momentum * meta_margin_pre[j] + (1 - args.meta_momentum) * meta_margin

            sample_total_loss_pre[j] = sample_total_loss
            meta_margin_pre[j] = meta_margin

            # dictionary 是从 training data (client_data_loader[j]['train']) 中选择 top k 个 meta_margin 最大的值; total_value 表示 dictionary中top k 个 meta_margin 最大的值的和
            client_data_loader[j]['reward_data'], total_value = load_client_reward_data(args.batch_size,
                                                                           args.reward_data_size,
                                                                           meta_margin_,
                                                                           client_data_loader[j]['train']
                                                                           )

            total_value_list[j] = total_value

        # 04 计算当前轮次的客户端训练时间
        client_end_time = time.time()  # 客户端训练结束时间
        client_training_time = client_end_time - client_start_time
        total_client_training_time += client_training_time  # 累加每轮次的客户端训练时间

        # 05 聚合开始时间
        aggregation_start_time = time.time()

        ### step3: server aggregation (服务器模型的参数将被更新为所有选定客户端模型参数的平均值)
        for name, params in server_model.state_dict().items():                            # 初始化一个字典, 用于累积模型参数; 遍历 server_model 的所有参数, 将每个参数的名称 name 和对应的张量 params 提取出来
            weight_accumulator[name] = torch.zeros_like(params)
        for j in client_select_list:        # 累积选定客户端的模型参数
            for name, params in client_model_list[j].state_dict().items():                # 返回客户端模型的参数名称和参数值
                weight_accumulator[name] += params
        for name in weight_accumulator.keys():
            weight_accumulator[name] = weight_accumulator[name] / client_select_num       # 平均累积的参数

        server_model.load_state_dict(weight_accumulator)                                  # 使用平均后的参数更新服务器模型的状态字典
        # server_model save to pth file

        print("best_acc", best_acc)

        ### step4: meta model training (即 server_meta_model)
        # 挑选 R 最大的 client (根据 total value 的值)
        #   get train loss: 同步服务器模型与元模型的参数到一个选定的客户端模型和元模型

        print("total_value_list: ", total_value_list)

        # for i in total_value_list:
        #     client_i = total_value_list.index(max(total_value_list))    # 找出 total_value_list 中最大的那个值的索引; 如果出现相同的最大值, 则返回第一个出现最大值的索引

        max_index = torch.argmax(total_value_list)    # 找出 total_value 最大值的 client 索引

        client_i = max_index.item()    # 将张量索引转换为 python 的整数类型

        print("client_i: ", client_i)

        client_meta_model.load_state_dict(server_meta_model.state_dict())  # 将服务器元模型的参数加载到客户端元模型中, 使得客户端元模型与服务器元模型的参数保持一致;
        client_model_list[client_i].load_state_dict(server_model.state_dict())  # 将服务器的参数加载到选择的客户端模型中, 使得该客户端模型与服务器模型的参数保持一致;

        #   train server_meta_model
        meta_model_train(client_data_loader[client_i]['train'], client_data_loader[client_i]['reward_data'], client_model_list[client_i], client_meta_model, epoch, args, server_meta_optimizer)        # client_data_loader[client_i]['reward_data'] 是 meta data

        # 06 计算聚合时间
        aggregation_end_time = time.time()
        aggregation_time = aggregation_end_time - aggregation_start_time
        total_aggregation_time += aggregation_time  # 累加聚合时间

        ### step5: 对 server_model 进行验证, 记录最终的 平均损失和平均准确率 (实验最终结果)
        accuracy_current = server_eval(server_data_loader['val'], server_model, epoch, args, logger, best_acc, device, strat_time)


        if accuracy_current > best_acc:
            best_acc_epoch = epoch
            best_acc = accuracy_current
            epochs_without_improvement = 0

            print("current_best_acc_epoch: ", best_acc_epoch)
            print("current_best_acc: ", best_acc)

            # 先判断有没有这个文件, 没有的话创建
            # if not os.path.exists(f'./saved_model/{args.algorithm}'):
            #     os.makedirs(f'./saved_model/{args.algorithm}')
            # 保存最佳模型
            # torch.save(server_model.state_dict(), f'./saved_model/{args.algorithm}/best_server_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print('Early stopping')
                # break

        # 07 检查是否连续达到目标精度
        if accuracy_current >= args.target_accuracy:
            successive_accuracy_count += 1
            if time_to_accuracy is None:
                time_to_accuracy = total_client_training_time + total_aggregation_time  # 记录第一次达到目标精度的时间
            if successive_accuracy_count >= required_successive_count:
                print(
                    f"连续 {required_successive_count} 个轮次达到目标精度 {args.target_accuracy}，停止训练，总用时为: {time_to_accuracy:.2f}s")
                # break
        else:
            successive_accuracy_count = 0  # 如果没有达到目标精度，则重置计数

        # 结束当前轮次的计时
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_training_time += epoch_time

    # 08
    print("--------------End----------------")
    print("final_best_acc_epoch: ", best_acc_epoch)
    print("final_best_acc: ", best_acc)
    print(f"Total client training time: {total_client_training_time:.2f}s")
    print(f"Total aggregation time: {total_aggregation_time:.2f}s")
    print(f"Total training time: {total_training_time:.2f}s")

    with open(f"./logs/{args.dataset_name}/{args.algorithm}/{args.algorithm}_final.txt", "a") as f:
        f.write(
            f"------------------- Summary ----------------------:\n"
            f"Seed: {args.seed}\n"
            f"Final best accuracy epoch: {best_acc_epoch}\n"
            f"Final best accuracy: {best_acc}\n"
            f"Total client training time: {total_client_training_time:.2f}s\n"
            f"Total aggregation time: {total_aggregation_time:.2f}s\n"
            f"Total training time: {total_training_time:.2f}s\n"
        )





# 挑选 pusher function 最大的 client去训练这个 meta model
def meta_model_train(client_train_loader, reward_data_loader, local_model, client_meta_model, epoch, args, meta_model_optimizer):  # model 即 server_model 服务器模型; vnet 即 server_meta_model 用于元学习的网络 (即 vnet)

    meta_loss = 0  # 初始化元损失
    meta_acc = []  # 初始化元准确率列表

    device = get_device(args)

    client_meta_model.train()  # 将元网络设置为训练模式, client_meta_model 就是加载 server_meta_model 的, 是一个意思

    for batch_idx, (inputs_val, targets_val, _) in enumerate(reward_data_loader):  # 遍历服务器验证数据 (meta data) 加载器
        client_local_model = build_model(args)  # 调用 build_model 函数构建一个新的模型, 这个新模型应该指的是就是全局模型 server_model
        client_local_model.load_state_dict(local_model.state_dict())  # 使用服务器模型 server_model 的参数初始化新模型, 因为 local_model 加载了 server model

        # 选中一个 client 更新 server_model (这里用 meta_model 表示)
        for (inputs, targets) in client_train_loader:  # client_loader = client_data_loader[client_i]['train'], 一个选中作为元模型更新的客户端的数据加载器, 获取输入数据和目标标签; 这不是在 client 端进行的?
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = client_local_model(inputs)

            cost = F.cross_entropy(outputs, targets, reduce=False)  # 计算每个样本的交叉熵损失
            cost_v = torch.reshape(cost, (len(cost), 1))  # 将 cost 重塑为二维张量 cost_v (批次大小, 1)

            v_lambda = client_meta_model(cost_v.data)  # 使用元网络 client_meta_model 计算损失权重
            l_f_meta = torch.sum(cost_v * v_lambda) / len(cost_v)  # 平均加权元损失(标量); cost_v * v_lambda 表示逐元素相乘, 每个样本的损失值乘以其对应的权重, 从而实现加权损失;

            grads = torch.autograd.grad(l_f_meta, client_local_model.parameters(), create_graph=True)  # 计算 l_f_meta 相对于 meta_model 参数的梯度; create_graph=True 表示需要创建计算图, 以便在后续计算中可以再次对这些梯度进行反向传播; 结果 grads 是一个包含 meta_model 参数梯度的元组;
            meta_lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))  # 根据当前训练周期动态调整学学习率, 即在 epoch >= 80 时乘以 0.1, 在epoch >= 100 时再乘以 0.1;

            pseudo_optimizer = MetaSGD(client_local_model, client_local_model.parameters(), lr=meta_lr,
                                       momentum=args.momentum)  # 自定义的优化器类, 用于实现元学习中的伪优化步骤; 返回的 grads 是一个与 client_local_model.parameters() 中参数顺序对应的梯度值列表, 这些梯度值可以用来进行元优化器的梯度更新或其他需要的操作
            pseudo_optimizer.meta_step(grads)  # 更新的是 client_local_model 的参数
            del grads  # 删除 grads 变量以释放xxq内存
            break  # 结束 for 循环, 只处理一个批次的数据;

        # 使用 server_model (这里用 meta_model 表示) 对验证数据 (meta data) 进行前向传播, 计算验证损失和准确率来指导模型参数 (server_meta_model) 的更新
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)  # 将验证数据和目标标签移动到指定的设备
        y_g_hat = client_local_model(inputs_val)
        l_g_meta = F.cross_entropy(y_g_hat, targets_val.long())  # 计算的是当前批次或样本上所有样本的损失之和
        meta_acc.append(accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0])

        meta_model_optimizer.zero_grad()  # 梯度清零; 为了在进行反向传播之前确保累积的梯度不会干扰当前计算
        l_g_meta.backward()  # 对验证损失进行反向传播, 计算模型参数的梯度

        meta_model_optimizer.step()  # 使用服务器优化器更新模型 (server_meta_model) 参数

        meta_loss += l_g_meta.item()  # 将当前批次的验证损失 l_g_meta 累加到 meta_loss 中; l_g_meta.item() 将损失张量转换为标量值，并将其添加到 meta_loss 中;

