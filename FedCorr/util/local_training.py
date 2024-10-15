# python version 3.7.1
# -*- coding: utf-8 -*-
import copy
import time

from FedCorr.util.loss import CORESLoss
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from FedCorr.util.loss import FedTwinCRLoss
import numpy as np
from FedCorr.util.optimizer import TwinOptimizer, adjust_learning_rate
from FedCorr.util.optimizer import FedProxOptimizer, f_beta, filter_noisy_data
from torch.autograd import Variable

import torch.nn.functional as F
import wandb

from utils.utils import get_device


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]
        return image, label, self.idxs[item]


class FedCorrLocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.loss_func = CrossEntropyLoss()  # loss function -- cross entropy
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))
        self.device = get_device(args)

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        if self.args.dataset_name == 'clothing1m':
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True, num_workers=3)
        else:
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net, w_g, epoch, mu=1, lr=None):
        net_glob = w_g

        net.train()
        # train and update
        if lr is None:
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=lr)

        epoch_loss = []
        for iter in range(epoch):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            for batch_idx, (images, labels, _) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)

                # print("images: ", images)
                # print("len(images): ", len(images))
                # print("labels: ", labels)
                # print("len(labels)", len(labels))

                if self.args.mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(images, labels, self.args.alpha)
                    net.zero_grad()
                    log_probs, _ = net(inputs)
                    loss = mixup_criterion(self.loss_func, log_probs, targets_a, targets_b, lam)
                else:
                    labels = labels.long()
                    net.zero_grad()
                    # log_probs, _ = net(images)
                    loss = self.loss_func(net(images), labels)
                    # print("len(loss): ", len(loss))

                if self.args.beta > 0:
                    if batch_idx > 0:
                        w_diff = torch.tensor(0.).to(self.device)
                        for w, w_t in zip(net_glob.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        w_diff = torch.sqrt(w_diff)
                        loss += self.args.beta * mu * w_diff

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

                if self.args.dataset_name == 'clothing1m':
                    if batch_idx >= 100:
                        # print(f'use 100 batches as one mini-epoch')
                        break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class FedTwinLocalUpdate:
    def __init__(self, args, dataset, idxs, client_idx):
        self.args = args
        self.loss_func = FedTwinCRLoss()  # loss function -- cross entropy
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))
        self.client_idx = client_idx
        self.device = get_device(args)

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        if self.args.dataset_name == 'clothing1m':
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True, num_workers=3)
        else:
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net_p, net_glob, rounds):
        net_p.train()
        net_glob.train()
        # net_global_param = copy.deepcopy(list(net_glob.parameters()))
        # train and update
        optimizer_theta = TwinOptimizer(net_p.parameters(), lr=self.args.plr, lamda=self.args.lamda)
        optimizer_w = torch.optim.SGD(net_glob.parameters(), lr=self.args.lr)
        epoch_loss = []
        n_bar_k = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            # use/load data from split training set "ldr_train"
            b_bar_p = []
            # lr = args.lr
            adjust_learning_rate(rounds * self.args.local_ep + iter, self.args, optimizer_theta)
            adjust_learning_rate(rounds * self.args.local_ep + iter, self.args, optimizer_w)
            plr = adjust_learning_rate(rounds * self.args.local_ep + iter, self.args, 'plr')
            lr = adjust_learning_rate(rounds * self.args.local_ep + iter, self.args, 'lr')

            for batch_idx, (images, labels, _) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.long()
                for _ in range(self.args.K):
                    log_probs_p, _ = net_p(images)
                    log_probs_g, _ = net_glob(images)
                    # log_probs = net(images)
                    loss_p, loss_g, len_loss_p, len_loss_g = self.loss_func(log_probs_p, log_probs_g,
                                                                                   labels, rounds, iter, self.args)
                    net_p.zero_grad()
                    loss_p.backward()
                    self.persionalized_model_bar, _ = optimizer_theta.step(list(net_glob.parameters()))

                # batch_loss.append(loss.item())
                # update local weight after finding aproximate theta
                for new_param, localweight in zip(self.persionalized_model_bar, net_glob.parameters()):
                    localweight.data = localweight.data - self.args.lamda * lr * (
                            localweight.data - new_param.data)

                net_glob.zero_grad()
                loss_g.backward()
                optimizer_w.step()
                batch_loss.append(loss_g.item())
                b_bar_p.append(len_loss_g)

                if self.args.dataset_name == 'clothing1m':
                    if batch_idx >= 100:
                        # print(f'use 100 batches as one mini-epoch')
                        break

            n_bar_k.append(sum(b_bar_p))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # print("\rRounds {:d} Client {:d} Epoch {:d}: train loss {:.4f}"
            #       .format(rounds, self.client_idx, iter, sum(epoch_loss) / len(epoch_loss)), end='\n', flush=True)
            # if any(math.isnan(loss) for loss in epoch_loss):
            #     print("debug epoch_loss")
        n_bar_k = sum(n_bar_k) / len(n_bar_k)
        return net_p.state_dict(), net_glob.state_dict(), sum(epoch_loss) / len(epoch_loss), n_bar_k


class RFLLocalUpdate:
    def __init__(self, args, dataset=None, user_idx=None, idxs=None):
        self.args = args
        self.dataset = dataset
        self.user_idx = user_idx
        self.idxs = idxs
        self.device = get_device(args)

        self.pseudo_labels = torch.zeros(len(self.dataset), dtype=torch.long, device=self.device)
        self.sim = torch.nn.CosineSimilarity(dim=1)
        self.loss_func = CrossEntropyLoss(reduction="none")
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))
        # self.ldr_train = DataLoader(DatasetSplitRFL(dataset, idxs), batch_size=self.args.batch_size, shuffle=True)
        self.ldr_train_tmp = DataLoader(DatasetSplit(dataset, idxs), batch_size=1, shuffle=True)

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        if self.args.dataset_name == 'clothing1m':
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True, num_workers=3)
        else:
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def RFLloss(self, logit, labels, feature, f_k, mask, small_loss_idxs, new_labels):
        mse = torch.nn.MSELoss(reduction='none')
        ce = torch.nn.CrossEntropyLoss()
        sm = torch.nn.Softmax(dim=1)
        lsm = torch.nn.LogSoftmax(dim=1)

        L_c = ce(logit[small_loss_idxs], new_labels)
        L_cen = torch.sum(
            mask[small_loss_idxs] * torch.sum(mse(feature[small_loss_idxs], f_k[labels[small_loss_idxs]]), 1))
        L_e = -torch.mean(torch.sum(sm(logit[small_loss_idxs]) * lsm(logit[small_loss_idxs]), dim=1))

        lambda_e = self.args.lambda_e
        lambda_cen = self.args.lambda_cen
        if self.args.g_epoch < self.args.T_pl:
            lambda_cen = (self.args.lambda_cen * self.args.g_epoch) / self.args.T_pl

        return L_c + (lambda_cen * L_cen) + (lambda_e * L_e)

    def get_small_loss_samples(self, y_pred, y_true, forget_rate, args):
        loss = self.loss_func(y_pred, y_true)
        ind_sorted = np.argsort(loss.data.cpu()).to(self.device)
        loss_sorted = loss[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]

        return ind_update

    def train(self, net, f_G, client_num):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        epoch_loss = []

        net.eval()
        f_k = torch.zeros(self.args.num_classes, net.fc1.in_features, device=self.device)
        n_labels = torch.zeros(self.args.num_classes, 1, device=self.device)

        # obtain global-guided pseudo labels y_hat by y_hat_k = C_G(F_G(x_k))
        # initialization of global centroids
        # obtain naive average feature
        with torch.no_grad():
            for batch_idx, (images, labels, idxs) in enumerate(self.ldr_train_tmp):
                images, labels = images.to(self.device), labels.to(self.device)
                logit, feature = net(images)
                self.pseudo_labels[idxs] = torch.argmax(logit)
                if self.args.g_epoch == 0:
                    f_k[labels] += feature
                    n_labels[labels] += 1

        if self.args.g_epoch == 0:
            for i in range(len(n_labels)):
                if n_labels[i] == 0:
                    n_labels[i] = 1
            f_k = torch.div(f_k, n_labels)
        else:
            f_k = f_G

        net.train()
        for iter in range(self.args.local_ep):
            batch_loss = []
            correct_num = 0
            total = 0
            for batch_idx, batch in enumerate(self.ldr_train):
                net.zero_grad()
                images, labels, idx = batch
                images, labels = images.to(self.device), labels.to(self.device)
                logit, feature = net(images)
                feature = feature.detach()
                f_k = f_k.to(self.device)

                small_loss_idxs = self.get_small_loss_samples(logit, labels, self.args.forget_rate, self.args)

                y_k_tilde = torch.zeros(self.args.batch_size, device=self.device)
                mask = torch.zeros(self.args.batch_size, device=self.device)
                for i in small_loss_idxs:
                    y_k_tilde[i] = torch.argmax(self.sim(f_k, torch.reshape(feature[i], (1, net.fc1.in_features))))
                    if y_k_tilde[i] == labels[i]:
                        mask[i] = 1

                # When to use pseudo-labels
                if self.args.g_epoch < self.args.T_pl:
                    for i in small_loss_idxs:
                        self.pseudo_labels[idx[i]] = labels[i]

                # For loss calculating
                new_labels = mask[small_loss_idxs] * labels[small_loss_idxs] + (1 - mask[small_loss_idxs]) * \
                             self.pseudo_labels[idx.to(self.device)[small_loss_idxs.to(self.device)]]
                new_labels = new_labels.type(torch.LongTensor).to(self.device)

                loss = self.RFLloss(logit, labels, feature, f_k, mask, small_loss_idxs, new_labels)

                # weight update by minimizing loss: L_total = L_c + lambda_cen * L_cen + lambda_e * L_e
                loss.backward()
                optimizer.step()

                # obtain loss based average features f_k,j_hat from small loss dataset
                f_kj_hat = torch.zeros(self.args.num_classes, net.fc1.in_features, device=self.device)
                n = torch.zeros(self.args.num_classes, 1, device=self.device)
                for i in small_loss_idxs:
                    f_kj_hat[labels[i]] += feature[i]
                    n[labels[i]] += 1
                for i in range(len(n)):
                    if n[i] == 0:
                        n[i] = 1
                f_kj_hat = torch.div(f_kj_hat, n)

                # update local centroid f_k
                one = torch.ones(self.args.num_classes, 1, device=self.device)
                f_k = (one - self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_k + (
                        self.sim(f_k, f_kj_hat).reshape(self.args.num_classes, 1) ** 2) * f_kj_hat

                batch_loss.append(loss.item())

                if self.args.dataset_name == 'clothing1m':
                    if batch_idx >= 100:
                        # print(f'use 100 batches as one mini-epoch')
                        break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), f_k


class FedAVGLocalUpdate:
    def __init__(self, args, dataset=None, idxs=None):    # 初始化函数; 参数 arg, 完整的训练数据集 dataset_train, 样本索引 idx
        self.args = args       # 将参数 arg 存储在类实例中
        self.loss_func = CrossEntropyLoss()      # loss function -- cross entropy
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))   # 从 dataset_train 中选出 idxs(某个client) 的训练样本; 所有的dataset_train 都是test数据
        self.device = get_device(args)

    # 是把每一个选中的 client 的 dataset_train 分为训练集(train) 和 验证集(validation), 但是 ldr_test 没有用呀?
    def train_test(self, dataset, idxs):   # idxs 就是相对于所有的 dataset 的索引
        # split training set, validation set and test set

        if self.args.dataset_name == 'clothing1m':        # batch_size: local batch size B
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True, num_workers=3)    # 创建训练数据加载器; shuffle=True 表示打乱数据; num_workers=3：使用多线程加载数据
        else:
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True)    # 不使用多线程加载器; 使用 DatasetSplit 的目的是从完整数据集中提取子集，并创建用于训练的 DataLoader

        test = DataLoader(dataset, batch_size=128)    # test 是测试数据加载器，基于整个数据集，batch_size 为 128，用于评估模型的性能

        # 返回训练数据加载器和测试数据加载器
        return train, test


    def update_weights(self, net):
        net.train()          #  将模型设置为训练模式
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)          # 使用SGD(随机梯度下降)优化器对模型参数进行优化, 学习率为 lr
        epoch_loss = []      #  初始化, 用于存储每个 epoch 的平均损失
        for iter in range(self.args.local_ep):        # local_ep: number of local epochs
            batch_loss = []                           # 用于存储每个 batch 的loss
            for batch_idx, (images, labels, _) in enumerate(self.ldr_train):    # 遍历训练数据加载器 self.ldr_train 中的每个batch
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()                    # 清除模型的梯度, 以避免梯度积累
                outputs, _ = net(images)           # 执行向前传播, 计算模型的输出
                loss = self.loss_func(outputs, labels)            # 使用定义的损失函数计算模型输出和真实标签之间的损失
                # print("outputs={}, labels={}".format(outputs, labels))
                # print("loss={}".format(loss))
                loss.backward()                    # 通过方向传播计算梯度
                optimizer.step()                   # 使用优化器更新模型参数
                batch_loss.append(loss.item())     # 将当前 batch 的损失值(标量)添加到 batch_loss 列表中


                # 因为 clothing1m 这个数据集可能非常大
                if self.args.dataset_name == 'clothing1m':
                    if batch_idx >= 100:
                        # print(f'use 100 batches as one mini-epoch')
                        break

                # print("batch_loss={}".format(batch_loss))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))     # 计算并存储当前 epoch 的平均损失
            # print("epoch_loss={}".format(epoch_loss))

        # state_dict() 模型的权重;
        # sum(epoch_loss) / len(epoch_loss): 所有 epoch 的平均损失
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class FedProxLocalUpdate:
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = CrossEntropyLoss()  # loss function -- cross entropy
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))
        self.device = get_device(args)

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        if self.args.dataset_name == 'clothing1m':
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True, num_workers=3)
        else:
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net):
        old_net = copy.deepcopy(net)
        net.train()         # 将模型设置为训练模式
        optimizer = FedProxOptimizer(net.parameters(), lr=self.args.lr, lamda=self.args.mu)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, _) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                net.zero_grad()
                outputs, _ = net(images)
                loss = self.loss_func(outputs, labels)
                # print("outputs={}, labels={}".format(outputs, labels))
                # print("loss={}".format(loss))
                loss.backward()
                optimizer.step(list(old_net.parameters()))
                batch_loss.append(loss.item())

                if self.args.dataset_name == 'clothing1m':
                    if batch_idx >= 100:
                        # print(f'use 100 batches as one mini-epoch')
                        break

                # print("batch_loss={}".format(batch_loss))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # print("epoch_loss={}".format(epoch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def get_local_update_objects(args, dataset_train, dict_users=None, net_glob=None):
    local_update_objects = []
    for idx in range(args.num_users):
        local_update_args = dict(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
        )
        local_update_objects.append(RFLLocalUpdate(**local_update_args))

    return local_update_objects


def globaltest(net, dataset, args):     #  神经网络模型, 测试集, 各种配置参数的对象
    device = get_device(args)

    loss_batch_total = 0
    server_loss = 0

    net.eval()     # 将模型设置为评估模式, 关闭 dropout 和 batch normalization, 以确保评估的一致性
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=False)    # 创建一个数据加载器 data_loader 用于从 dataset 中按批次加载数据, shuffle=false 表示不打乱数据顺序
    with torch.no_grad():        # 关闭梯度计算
        correct = 0              # 初始化正确预测的样本计数器
        total = 0                # 初始化总样本计数器
        for images, labels in data_loader:     # 遍历数据加载器中的每个批次 (100张)
            images = images.to(device)
            labels = labels.to(device)
                                               #  在深度学习模型中, 特别是一些复杂的模型 (例如, 包含多个输出的模型), 前向传到过程中可能会返回多个值, 这些值可以包括: 模型的主输出 (例如, 分类分数或回归值), 其他辅助信息 (例如, 中间层的激活值、注意力权重等).
                                               #  获取主输出 (outputs), _ 一个占位符, 用于忽略第二个返回值 (这个返回值可以是模型的中间层输出、注意力权重等辅助信息, 但在这个上下文中不需要使用);
            # outputs, _ = net(images)           #  输出是一个二维张量 [batch_size, num_classes], 其中每一行对应一个样本的输出, 每列对应一个类别的分数 (通常是逻辑回归或概率); 模型的预测类别是分数最高的那个类别.
            # outputs = net(images)


                                                            #  outputs：这是模型的输出张量，形状通常为 [batch_size, num_classes]。每行包含一个样本的所有类别分数
                                                            #  outputs.data：访问 outputs 的数据部分 (取出张量的数据)
                                                            #  torch.max(outputs.data, 1): 在维度 1 上（即每行）找到最大值
            # _, predicted = torch.max(outputs.data, 1)       #  torch.max 返回两个值: 最大值的张量, 最大值所在位置的索引张量; 使用下划线 _ 来忽略最大值的张量, 只保留最大值所在位置的索引张量 predicted (这些索引就是模型预测的类别).
            _, predicted = torch.max(net(images).data, 1)       #  torch.max 返回两个值: 最大值的张量, 最大值所在位置的索引张量; 使用下划线 _ 来忽略最大值的张量, 只保留最大值所在位置的索引张量 predicted (这些索引就是模型预测的类别).
            # print("predicted: ", predicted)

            total += labels.size(0)                         #  累计样本总数
            correct += (predicted == labels).sum().item()   #  累积正确预测的样本数

            # l_g_meta = F.cross_entropy(predicted, labels)
            # server_loss += l_g_meta.item()  # 累积损失

    acc = (correct / total) * 100      #  计算准确率, 正确预测的样本数量/总样本数量, 再乘以100转换为百分比
    # print("acc: ", acc)

    return acc


def globaltest_final(net, dataset, args, logger, rnd, strat_time):     #  神经网络模型, 测试集, 各种配置参数的对象
    device = get_device(args)

    current_time = time.time()


    loss_batch_total = 0
    server_loss = 0

    net.eval()     # 将模型设置为评估模式, 关闭 dropout 和 batch normalization, 以确保评估的一致性
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=100, shuffle=False)    # 创建一个数据加载器 data_loader 用于从 dataset 中按批次加载数据, shuffle=false 表示不打乱数据顺序
    with torch.no_grad():        # 关闭梯度计算
        correct = 0              # 初始化正确预测的样本计数器
        total = 0                # 初始化总样本计数器
        for images, labels in data_loader:     # 遍历数据加载器中的每个批次 (100张)
            images = images.to(device)
            labels = labels.to(device)
                                               #  在深度学习模型中, 特别是一些复杂的模型 (例如, 包含多个输出的模型), 前向传到过程中可能会返回多个值, 这些值可以包括: 模型的主输出 (例如, 分类分数或回归值), 其他辅助信息 (例如, 中间层的激活值、注意力权重等).
                                               #  获取主输出 (outputs), _ 一个占位符, 用于忽略第二个返回值 (这个返回值可以是模型的中间层输出、注意力权重等辅助信息, 但在这个上下文中不需要使用);
            # outputs, _ = net(images)           #  输出是一个二维张量 [batch_size, num_classes], 其中每一行对应一个样本的输出, 每列对应一个类别的分数 (通常是逻辑回归或概率); 模型的预测类别是分数最高的那个类别.
            # outputs = net(images)


                                                            #  outputs：这是模型的输出张量，形状通常为 [batch_size, num_classes]。每行包含一个样本的所有类别分数
                                                            #  outputs.data：访问 outputs 的数据部分 (取出张量的数据)
                                                            #  torch.max(outputs.data, 1): 在维度 1 上（即每行）找到最大值
            # _, predicted = torch.max(outputs.data, 1)       #  torch.max 返回两个值: 最大值的张量, 最大值所在位置的索引张量; 使用下划线 _ 来忽略最大值的张量, 只保留最大值所在位置的索引张量 predicted (这些索引就是模型预测的类别).
            _, predicted = torch.max(net(images).data, 1)       #  torch.max 返回两个值: 最大值的张量, 最大值所在位置的索引张量; 使用下划线 _ 来忽略最大值的张量, 只保留最大值所在位置的索引张量 predicted (这些索引就是模型预测的类别).


            total += labels.size(0)                         #  累计样本总数
            correct += (predicted == labels).sum().item()   #  累积正确预测的样本数

            l_g_meta = F.cross_entropy(net(images), labels)
            server_loss += l_g_meta.item()  # 累积损失

    acc = (correct / total) * 100      #  计算准确率, 正确预测的样本数量/总样本数量, 再乘以100转换为百分比

    logger.info(f'\t Eval: \t\t'
                f'Epoch: {rnd + 1}\t'
                f'Loss: %.4f\t'
                f'Prec@1: %.4f' % ((server_loss / len(data_loader)), acc)
                )

    with open(f"./logs/{args.dataset_name}/{args.algorithm}/{args.algorithm}.txt",
              "a") as f:  # 使用 with open 语句打开文件, 确保文件在写入后自动关闭
        f.write(f'{rnd + 1},'
                f'{current_time - strat_time},'
                f'%.4f,'
                f'%.4f\n' % ((server_loss / len(data_loader)), acc)
                )

    wandb.log({
        'epoch': rnd,
        'test_avg_loss': (server_loss / len(data_loader)),
        'test_avg_acc': acc,
    })



    return acc



def personalizedtest(args, p_models, dataset_test):
    pass


class LocalCORESUpdate:
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = CrossEntropyLoss()  # loss function -- cross entropy
        self.cores_loss_func = CORESLoss(reduction='none')
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))
        self.device = get_device(args)

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        if self.args.dataset_name == 'clothing1m':
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True, num_workers=3)
        else:
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net, pnet_dict, rounds):
        pnet = copy.deepcopy(net)
        pnet.load_state_dict(pnet_dict)
        pnet.to(self.device)
        net.train()
        pnet.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        optimizer_p = torch.optim.SGD(pnet.parameters(), lr=self.args.lr)
        epoch_loss = []
        for iter in range(self.args.local_epochs):
            batch_loss = []
            adjust_learning_rate(rounds * self.args.local_epochs + iter, self.args, optimizer_p)
            for batch_idx, (images, labels, _) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)

                # filtered noisy samples
                log_probs_p, _ = pnet(images)
                Beta = f_beta(rounds * self.args.local_epochs + iter, self.args)
                if rounds <= self.args.begin_sel:
                    loss_p_update = self.cores_loss_func(log_probs_p, labels, Beta)
                    ind_p_update = Variable(torch.from_numpy(np.ones(len(loss_p_update)))).bool()
                else:
                    ind_p_update = filter_noisy_data(log_probs_p, labels)
                    loss_p_update = self.cores_loss_func(log_probs_p[ind_p_update], labels[ind_p_update], Beta)
                loss_batch_p = loss_p_update.data.cpu().numpy()
                if len(loss_batch_p) == 0.0:
                    loss_p = self.cores_loss_func(log_probs_p, labels, Beta)
                    loss_p = torch.mean(loss_p) / 100000000
                else:
                    loss_p = torch.sum(loss_p_update) / len(loss_batch_p)
                # p model updates
                pnet.zero_grad()
                loss_p.backward()
                optimizer_p.step()

                # global model updates
                net.zero_grad()
                outputs, _ = net(images)
                loss = self.loss_func(outputs[ind_p_update], labels[ind_p_update])
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                if self.args.dataset_name == 'clothing1m':
                    if batch_idx >= 100:
                        # print(f'use 100 batches as one mini-epoch')
                        break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        pnet_dict.update(pnet.state_dict())
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class GlobalCORESUpdate:
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        # self.loss_func = CrossEntropyLoss()  # loss function -- cross entropy
        self.cores_loss_func = CORESLoss(reduction='none')
        self.ldr_train, self.ldr_test = self.train_test(dataset, list(idxs))
        self.device = get_device(args)

    def train_test(self, dataset, idxs):
        # split training set, validation set and test set
        if self.args.dataset_name == 'clothing1m':
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True, num_workers=3)
        else:
            train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batch_size, shuffle=True)
        test = DataLoader(dataset, batch_size=128)
        return train, test

    def update_weights(self, net, rounds):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        epoch_loss = []
        for iter in range(self.args.local_epochs):
            batch_loss = []
            adjust_learning_rate(rounds * self.args.local_epochs + iter, self.args, optimizer)
            for batch_idx, (images, labels, _) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)

                # filtered noisy samples
                log_probs_g, _ = net(images)
                Beta = f_beta(rounds * self.args.local_epochs + iter, self.args)
                if rounds <= self.args.begin_sel:
                    loss_g_update = self.cores_loss_func(log_probs_g, labels, Beta)
                else:
                    ind_g_update = filter_noisy_data(log_probs_g, labels)
                    loss_g_update = self.cores_loss_func(log_probs_g[ind_g_update], labels[ind_g_update], Beta)
                loss_batch_g = loss_g_update.data.cpu().numpy()
                if len(loss_batch_g) == 0.0:
                    loss_g = self.cores_loss_func(log_probs_g, labels, Beta)
                    loss_g = torch.mean(loss_g) / 100000000
                else:
                    loss_g = torch.sum(loss_g_update) / len(loss_batch_g)

                # loss = self.loss_func(outputs, labels)

                net.zero_grad()
                loss_g.backward()
                optimizer.step()
                batch_loss.append(loss_g.item())

                if self.args.dataset_name == 'clothing1m':
                    if batch_idx >= 100:
                        # print(f'use 100 batches as one mini-epoch')
                        break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)