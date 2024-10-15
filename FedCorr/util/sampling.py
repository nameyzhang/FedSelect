# python version 3.7.1
# -*- coding: utf-8 -*-
import numpy as np


def iid_sampling(n_train, num_users, seed):
    np.random.seed(seed)   # 设置随机数种子, 以确保每次运行时生成的随机数序列相同, 从而使结果可重复
    num_items = int(n_train / num_users)    # 每个用户应分配的样本数
    # dict_users 是一个空字典, 用于存储每个用户的样本索引; all_idxs 是一个列表, 包含从 0 ~ n_train-1 的所有索引, 用于表示所有训练样本的索引
    dict_users, all_idxs = {}, [i for i in range(n_train)]   # initial user and index for whole dataset

    for i in range(num_users): # 依次分配样本给用户
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))    # 从 all_idxs 中随机选择 num_items 个索引, 不允许重复; 并将选择的索引存储在一个集合中 set(), 集合是由不重复元素组成的无序集合。
        all_idxs = list(set(all_idxs) - dict_users[i])    # 从 all_idxs 中移除已经分配给当前用户 i 的索引, 具体做法是将 all_idxs 和 dict_users[i] 都转换为集合, 然后进行集合差运算, 最后将结果转换回列表 (字典);

    return dict_users    # 返回的是 用户分配情况字典{users_id, 说分配的样本索引}




def non_iid_dirichlet_sampling(y_train, num_classes, p, num_users, seed, alpha_dirichlet=100):
    np.random.seed(seed)
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))  # indicate the classes chosen by each client
    n_classes_per_client = np.sum(Phi, axis=1)

    # 确保每个客户端至少选择一个类别
    while np.min(n_classes_per_client) == 0:
        invalid_idx = np.where(n_classes_per_client == 0)[0]
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)

    # 确保每个类别至少被一个客户端选择
    for class_idx in range(num_classes):
        if np.sum(Phi[:, class_idx]) == 0:  # 如果某个类别没有被任何客户端选择
            client_idx = np.random.choice(range(num_users))  # 随机选择一个客户端
            Phi[client_idx, class_idx] = 1  # 将该类别分配给该客户端

    Psi = [list(np.where(Phi[:, j] == 1)[0]) for j in range(num_classes)]  # indicate the clients that choose each class
    num_clients_per_class = np.array([len(x) for x in Psi])
    dict_users = {}

    for class_i in range(num_classes):
        all_idxs = np.where(y_train == class_i)[0]
        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])

        # 如果 Psi[class_i] 为空列表，则跳过
        if len(Psi[class_i]) == 0:
            continue

        assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())

        for client_k in Psi[class_i]:
            if client_k in dict_users:
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
            else:
                dict_users[client_k] = set(all_idxs[(assignment == client_k)])

    return dict_users

# def non_iid_dirichlet_sampling(y_train, num_classes, p, num_users, seed, alpha_dirichlet=100):
#     np.random.seed(seed)
#     Phi = np.random.binomial(1, p, size=(num_users, num_classes))  # indicate the classes chosen by each client
#     n_classes_per_client = np.sum(Phi, axis=1)
#     while np.min(n_classes_per_client) == 0:
#         invalid_idx = np.where(n_classes_per_client==0)[0]
#         Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
#         n_classes_per_client = np.sum(Phi, axis=1)
#     Psi = [list(np.where(Phi[:, j]==1)[0]) for j in range(num_classes)]   # indicate the clients that choose each class
#     num_clients_per_class = np.array([len(x) for x in Psi])
#     dict_users = {}
#     for class_i in range(num_classes):
#         all_idxs = np.where(y_train==class_i)[0]
#         p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])
#         assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())
#
#         for client_k in Psi[class_i]:
#             if client_k in dict_users:
#                 dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
#             else:
#                 dict_users[client_k] = set(all_idxs[(assignment == client_k)])
#     return dict_users
