# python version 3.7.1
# -*- coding: utf-8 -*-

import copy
import torch


def FedAvg(w, dict_len):             #  所有本地模型的权重; 每个客户端的样本数量
    w_avg = copy.deepcopy(w[0])      #  对一个客户端的模型权重进行深拷贝, 作为加权和的初始值
    for k in w_avg.keys():           #  遍历权重字典 w_avg 字典中的所有键; w_avg 是一个字典,其中每个键代表模型的一部分(例如权重矩阵或便置向量), 而对应的值是这些部分的参数值; w_avg.keys() 返回一个包含字典中所有键的视图对象, 这个视图对象可以用于迭代字典中的所有键
        w_avg[k] = w_avg[k] * dict_len[0]       #  将第一个客户端的权重乘以对应的样本数量 dict_len[0]
        for i in range(1, len(w)):              #  从第二个客户端开始遍历所有客户端的权重
            w_avg[k] += w[i][k] * dict_len[i]   #  将每个客户端的权重 w[i][k] 乘以对应的样本数量 dict_len[i] 并累加到 w_avg[k]
        w_avg[k] = w_avg[k] / sum(dict_len)     #  将累加后的权重 w_avg[k] 除以所有客户端的样本数量, 得到加权平均

    #      将多个客户端的模型权重聚合为一个全局模型
    return w_avg

def personalized_aggregation(netglob, w, n_bar, gamma):
    # w是新 netglobal是旧的
    w_agg = copy.deepcopy(w[0])
    for k in w_agg.keys():
        w_agg[k] = w_agg[k] * n_bar[0]
        for i in range(1, len(w)):
            w_agg[k] += w[i][k] * n_bar[i]
        if sum(n_bar) == 0:
            w_agg[k] = gamma * torch.div(w_agg[k], sum(n_bar)+100000000) + (1 - gamma) * netglob[k]
        else:
            w_agg[k] = gamma * torch.div(w_agg[k], sum(n_bar)) + (1 - gamma) * netglob[k]
    return w_agg
