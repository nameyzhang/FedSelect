import sys
sys.path.append("..")
from FedCorr.util.dataset import get_dataset
from scipy import stats
import torch.utils.data as data
import numpy as np
import torch
 
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x_train, label) -> None:
        self.imgs = x_train
        self.labels = label
        
    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)

def split_data(args):

    dataset_train, dataset_test, dict_users = get_dataset(args)
    img, target = dataset_train[0]
    # print(img.shape)
    # print(target)

    # 合并训练集和测试集
    # dataset = data.ConcatDataset([dataset_train, dataset_test])

    # 打印合并后的数据集大小
    # print("合并后的数据集大小:", len(dataset))

    # 定义拆分比例
    benchmark_ratio = 0.03  # benchmark dataset比例
    fliter_ratio = 0.97    # 待过滤 dataset比例

    # 计算拆分的长度
    # benchmark_size = int(benchmark_ratio * len(dataset))
    # fliter_size = int(fliter_ratio * len(dataset))
    # test_size = 0

    # 使用random_split函数拆分数据集
    # benchmark_dataset, fliter_dataset, test_dataset = data.random_split(dataset, [benchmark_size, fliter_size, test_size])

    benchmark_dataset_train, fliter_dataset_train = my_split(args=args, ratio1=benchmark_ratio, ratio2=fliter_ratio, dataset=dataset_train);
    benchmark_dataset_test, fliter_dataset_test = my_split(args=args, ratio1=benchmark_ratio, ratio2=fliter_ratio, dataset=dataset_test);
    benchmark_dataset = data.ConcatDataset([benchmark_dataset_train, benchmark_dataset_test])

    return benchmark_dataset, fliter_dataset_train, fliter_dataset_test

def my_split(args, ratio1, ratio2, dataset):
    # 计算拆分的长度
    dataset1_size = int(ratio1 * len(dataset))
    dataset2_size = int(len(dataset) - dataset1_size)
    dataset1, dataset2 = data.random_split(dataset, [dataset1_size, dataset2_size], generator=torch.Generator().manual_seed(args.seed))
    return dataset1, dataset2

def get_n_sample_to_keep(sorted_d, list_loss_benchmark): # select n_sample to keep
    # distance_list_validation = []
    # with open(str(results_file_path) + './loss_benchmark.csv') as f:
    #     for line in f:
    #         # l = line.replace('\n','').split(',')
    #         l = line.replace('[', '')
    #         l = l.replace(']', '')
    #         l = l.replace('(', '')
    #         l = l.replace(')', '')
    #         l = l.split(',')
    #         distance_list_validation.append(float(l[0]))

    distance_list_validation = list_loss_benchmark

    indices_list = []
    distance_list = []
    for k, v in sorted_d:
        indices_list.append(k)
        distance_list.append(v)

    # with open(str(results_file_path) + './loss_fliter.csv') as f:
    #     for line in f:
    #         # l = line.replace('\n','').split(',')
    #         l = line.replace('[', '')
    #         l = l.replace(']', '')
    #         l = l.replace('(', '')
    #         l = l.replace(')', '')
    #         l = l.split(',')
    #         indices_list.append(int(l[0]))
    #         distance_list.append(float(l[1]))

    max_each_cut = []
    steps = np.arange(10, len(distance_list), 100)


    list_cut = []
    for i in steps:
        list_cut.append(distance_list[i])

    cdf = stats.cumfreq(distance_list_validation, numbins=100, defaultreallimits=(0, 30))  # 150

    for t in steps:
        cdf2 = stats.cumfreq(distance_list[0:t], numbins=100, defaultreallimits=(0, 30))  # 150
        difference = abs(cdf.cumcount / cdf.cumcount[-1] - cdf2.cumcount / cdf2.cumcount[-1])

        max_value = max(difference)

        max_each_cut.append(max_value)

    n_sample_to_keep=steps[np.argmin(max_each_cut)]
    # print('n sample to keep',n_sample_to_keep)

    indices_to_keep = []
    for i in range(0, n_sample_to_keep):
        # print(i)

        # indices_to_keep.append(sorted_d[i][0])
        indices_to_keep.append(indices_list[i])
    return n_sample_to_keep,indices_to_keep

def wash_data(dict_users, indices_to_keep):
    # update dict_users
    for user in dict_users:
        for index in dict_users[user].copy():
            if index not in indices_to_keep:
                dict_users[user].remove(index)
    return dict_users