import copy
import numpy as np
import torch
import random
import torchvision
import heapq
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler, Subset, Dataset, Sampler


# 该段代码用于生成一个均匀分布的腐败矩阵, 该矩阵表示在给定 corruption_ratio 和 num_classes 的数量下, 每个类标签被改变为其他类标签的概率
def uniform_corruption(corruption_ratio, num_classes):        # 该段代码用于生成一个均匀分布的腐败矩阵, 该矩阵表示在给定 corruption_ratio 和 num_classes 的数量下, 每个类标签被改变为其他类标签的概率
    corruption_matrix = np.ones((num_classes, num_classes))   # 该矩阵将存储标签转换的概率
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                corruption_matrix[i, j] = 1 - corruption_ratio                   # 如果当前类索引 i 和目标类索引 j 相同 (即对角线上的元素), 设置当前类保持不变的概率为 1 - corruption_ratio
            else:
                corruption_matrix[i, j] = corruption_ratio / (num_classes - 1)   # 如果当前类索引 i 和目标类索引 j 不同 (即非对角线上的元素), 设置当前类改变为其他类的概率为 corruption_ratio / (num_classes - 1), 即将 corruption_ratio 平分给其他所有类
    return corruption_matrix


def flip1_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i])] = corruption_ratio
    return corruption_matrix


def flip2_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_ratio / 2
    return corruption_matrix


def build_dataset(dataset_name):

    data_train = None
    data_test = None
    num_classes = 0

    if dataset_name == 'mnist':
        transform = transforms.Compose([       # 定义数据预处理的流水线, 将图像转换为张量并进行标准化处理
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        data_train = MNIST(root='data/mnist', train=True, transform=transform, download=True)
        data_test = MNIST(root='data/mnist', train=False, transform=transform, download=True)   # 10,000

        test_dataset, valid_dataset = torch.utils.data.random_split(data_test, [8000, 2000])

        num_classes = 10



    elif dataset_name == 'cifar10':
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        data_train = torchvision.datasets.CIFAR10(root='data/cifar10', train=True, download=True,
                                                  transform=train_transforms)
        data_test = torchvision.datasets.CIFAR10(root='data/cifar10', train=False, transform=test_transforms)

        test_dataset, valid_dataset = torch.utils.data.random_split(data_test, [8000, 2000])

        num_classes = 10


    elif dataset_name == 'cifar100':
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        data_train = torchvision.datasets.CIFAR100(root='data/cifar100', train=True, download=True,
                                                  transform=train_transforms)
        data_test = torchvision.datasets.CIFAR100(root='data/cifar100', train=False, transform=test_transforms)

        test_dataset, valid_dataset = torch.utils.data.random_split(data_test, [8000, 2000])  # 需要改, 数字不对

        num_classes = 100

    return data_train, valid_dataset, test_dataset, num_classes


# 之前的版本, 没有 validation dataset
# def build_dataset(dataset_name):
#     data_train = None
#     data_test = None
#     num_classes = 0
#
#     if dataset_name == 'mnist':
#         transform = transforms.Compose([       # 定义数据预处理的流水线, 将图像转换为张量并进行标准化处理
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.1307], std=[0.3081])
#         ])
#         data_train = MNIST(root='data/mnist', train=True, transform=transform, download=True)
#         data_test = MNIST(root='data/mnist', train=False, transform=transform, download=True)
#         num_classes = 10
#     elif dataset_name == 'cifar10':
#         normalize = transforms.Normalize(
#             mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#             std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
#         )
#
#         train_transforms = transforms.Compose([
#             transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ])
#
#         test_transforms = transforms.Compose([
#             transforms.ToTensor(),
#             normalize,
#         ])
#         data_train = torchvision.datasets.CIFAR10(root='data/cifar10', train=True, download=True,
#                                                   transform=train_transforms)
#         data_test = torchvision.datasets.CIFAR10(root='data/cifar10', train=False, transform=test_transforms)
#
#
#         num_classes = 10
#     elif dataset_name == 'cifar100':
#         normalize = transforms.Normalize(
#             mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#             std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
#         )
#
#         train_transforms = transforms.Compose([
#             transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ])
#
#         test_transforms = transforms.Compose([
#             transforms.ToTensor(),
#             normalize,
#         ])
#         data_train = torchvision.datasets.CIFAR100(root='data/cifar100', train=True, download=True,
#                                                   transform=train_transforms)
#         data_test = torchvision.datasets.CIFAR100(root='data/cifar100', train=False, transform=test_transforms)
#         num_classes = 100
#
#     return data_train, data_test, num_classes


# 没用到
def load_client_data(dataset_name, client_num, batch_size):
    # Build data
    data_train, data_val, data_test, num_classes = build_dataset(dataset_name)


    # Split data into dict
    data_dict = dict()
    train_per_client = len(data_train) // client_num
    test_per_client = len(data_test) // client_num

    for client_idx in range(1, client_num + 1):
        dataloader_dict = {
            'train':
                DataLoader([
                    data_train[i]
                    for i in range((client_idx - 1) *
                                   train_per_client, client_idx * train_per_client)
                ],
                    batch_size,
                    shuffle=True),
            'val':
                DataLoader([
                    data_test[i]
                    for i in range((client_idx - 1) * test_per_client, client_idx *
                                   test_per_client)
                ],
                    batch_size,
                    shuffle=False),
            'test':
                DataLoader([
                    data_test[i]
                    for i in range((client_idx - 1) * test_per_client, client_idx *
                                   test_per_client)
                ],
                    batch_size,
                    shuffle=False)
        }
        data_dict[client_idx - 1] = dataloader_dict

    return data_dict



# 需要 validation dataset (meta data) 的加载 ['train'], 所有都用 ['val'] 去验证 server model
def load_server_data(args):       # 即加载 meta data, 从 data_train 里面每一类都拿出一些样本组成 meta data
    # Build data
    data_train, data_val, data_test, num_classes = build_dataset(args.dataset_name)

    num_meta_total = args.validation_num              # server 拥有的 meta data (reward data) 的数量
    num_meta = int(num_meta_total / num_classes)      # 每个类别的元数据数量

    index_to_meta = []       # 存储元数据的索引
    index_to_train = []      # 存储训练数据的索引

    for class_index in range(num_classes):            # 每个类都放 num_meta 个元数据样本
        index_to_class = [index for index, label in enumerate(data_train.targets) if label == class_index]  # 列表包含了所有属于当前类别的样本索引
        np.random.shuffle(index_to_class)
        index_to_meta.extend(index_to_class[:num_meta])         # 这些是元数据的索引, 截取 num_meta 长度的索引作为 index_to_meta 的索引
        index_to_class_for_train = index_to_class[:]            # 是 index_to_class 的副本, 包含当前类别的所有样本索引

        index_to_train.extend(index_to_class_for_train)         # 训练数据的索引 (这样会不会训练样本中包含元数据？)

    random.shuffle(index_to_meta)

    meta_dataset = copy.deepcopy(data_train)                    # 从 training data 里面选出 meta data

    # 好像没啥用？
    data_train.data = data_train.data[index_to_train]           # 更新 data_train 的数据和标签, 仅保留训练数据索引 index_to_train 对应的样本和标签
    data_train.targets = list(np.array(data_train.targets)[index_to_train])

    meta_dataset.data = meta_dataset.data[index_to_meta]        # 更新 meta_dataset 的数据和标签, 仅保留元数据索引 index_to_meta 对应的样本和标签, 存在 meta_dataset 中 (索引会重新从 0 开始排列)
    meta_dataset.targets = list(np.array(meta_dataset.targets)[index_to_meta])

    # 对 meta data 加 uniform_corruption 噪声 (meta data 也有一定概率的噪声)
    corruption_matrix = uniform_corruption(args.validation_noise_ratio, num_classes)
    for index in range(len(meta_dataset.targets)):
        p = corruption_matrix[int(meta_dataset.targets[index])]
        meta_dataset.targets[index] = np.random.choice(num_classes, p=p)

    server_dataloader_dict = {
        'train':   # meta data
            DataLoader(meta_dataset, min(args.batch_size, num_meta_total), shuffle=False,    # 从 meta_dataset 中加载数据
                       collate_fn=None),
        'val':
            DataLoader(data_test, args.batch_size, shuffle=False,
                       collate_fn=None)
    }

    return server_dataloader_dict

    # Q: data_train 里面划分出 meta data ? 不合理, 因为这些 meta data 之前 client 训练的时候见过了呀
    #    应该再训练之前最开始划分数据的时候就把 mata data 划分出来




# 加载带有样本权重的数据
def load_client_weight_data(dataset_name, batch_size, select_samples_ratio, weight, client_index, loader):     # loader 表示原始数据加载器
    dataset = copy.deepcopy(loader.dataset)      # 对传入的 loader 的数据集进行深拷贝, 生成一个新的数据集 dataset;

    select_samples_num = min(int(select_samples_ratio * len(weight)), len(weight))

    dataloader = DataLoader(dataset, batch_size, sampler=WeightedRandomSampler(weight, select_samples_num), collate_fn=None)

    return dataloader




# Define the BernoulliSampler for sampling data based on Bernoulli probabilities
class BernoulliSampler(Sampler):
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def __iter__(self):
        # Perform Bernoulli sampling, only include samples with success in Bernoulli trial
        sample_indices = [i for i, p in enumerate(self.probabilities) if np.random.binomial(1, p) == 1]
        return iter(sample_indices)

    def __len__(self):
        return len(self.probabilities)


# Function to load client data using Bernoulli sampling
def load_client_bernoulli_data(batch_size, probabilities, loader):
    dataset = copy.deepcopy(loader.dataset)

    # Use BernoulliSampler instead of WeightedRandomSampler
    dataloader = DataLoader(dataset, batch_size, sampler=BernoulliSampler(probabilities), collate_fn=None)

    return dataloader






class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, target, idx

def create_dataloader(dataset, indices, batch_size):
    # 根据索引创建子集，并构建 DataLoader
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return dataloader

# 从训练样本中选取
def load_client_reward_data(batch_size, reward_data_num, weight, loader):
    dataset = copy.deepcopy(loader.dataset)

    dataset = CustomDataset(dataset)

    select_reward_data_num = min(reward_data_num, len(weight))

    # 获取权重最大的样本索引, 是数组[]
    top_k_indices = heapq.nlargest(select_reward_data_num, range(len(weight)), key=lambda i: weight[i])     # key=lambda i: weight[i] 的作用是将索引转换为对应的权重值, 使得 heapq.nlargest 函数可以根据权重值来选择最大的元素

    dataloader = create_dataloader(dataset, top_k_indices, min(batch_size, select_reward_data_num))

    # 计算top_k的 R 的值
    total_value = 0.0
    for i in top_k_indices:

        total_value += weight[i].item()

        # print("i: ", i)
        # print("weight[i]", weight[i].item())

    print("total_value: ", total_value)

    return dataloader, total_value


def load_corrupt_client_data(
        args,
        client_num,
        # imbalanced_factor = None,         # 不平衡因子, 这是一个控制不平衡程度的参数; 通常, 大于1表示增加不平衡 (本文中是大于1的), 小于1表示减少不平衡
        corruption_type = None,           # noisy data 的类型
        corruption_prob = 0.,            # noisy data 的比例
        # corrupt_num = 0
        ):                 # noisy data 的数量
    corruption_list = {
        'uniform': uniform_corruption,
        'flip1': flip1_corruption,
        'flip2': flip2_corruption,
    }

    # Build data

    data_train, data_val, data_test, num_classes = build_dataset(args.dataset_name)

    # data_train, data_test, num_classes = build_dataset(args.dataset_name)


    # Split data into dict
    data_dict = dict()                                  # 用于存储每个客户端的数据字典
    test_per_client = len(data_test) // client_num      # 每个客户端的测试样本数量
    num_meta_total = test_per_client                    # 元数据数量 (等于一个客户端的测试样本数量), 因为论文中选择一个 client 去训练 meta model, 元数据是从 training data 里面拿的

    index_to_train = []         # 用于存储训练数据的索引列表

    # imbalanced_factor, 主要思想就是根据不平衡因子调整每个类的样本数量
    # imbalanced_factor = (the number of training samples largest_class) / the smallest class
    if args.imbalanced_factor > 1:
        imbalanced_num_list = []                   # 用于存储每个类的不平衡样本数量
        sample_num = int((len(data_train.targets) - num_meta_total) / num_classes)                      # 在没有平衡因子的情况下, 每个类应分配到的样本数量
        for class_index in range(num_classes):     # 遍历每个类的索引
            # 计算每个类的不平衡样本数;
            # sample_num 表示每个类的基础样本数, 这是一个平均值, 表示在没有不平衡因子的情况下, 每个类应分配到的样本数
            # class_index 为当前类的索引, 从 0 到 num_classes - 1
            # (class_index / (num_classes - 1)) 为指数部分, 将类索引标准化为 [0, 1] 范围; 这种标准化使得第一个类和最后一个类分别具有最低和最高的不平衡因子的影响
            imbalanced_num = sample_num / (args.imbalanced_factor ** (class_index / (num_classes - 1)))      # 根据不平衡因子调整当前类的样本数量
            imbalanced_num_list.append(int(imbalanced_num))       # 将当前类的不平衡样本数量添加到 imbalanced_num_list
        np.random.shuffle(imbalanced_num_list)                    # 随机打乱 imbalanced_num_list 列表的顺序, 以确保不平衡样本数量的随机性
        print(imbalanced_num_list)
    else:
        imbalanced_num_list = None

    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(data_train.targets) if label == class_index]       # 结果是一个包含当前类所有样本索引的列表; enumerate(data_train.targets)：枚举 data_train.targets，返回每个样本的索引和标签
        np.random.shuffle(index_to_class)
        index_to_class_for_train = index_to_class[:]       # 创建当前类的训练样本索引的副本

        if imbalanced_num_list is not None:                # 如果设置了不平衡因子, 根据不平衡样本数量截取训练样本索引
            index_to_class_for_train = index_to_class_for_train[:min(imbalanced_num_list[class_index], len(index_to_class_for_train))]    # 从index_to_class_for_train列表中截取前XX个样本索引的一个子列表, 具体长度为两者的较小值决定; imbalanced_num_list[class_index] 当前类的不平衡样本数量; index_to_class_for_train 当前类的总样本数量

        index_to_train.extend(index_to_class_for_train)    # 将当前类的训练样本索引添加到总训练样本索引列表

    train_per_client = len(index_to_train) // client_num   # 计算每个客户端的训练样本数量

    np.random.shuffle(index_to_train)                      # 随机打乱总的训练样本索引顺序, 以确保数据随机性
    data_train.data = data_train.data[index_to_train]      # 根据新的索引重新分配训练数据, 重新分配后的 data_train.data 的样本索引会从重新从 0 - len(data_train.data-1)
    data_train.targets = list(np.array(data_train.targets)[index_to_train])   # 根据新的索引重新分配训练数据的目标标签

    targets_true = copy.deepcopy(data_train.targets)       # 深拷贝训练数据的目标标签, 保存原始目标标签的副本


    print('corruption_type', corruption_type)
    if corruption_type != 'None':                               # corruption_type 不为 None, 表示需要引入 noisy data
        print('have corruption_type: ', corruption_type)
    # if corruption_type is not None:                         # corruption_type 不为 None, 表示需要引入 noisy data
        corruption_matrix = corruption_list[corruption_type](args.corruption_prob, num_classes)
        print(corruption_matrix)
        if args.corrupt_num == -1:                              # -1 表示对所有数据引入腐败
            for index in range(len(data_train.targets)):
                p = corruption_matrix[int(data_train.targets[index])]              # 从 corruption_matrix 获取该标签对应的概率分布 p
                data_train.targets[index] = np.random.choice(num_classes, p=p)     # 从 0 到 num_classes-1 中随机选择一个类标签, 概率分布为 p
        else:
            for index in range(0, args.corrupt_num * train_per_client):                 # 总共需要引入腐败的样本数量 (corrupt_num: 有多少client的数据是腐败的 乘以 每个客户端的训练样本数量)
                p = corruption_matrix[int(data_train.targets[index])]
                data_train.targets[index] = np.random.choice(num_classes, p=p)

    # data_train = CustomDataset(data_train)   # 能返回 inx

    # 每个 client 创建一个字典 dataloader_dict
    for client_idx in range(1, client_num + 1):
        dataloader_dict = {
            'train':                             # 创建训练数据的加载器
                DataLoader([
                    data_train[i] for i in range((client_idx - 1) * train_per_client, client_idx * train_per_client)],     # 根据当前客户端的索引范围, 生成该客户端的训练数据
                    batch_size=args.batch_size, shuffle=False,     # batch_size 表示批次大小
                    collate_fn=None),            # 使用默认的批处理函数
            'train_targets_true': [              # 获取对应索引的真实标签
                targets_true[i] for i in range((client_idx - 1) * train_per_client, client_idx * train_per_client)
            ],
            'meta_train': [],                    # 初始化一个空列表, 用于存储元训练数据
            'reward_data':[],                    # FedFUEL 中存在 dictionary 里面的
            'random': [],                        # FLRD 中随机选择的样本
            'val':                               # 创建验证数据的加载器
                DataLoader([
                    data_test[i] for i in range((client_idx - 1) * test_per_client, client_idx * test_per_client)],
                    args.batch_size,
                    shuffle=False,
                    collate_fn=None),
            'test':
                DataLoader([
                    data_test[i] for i in range((client_idx - 1) * test_per_client, client_idx * test_per_client)],
                    args.batch_size,
                    shuffle=False,
                    collate_fn=None)
        }

        data_dict[client_idx - 1] = dataloader_dict    # 用于存储每个客户端的数据字典

    return data_dict



# 每个 client 只有一个类是 NonIID 的
# every client have noniid_ratio of one class, remain of this class give averagely to other clients
def load_non_iid_data(args,
                      client_num,
                      corruption_type=None,
                      corruption_ratio=0.,
                      #corrupt_num=0
                      ):           # 腐败的客户端数量
    corruption_list = {
        'uniform': uniform_corruption,
        'flip1': flip1_corruption,
        'flip2': flip2_corruption,
    }
    # Build data
    data_train, data_val, data_test, num_classes = build_dataset(args.dataset_name)

    # data_train = CustomDataset(data_train)  # 能返回 inx

    # Split data into dict
    data_dict = dict()
    test_per_client = len(data_test) // client_num            # 计算每个客户端的测试样本数量

    client_train_index = [[] for i in range(client_num)]      # 创建一个列表 client_train_index, 其中包含 client_num 个空列表, 用于存储每个客户端的训练数据索引
    main_ratio = args.noniid_ratio

    # NonIID 的主要代码 (主要就是改变每个客户端的数量来实现 NonIID)
    for class_index in range(num_classes):   # 为每个类分配样本索引并打乱顺序
        index_to_class = [index for index, label in enumerate(data_train.targets) if label == class_index]      # 遍历每个类的索引, 生成当前类的所有样本索引列表
        np.random.shuffle(index_to_class)
        total_num = len(index_to_class)           # 当前类的样本总数
        main_num = int(total_num * main_ratio)    # 主要客户端的样本数量
        other_num = round(float(total_num - main_num) / (client_num - 1))    # 其他客户端的样本数量; round() 用于将数字四舍五入到指定的位数

        client_train_index[class_index % client_num].extend(index_to_class[0:main_num])    # class_index % client_num 通过取模运算, 将特定类的主要样本索引分配给某一个客户端 (class_index % client_num)

        cnt = 0                     # 初始化计数器, 用于跟踪已经分配了样本的客户端数量
        prev_idx = main_num         # 初始化索引, 指示剩余样本的起始位置
        for client_idx in range(client_num):        # 遍历每个客户端的索引, 目的是将剩余的样本索引分配给不同的客户端
            # 有点问题, 感觉应该是 if client_idx != (class_index % client_num) 来跳过分配主要样本的客户端
            if client_idx != class_index:           # 跳过已经分配主要样本的客户端, 即 class_index % client_num 所对应的客户端; 只有其他客户端会继续进行样本索引的分配
                cnt += 1
                if cnt != client_num - 1:           # 检查当前分配样本的客户端是否是最后一个没有分配的客户端 (因为主要分配的 main_num 的客户端没有加到 cnt 中)
                    client_train_index[client_idx].extend(index_to_class[prev_idx: prev_idx + other_num])      # 将从 prev_idx 开始的 other_num 个样本索引扩展到当前客户端的训练样本索引列表中
                    prev_idx += other_num           # 更新 prev_idx, 指示下一批剩余样本的起始位置
                else:
                    client_train_index[client_idx].extend(index_to_class[prev_idx:])         # 如果当前客户端是最后一个需要分配样本的客户端, 将从 prev_idx 开始的所有剩余样本索引扩展到当前客户端的训练样本索引列表中; 这一步确保所有的样本都被分配完

    for client_idx in range(client_num):            # 遍历每个客户端并打乱样本索引
        np.random.shuffle(client_train_index[client_idx])

    targets_true = copy.deepcopy(data_train.targets)   # 深拷贝确保 targets_true 是 data_train.targets 的独立副本, 即后续修改 data_train.targets, targets_true 仍保持原值.

    if corruption_type != 'None':                               # corruption_type 不为 None, 表示需要引入 noisy data
        print('have corruption_type: ', corruption_type)
    # if corruption_type is not None:
        corruption_matrix = corruption_list[corruption_type](args.corruption_prob, num_classes)
        print(corruption_matrix)
        if args.corrupt_num == -1:       # -1 表示对所有样本进行腐败处理
            for index in range(len(data_train.targets)):                              # 遍历 data_train.targets 中所有样本的索引
                p = corruption_matrix[int(data_train.targets[index])]                 # 根据当前样本的标签从腐败矩阵中获取对应的概率分布 p
                data_train.targets[index] = np.random.choice(num_classes, p=p)        # 根据概率分布 p 随机选择一个新的类标签, 并将其赋值给当前样本
        else:
            for client_idx in range(args.corrupt_num):                                     # 对部分客户端的样本进行腐败处理, 遍历 corrupt_num 个客户端
                for index in client_train_index[client_idx]:                          # 对每个客户端, 遍历其训练样本的索引
                    p = corruption_matrix[int(data_train.targets[index])]
                    data_train.targets[index] = np.random.choice(num_classes, p=p)


    for client_idx in range(1, client_num + 1):     # 遍历每个客户端的索引, 为每个客户端构建数据加载器并将其存储在 dataloader_dict 字典中
        dataloader_dict = {
            'train':
                DataLoader([data_train[i] for i in client_train_index[client_idx - 1]],
                           batch_size=args.batch_size,
                           shuffle=False,
                collate_fn=None),
            'train_targets_true': [
                targets_true[i] for i in client_train_index[client_idx - 1]],
            'meta_train': [],
            'val':               # val 和 test 是一样的?
                DataLoader([data_test[i] for i in range((client_idx - 1) * test_per_client, client_idx * test_per_client)],    # 遍历当前客户端的验证样本索引范围
                    args.batch_size,
                    shuffle=False,
                collate_fn=None),
            'test':
                DataLoader([data_test[i] for i in range((client_idx - 1) * test_per_client, client_idx * test_per_client)],
                    args.batch_size,
                    shuffle=False,
                collate_fn=None)
        }
        data_dict[client_idx - 1] = dataloader_dict

    return data_dict


# 每一个 client 有多个类是 noniid 的
# every client have noniid_class_num classes
def load_non_iid_class_data(args,
                      client_num,
                      corruption_type=None,
                      # corruption_ratio=0.,
                      # corrupt_num=0
                    ):
    corruption_list = {
        'uniform': uniform_corruption,
        'flip1': flip1_corruption,
        'flip2': flip2_corruption,
    }
    # Build data
    data_train, data_val, data_test, num_classes = build_dataset(args.dataset_name)

    # data_train = CustomDataset(data_train)  # 能返回 inx

    # Split data into dict
    data_dict = dict()
    test_per_client = len(data_test) // client_num

    client_train_index = [[] for i in range(client_num)]                  # 存储每个客户端的训练数据索引

    noniid_class_num = int(num_classes * args.noniid_class_ratio)         # 每个客户端分配的 nonIID 类别数
    client_per_class = int(client_num * noniid_class_num / num_classes)   # 每个类别分配给的客户端的数量


    # 多个类是 noniid 的 (把这些类平分给要分配的客户端列表中, 其他客户端不平分)
    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(data_train.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        total_num = len(index_to_class)                       # 计算当前类别的样本总数
        sample_num = round(total_num / client_per_class)      # 每个客户端应分配的当前类别样本的平均数量; round() 函数进行四舍五入

        cnt = 0                # 用于计数, 记录当前类别已经分配给多少个客户端
        prev_idx = 0           # 用于跟踪当前类别的样本索引位置
        for client_idx in range(client_num):
            if client_idx not in [(j + class_index) % client_num for j in range(0, client_num - client_per_class)]:    # 确定当前客户端是否应该被分配当前类别的样本; [(j + class_index) % client_num for j in range(0, client_num - client_per_class)] 列表推导式生成一个不应该分配当前类别样本的客户端索引列表
                cnt += 1       # 计数器增加1, 记录已经有一个客户端分配了当前类别的样本
                if cnt != client_per_class:     # 检查是否已经分配了足够数量的客户端
                    client_train_index[client_idx].extend(index_to_class[prev_idx:prev_idx + sample_num])
                    prev_idx += sample_num
                else:                           # 剩余的所有样本索引分配给最后一个需要分配样本的客户端
                    client_train_index[client_idx].extend(index_to_class[prev_idx:])

    for client_idx in range(client_num):
        np.random.shuffle(client_train_index[client_idx])

    targets_true = copy.deepcopy(data_train.targets)

    if corruption_type != 'None':                               # corruption_type 不为 None, 表示需要引入 noisy data
        print('have corruption_type: ', corruption_type)
    # if corruption_type is not None:
        corruption_matrix = corruption_list[corruption_type](args.corruption_prob, num_classes)
        print(corruption_matrix)
        if args.corrupt_num == -1:        # 对所有(客户端)训练数据进行破坏
            for index in range(len(data_train.targets)):
                p = corruption_matrix[int(data_train.targets[index])]
                data_train.targets[index] = np.random.choice(num_classes, p=p)
        else:                        # 对部分客户端的数据进行破坏
            for client_idx in range(args.corrupt_num):
                for index in client_train_index[client_idx]:
                    p = corruption_matrix[int(data_train.targets[index])]
                    data_train.targets[index] = np.random.choice(num_classes, p=p)



    for client_idx in range(1, client_num + 1):
        dataloader_dict = {
            'train':
                DataLoader([data_train[i] for i in client_train_index[client_idx - 1]],
                batch_size=args.batch_size,
                shuffle=False,
            collate_fn=None),
            'train_targets_true': [
                targets_true[i] for i in client_train_index[client_idx - 1]],
            'meta_train': [],
            'val':
                DataLoader([data_test[i] for i in range((client_idx - 1) * test_per_client, client_idx * test_per_client)],
                    args.batch_size,
                    shuffle=False,
                collate_fn=None),
            'test':
                DataLoader([data_test[i] for i in range((client_idx - 1) * test_per_client, client_idx * test_per_client)],
                    args.batch_size,
                    shuffle=False,
                collate_fn=None)
        }
        data_dict[client_idx - 1] = dataloader_dict

    return data_dict
