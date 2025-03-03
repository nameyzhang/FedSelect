import copy
import numpy as np
import torch
import random
import torchvision
import heapq
import os

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler, Subset, Dataset, Sampler


def build_dataset(dataset_name):

    data_train = None
    data_test = None
    num_classes = 0

    data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
    assert dataset_name in data_dict, "The dataset is not present"

    root = f'./data/{dataset_name}'

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)


    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])])
        data_train = MNIST(root=root, train=True, transform=transform, download=True)
        data_test = MNIST(root=root, train=False, transform=transform, download=True)   # 10,000

    elif dataset_name == 'EMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        data_train = torchvision.datasets.EMNIST(root=root, train=True, split='letters', download=True, transform=transform)
        data_test = torchvision.datasets.EMNIST(root=root, train=False, split='letters', download=True, transform=transform)

    elif dataset_name == 'CIFAR10':
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
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
        data_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True,
                                                  transform=train_transforms)
        data_test = torchvision.datasets.CIFAR10(root=root, train=False, transform=test_transforms)

    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        data_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        data_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)


    elif dataset_name == 'CIFAR100':
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
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
        data_train = torchvision.datasets.CIFAR100(root=root, train=True, download=True,
                                                  transform=train_transforms)
        data_test = torchvision.datasets.CIFAR100(root=root, train=False, transform=test_transforms)


    len_classes_dict = {
        'MNIST': 10,
        'EMNIST': 26,  # ByClass: 62. ByMerge: 814,255 47.Digits: 280,000 10.Letters: 145,600 26.MNIST: 70,000 10.
        'FashionMNIST': 10,
        'CIFAR10': 10,
        'CIFAR100': 100
    }

    num_classes = len_classes_dict[dataset_name]

    # split the data_test into data_valid and data_test 0.2 : 0.8
    valid_size = int(len(data_test) * 0.2)
    dataset_valid, dataset_test = torch.utils.data.random_split(data_test, [valid_size, len(data_test) - valid_size])


    return data_train, dataset_valid, dataset_test, num_classes



def uniform_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.ones((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                corruption_matrix[i, j] = 1 - corruption_ratio
            else:
                corruption_matrix[i, j] = corruption_ratio / (num_classes - 1)
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



def load_server_data(args):
    # Build data
    data_train, data_val, data_test, num_classes = build_dataset(args.dataset_name)

    num_meta_total = args.validation_num
    num_meta = int(num_meta_total / num_classes)

    index_to_meta = []
    # index_to_train = []

    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(data_train.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        index_to_meta.extend(index_to_class[:num_meta])

        # index_to_class_for_train = index_to_class[:]
        # index_to_train.extend(index_to_class_for_train)

    random.shuffle(index_to_meta)

    meta_dataset = copy.deepcopy(data_train)

    meta_dataset.data = meta_dataset.data[index_to_meta]
    meta_dataset.targets = list(np.array(meta_dataset.targets)[index_to_meta])

    corruption_matrix = uniform_corruption(args.validation_noise_ratio, num_classes)
    for index in range(len(meta_dataset.targets)):
        p = corruption_matrix[int(meta_dataset.targets[index])]
        meta_dataset.targets[index] = np.random.choice(num_classes, p=p)

    server_dataloader_dict = {
        'train':   # meta data
            DataLoader(meta_dataset, min(args.batch_size, num_meta_total), shuffle=False,
                       collate_fn=None),
        'val':
            DataLoader(data_test, args.batch_size, shuffle=False,
                       collate_fn=None)
    }

    return server_dataloader_dict


def load_client_weight_data(batch_size, select_samples_ratio, weight, loader):
    dataset = copy.deepcopy(loader.dataset)

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
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return dataloader




def load_client_reward_data(batch_size, reward_data_num, weight, loader):
    dataset = copy.deepcopy(loader.dataset)

    dataset = CustomDataset(dataset)

    select_reward_data_num = min(reward_data_num, len(weight))

    top_k_indices = heapq.nlargest(select_reward_data_num, range(len(weight)), key=lambda i: weight[i])


    dataloader = create_dataloader(dataset, top_k_indices, min(batch_size, select_reward_data_num))

    total_value = 0.0
    for i in top_k_indices:

        total_value += weight[i].item()

    print("total_value: ", total_value)

    return dataloader, total_value





def load_corrupt_client_data(
        args,
        client_num,
        # imbalanced_factor = None,
        corruption_type = None,
        corruption_prob = 0.,
        # corrupt_num = 0
        ):
    corruption_list = {
        'uniform': uniform_corruption,
        'flip1': flip1_corruption,
        'flip2': flip2_corruption,
    }

    # Build data

    data_train, data_val, data_test, num_classes = build_dataset(args.dataset_name)

    # data_train, data_test, num_classes = build_dataset(args.dataset_name)


    # Split data into dict
    data_dict = dict()
    test_per_client = len(data_test) // client_num
    num_meta_total = test_per_client

    index_to_train = []

    if args.imbalanced_factor > 1:
        imbalanced_num_list = []
        sample_num = int((len(data_train.targets) - num_meta_total) / num_classes)
        for class_index in range(num_classes):
            imbalanced_num = sample_num / (args.imbalanced_factor ** (class_index / (num_classes - 1)))
            imbalanced_num_list.append(int(imbalanced_num))
        np.random.shuffle(imbalanced_num_list)
        print(imbalanced_num_list)
    else:
        imbalanced_num_list = None

    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(data_train.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        index_to_class_for_train = index_to_class[:]

        if imbalanced_num_list is not None:
            index_to_class_for_train = index_to_class_for_train[:min(imbalanced_num_list[class_index], len(index_to_class_for_train))]

        index_to_train.extend(index_to_class_for_train)

    train_per_client = len(index_to_train) // client_num

    np.random.shuffle(index_to_train)
    data_train.data = data_train.data[index_to_train]
    data_train.targets = list(np.array(data_train.targets)[index_to_train])

    targets_true = copy.deepcopy(data_train.targets)


    print('corruption_type', corruption_type)
    if corruption_type != 'None':
        print('have corruption_type: ', corruption_type)
    # if corruption_type is not None:
        corruption_matrix = corruption_list[corruption_type](args.corruption_prob, num_classes)
        print(corruption_matrix)
        if args.corrupt_num == -1:
            for index in range(len(data_train.targets)):
                p = corruption_matrix[int(data_train.targets[index])]
                data_train.targets[index] = np.random.choice(num_classes, p=p)
        else:
            for index in range(0, args.corrupt_num * train_per_client):
                p = corruption_matrix[int(data_train.targets[index])]
                data_train.targets[index] = np.random.choice(num_classes, p=p)


    for client_idx in range(1, client_num + 1):
        dataloader_dict = {
            'train':
                DataLoader([
                    data_train[i] for i in range((client_idx - 1) * train_per_client, client_idx * train_per_client)],
                    batch_size=args.batch_size, shuffle=False,
                    collate_fn=None),
            'train_targets_true': [
                targets_true[i] for i in range((client_idx - 1) * train_per_client, client_idx * train_per_client)
            ],
            'meta_train': [],
            'reward_data':[],
            'random': [],
            'val':
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

        data_dict[client_idx - 1] = dataloader_dict

    return data_dict





# every client have noniid_ratio of one class, remain of this class give averagely to other clients
def load_non_iid_data(args,
                      client_num,
                      corruption_type=None,
                      corruption_ratio=0.,
                      #corrupt_num=0
                      ):
    corruption_list = {
        'uniform': uniform_corruption,
        'flip1': flip1_corruption,
        'flip2': flip2_corruption,
    }
    # Build data
    data_train, data_val, data_test, num_classes = build_dataset(args.dataset_name)


    # Split data into dict
    data_dict = dict()
    test_per_client = len(data_test) // client_num

    client_train_index = [[] for i in range(client_num)]
    main_ratio = args.noniid_ratio

    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(data_train.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        total_num = len(index_to_class)
        main_num = int(total_num * main_ratio)
        other_num = round(float(total_num - main_num) / (client_num - 1))

        client_train_index[class_index % client_num].extend(index_to_class[0:main_num])

        cnt = 0
        prev_idx = main_num
        for client_idx in range(client_num):
            if client_idx != class_index:
                cnt += 1
                if cnt != client_num - 1:
                    client_train_index[client_idx].extend(index_to_class[prev_idx: prev_idx + other_num])
                    prev_idx += other_num
                else:
                    client_train_index[client_idx].extend(index_to_class[prev_idx:])

    for client_idx in range(client_num):
        np.random.shuffle(client_train_index[client_idx])

    targets_true = copy.deepcopy(data_train.targets)

    if corruption_type != 'None':
        print('have corruption_type: ', corruption_type)
    # if corruption_type is not None:
        corruption_matrix = corruption_list[corruption_type](args.corruption_prob, num_classes)
        print(corruption_matrix)
        if args.corrupt_num == -1:
            for index in range(len(data_train.targets)):
                p = corruption_matrix[int(data_train.targets[index])]
                data_train.targets[index] = np.random.choice(num_classes, p=p)
        else:
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
            'reward_data': [],
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

    # data_train = CustomDataset(data_train)

    # Split data into dict
    data_dict = dict()
    test_per_client = len(data_test) // client_num

    client_train_index = [[] for i in range(client_num)]

    noniid_class_num = int(num_classes * args.noniid_class_ratio)
    client_per_class = int(client_num * noniid_class_num / num_classes)


    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(data_train.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        total_num = len(index_to_class)
        sample_num = round(total_num / client_per_class)

        cnt = 0
        prev_idx = 0
        for client_idx in range(client_num):
            if client_idx not in [(j + class_index) % client_num for j in range(0, client_num - client_per_class)]:
                cnt += 1
                if cnt != client_per_class:
                    client_train_index[client_idx].extend(index_to_class[prev_idx:prev_idx + sample_num])
                    prev_idx += sample_num
                else:
                    client_train_index[client_idx].extend(index_to_class[prev_idx:])

    for client_idx in range(client_num):
        np.random.shuffle(client_train_index[client_idx])

    targets_true = copy.deepcopy(data_train.targets)

    if corruption_type != 'None':
        print('have corruption_type: ', corruption_type)
    # if corruption_type is not None:
        corruption_matrix = corruption_list[corruption_type](args.corruption_prob, num_classes)
        print(corruption_matrix)
        if args.corrupt_num == -1:
            for index in range(len(data_train.targets)):
                p = corruption_matrix[int(data_train.targets[index])]
                data_train.targets[index] = np.random.choice(num_classes, p=p)
        else:
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
            'reward_data': [],
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
