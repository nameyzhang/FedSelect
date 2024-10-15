from PIL import Image
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from FedCorr.util.sampling import iid_sampling, non_iid_dirichlet_sampling
import torch.utils


def get_dataset(args):

    if args.dataset_name == 'mnist':
        data_path = './data/mnist'
        num_classes = 10
        from six.moves import urllib  # 用于处理URL请求

        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        trans_mnist_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4), # 随机裁剪图像到28*28, 并使用4个像素填充
            transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
            transforms.Resize((28, 28)),  # 将图像调整为32*32像素, 后来报错又改成了 28 * 28
            transforms.ToTensor(),        # 将图像转换为张量
            transforms.Normalize((0.1307,), (0.3081,)),  # 使用均值 0.1307和标准差0.3081对图像进行归一化
        ])
        trans_mnist_val = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset_train = datasets.MNIST(
            root=data_path,
            download=True,  # 如果数据不存在，则从互联网上下载
            train=True,
            transform=trans_mnist_train,  # 将其进行之前定义的训练转换
        )
        dataset_test = datasets.MNIST(
            root=data_path,
            download=True,
            train=False,
            transform=trans_mnist_val,
        )
        n_train = len(dataset_train)   # number of training
        y_train = np.array(dataset_train.targets)   # 将训练数据集的标签转换为NumPy数组, 并将其存储在变量 y_train 中,‘dataset_train.targets’包含所有训练样本的标签, 例如: array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4, ...]), 一维的NumPy数组


    elif args.dataset_name == 'cifar10':
        data_path = './data/cifar10'
        num_classes = 10
        # args.model = 'resnet18'
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)
        print("cifar 10 ok")
    elif args.dataset_name == 'cifar100':
        data_path = './data/cifar100'
        num_classes = 100
        # args.model = 'resnet34'
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        dataset_train = datasets.CIFAR100(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    elif args.dataset_name == 'clothing1m':
        data_path = os.path.abspath('.') + '/data/clothing1M/'
        num_classes = 14
        # args.model = 'resnet50'
        trans_train = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 ])
        trans_val = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 ])
        dataset_train = Clothing(data_path, trans_train, "train")
        dataset_test = Clothing(data_path, trans_val, "test")
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    elif args.dataset_name == 'fashion':
        pass
        data_path = './data/fashion'
        num_classes = 10
        
        trans_fashion_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        trans_fashion_val = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        dataset_train = datasets.FashionMNIST(
            root=data_path,
            download=True,
            train=True,
            transform=trans_fashion_train
        )
        dataset_test = datasets.FashionMNIST(
            root=data_path,
            download=True,
            train=False,
            transform=trans_fashion_val,
        )
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    else:
        exit('Error: unrecognized dataset')

    if args.iid == 0:
        dict_users = iid_sampling(n_train, args.client_num, args.seed)
    else:
        dict_users = non_iid_dirichlet_sampling(y_train, num_classes, args.non_iid_prob_class, args.client_num, args.seed, args.alpha_dirichlet)
    print("get_dataset end")

    return dataset_train, dataset_test, dict_users     # dataset_train, dataset_test, 用户分配情况字典{users_id, 所分配的样本索引}


class Clothing(torch.utils.data.Dataset):
    def __init__(self, root, transform, mode):
        self.root = root
        self.noisy_labels = {}
        self.clean_labels = {}
        self.data = []
        self.targets = []
        self.transform = transform
        self.mode = mode

        with open(self.root + 'noisy_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.noisy_labels[img_path] = int(entry[1])

        with open(self.root + 'clean_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.clean_labels[img_path] = int(entry[1])

        if self.mode == 'train':
            with open(self.root + 'noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)
        elif self.mode == 'minitrain':
            with open(self.root + 'noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            n = len(lines)
            np.random.seed(13)
            subset_idx = np.random.choice(n, int(n/10), replace=False)
            for i in subset_idx:
                l = lines[i]
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)
        elif self.mode == 'test':
            with open(self.root + 'clean_test_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.clean_labels[img_path]
                self.targets.append(target)

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.data)
