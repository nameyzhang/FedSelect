from FedCorr.util.dataset import get_dataset
from FedCorr.util.util import add_noise
import numpy as np


def load_data_with_noisy_label(args):

    dataset_train, dataset_test, dict_users = get_dataset(args)     # dataset_train, dataset_test, 用户分配情况字典{users_id, 说分配的样本索引}

    print("args.dataset_name", args.dataset_name)

    # --------------------- Add Noise ---------------------------
    if args.dataset_name == 'clothing1m':
        y_train_noisy = np.array(dataset_train.targets)
        dataset_train.targets = y_train_noisy
        gamma_s = None
        noisy_sample_idx = None
        y_train = y_train_noisy
        print(f"len(dataset_train)= {len(dataset_train)}, len(dataset_test) = {len(dataset_test)}")
    else:
        print("args.dataset_name_else", args.dataset_name)
        y_train = np.array(dataset_train.targets)  # dataset_train 的标签
        y_train_noisy, gamma_s, real_noise_level, noisy_sample_idx = add_noise(args, y_train, dict_users)
        dataset_train.targets = y_train_noisy
        print("load_data_with_noisy_label ok")

    #      dataset_train: 加了噪声之后的训练集;
    #      dataset_test 没变；
    #      dict_users: 用户分配情况字典 { users_id, 所分配的样本索引 }, 因为只是索引，所以分配的图片的 label 都是加了噪声的
    #      y_train: dataset_train 的标签
    #      系统级噪声标记(哪些客户点受噪声影响) gamma_s, 是一个数组;
    #      被噪声影响的样本索引 noisy_samples_idx;
    return dataset_train, dataset_test, dict_users, y_train, gamma_s, noisy_sample_idx


    # start = time.time()
    # # parse args
    # args = args_parser()
    #
    #
    # for x in vars(args).items():
    #     print(x)
    #
    # if not torch.cuda.is_available():
    #     exit('ERROR: Cuda is not available!')
    # print('torch version: ', torch.__version__)
    # print('torchvision version: ', torchvision.__version__)
    #
    # # Seed
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # np.random.seed(args.seed)
    #
    # ##############################
    # # Load dataset and split users
    # ##############################
    #
    # if args.dataset == 'mnist':
    #     from six.moves import urllib
    #
    #     opener = urllib.request.build_opener()
    #     opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    #     urllib.request.install_opener(opener)
    #
    #     trans_mnist_train = transforms.Compose([
    #         transforms.RandomCrop(28, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,)),
    #     ])
    #     trans_mnist_val = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,)),
    #     ])
    #     dataset_train = MNIST(
    #         root='./data/mnist',
    #         download=True,
    #         train=True,
    #         transform=trans_mnist_train,
    #     )
    #     dataset_test = MNIST(
    #         root='./data/mnist',
    #         download=True,
    #         train=False,
    #         transform=trans_mnist_val,
    #     )
    #     num_classes = 10
    #     input_channel = 1
    #
    # elif args.dataset == 'cifar':
    #     trans_cifar10_train = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])],
    #     )
    #     trans_cifar10_val = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])],
    #     )
    #     dataset_train = CIFAR10(
    #         root='./data/cifar',
    #         download=True,
    #         train=True,
    #         transform=trans_cifar10_train,
    #     )
    #     dataset_test = CIFAR10(
    #         root='./data/cifar',
    #         download=True,
    #         train=False,
    #         transform=trans_cifar10_val,
    #     )
    #     num_classes = 10
    #     input_channel = 3
    #
    # else:
    #     raise NotImplementedError('Error: unrecognized dataset')
    #
    # labels = np.array(dataset_train.train_labels)
    # num_imgs = len(dataset_train) // args.num_shards
    # args.img_size = dataset_train[0][0].shape  # used to get model
    # args.num_classes = num_classes
    #
    # # Sample users (iid / non-iid)
    # if args.iid:
    #     dict_users = sample_iid(dataset_train, args.num_users)
    # else:
    #     dict_users = sample_noniid(
    #         labels=labels,
    #         num_users=args.num_users,
    #         num_shards=args.num_shards,
    #         num_imgs=num_imgs,
    #     )
    #
    # ##############################
    # # Add label noise to data
    # ##############################
    #
    # if args.noise_type != "clean":
    #     for user in range(args.num_users):
    #         data_indices = list(copy.deepcopy(dict_users[user]))
    #
    #         # for reproduction
    #         random.seed(args.seed)
    #         random.shuffle(data_indices)
    #
    #         noise_index = int(len(data_indices) * args.noise_rate)
    #
    #         for d_idx in data_indices[:noise_index]:
    #             true_label = dataset_train.train_labels[d_idx]
    #             noisy_label = noisify_label(true_label, num_classes=num_classes, noise_type=args.noise_type)
    #             dataset_train.train_labels[d_idx] = noisy_label
    #
    # # for logging purposes
    # log_train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs)
    # log_test_data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.bs)
