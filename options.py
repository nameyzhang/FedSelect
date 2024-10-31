import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')

    parser.add_argument('--gpu', type=int, default=2, help='GPU ID to use, -1 for CPU')

    parser.add_argument('--algorithm', type=str, default='FedCSS', help="FedCSS, FedFUEL")
    parser.add_argument('--model', type=str, default='lenet', help='model name')
    parser.add_argument('--dataset_name', default='mnist', type=str, help='MNIST, EMNIST, FashionMNIST, CIFAR10, CIFAR100')

    parser.add_argument('--client_num', type=int, default=10, help='the total number of client')
    parser.add_argument('--select_client', type=int, default=1, help='if select client by meta; An integer flag (0 or 1)')
    parser.add_argument('--select_ratio', type=float, default=0.4)

    parser.add_argument('--select_samples', type=int, default=1, help='if select sample; An integer flag (0 or 1)')
    parser.add_argument('--select_samples_ratio', type=float, default=0.5, help='select samples ratio of each client')


    parser.add_argument('--corruption_type', type=str, default=None, help='None, uniform, flip1, flip2')
    parser.add_argument('--corrupt_num', type=int, default=-1)
    parser.add_argument('--corruption_prob', type=float, default=0.4, help='label noise')


    # iid or non-iid
    parser.add_argument('--noniid_ratio', type=float, default=-1,
                        help='class ratio of each client in non-iid situation')
    parser.add_argument('--noniid_class_ratio', type=float, default=-1,
                        help='class ratio of each client in non-iid situation')



    # parser.add_argument('--imbalanced_server', type=float, default=None)


    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
    parser.add_argument('--local_epochs', default=5, type=int, help='number of local epochs')

    parser.add_argument('--batch_size', default=100, type=int,
                        help='mini-batch size (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)


    parser.add_argument('--target_accuracy', type=int, default=94, help='mnist:94.0; cifar-10: 50.0')
    parser.add_argument('--calculate_efficiency', type=int, default=0, help='for the efficiency experiment')


    parser.add_argument('--dictionary_corruption_prob', type=float, default=0.2, help=('the noise ratio of the dictionary'))


    parser.add_argument('--gaussian', type=float, default=0, help='0 means no gaussian')
    parser.add_argument('--imbalanced_factor', type=int, default=1,
                        help='1 means no imbalance, >1 means to increase the imbalance degree')



    parser.add_argument('--warmup_epochs', default=5, type=int, help='Warmup with standard training')
    parser.add_argument('--reward_data_size', type=int, default=500,
                        help='Number of dictionary size of R')
    parser.add_argument('--meta_momentum', type=float, default=0.9, help='Meta momentum')


    parser.set_defaults(augment=True)

    return parser.parse_args()