import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')

    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use, -1 for CPU')

    parser.add_argument('--algorithm', type=str, default='FedSelect', help="FedSelect")
    parser.add_argument('--model', type=str, default='lenet', help='model name')
    parser.add_argument('--dataset_name', default='mnist', type=str, help='MNIST, CIFAR10, CIFAR100')


    parser.add_argument('--client_num', type=int, default=10, help='the total number of client')
    parser.add_argument('--select_client', type=int, default=1, help='if select client by meta; An integer flag (0 or 1)')
    parser.add_argument('--select_ratio', type=float, default=0.4)


    parser.add_argument('--select_samples', type=int, default=1, help='if select sample; An integer flag (0 or 1)')
    parser.add_argument('--select_samples_ratio', type=float, default=0.6, help='select samples ratio of each client')


    parser.add_argument('--corruption_type', type=str, default=None, help='uniform')
    parser.add_argument('--corrupt_num', type=int, default=-1)
    parser.add_argument('--corruption_prob', type=float, default=0.4, help='label noise')


    # iid or non-iid
    parser.add_argument('--noniid_class_ratio', type=float, default=-1,
                        help='class ratio of each client in non-iid situation')



    parser.add_argument('--epochs', default=100, type=int, help='the max number of total epochs to run')
    parser.add_argument('--local_epochs', default=1, type=int, help='number of local epochs')

    parser.add_argument('--batch_size', default=100, type=int,
                        help='mini-batch size (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-2, help='0.1: MNIST, CIFAR 100; 0.01: CIFAR 10')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=18)


    parser.add_argument('--warmup_epochs', default=5, type=int, help='Warmup with standard training')
    parser.add_argument('--reward_data_size', type=int, default=100,
                        help='Number of dictionary size of R')
    parser.add_argument('--meta_momentum', type=float, default=0.5, help='Meta momentum')


    parser.add_argument('--main_path', type=str, default="./logs", help='path to the main')
    parser.add_argument('--wandb_name', type=str, default="my_project", help='wandb name')
    # parser.add_argument('--is_wandb', type=int, default=0, help='if use wandb to record')



    parser.add_argument('--dictionary_corruption_prob', type=float, default=0, help=('the noise ratio of the dictionary'))
    parser.add_argument('--gaussian', type=float, default=0, help='0 means no gaussian')
    parser.add_argument('--imbalanced_factor', type=int, default=1,
                        help='1 means no imbalance, >1 means to increase the imbalance degree')
    parser.add_argument('--noniid_ratio', type=float, default=-1, help='class ratio of each client in non-iid situation')



    parser.add_argument('--validation_noise_ratio', type=float, default=0.2,
                        help='the nosie ratio of validation dataset (or reward dataset)')
    parser.add_argument('--validation_num', type=int, default=1000, help='number of server validation set')


    parser.set_defaults(augment=True)

    return parser.parse_args()