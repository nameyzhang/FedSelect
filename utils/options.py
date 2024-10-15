import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')  # 初始化一个参数解析器, 准备接收和解析命令行传递给脚本的参数

    # 添加命令行参数
    parser.add_argument('--gpu', type=int, default=2, help='GPU ID to use, -1 for CPU')

    parser.add_argument('--algorithm', type=str, default='FedCSS', help="FedCSS, FedFUEL")
    parser.add_argument('--model', type=str, default='lenet', help='model name')                  # 有的论文中是 CIFAR 10: resnet18; CIFAR100: resnet34; CLOTHING1M: resnet50
    parser.add_argument('--dataset_name', default='mnist', type=str, help='dataset (cifar10 [default] or mnist or cifar100)')

    # client selection 的方式和比率
    parser.add_argument('--client_num', type=int, default=10, help='the total number of client')  # 总共的 client 的数量
    parser.add_argument('--select_client', type=int, default=1, help='if select client by meta; An integer flag (0 or 1)')  # 是否进行 client_select 操作; True 就用 meta 进行选择, Flase 就是随机选择
    parser.add_argument('--select_ratio', type=float, default=0.4)   # 选择的client的比例, 选择的 client 的数量为 client_select_num = int(client_num * args.select_ratio)


    # sample selection 的方式和比率
    parser.add_argument('--select_samples', type=int, default=1, help='if select sample; An integer flag (0 or 1)')  # 是否进行 sample_select 操作
    parser.add_argument('--select_samples_ratio', type=float, default=0.5, help='select samples ratio of each client')  # 是不是改成比率会比较合理, 因为每一个 client 拥有的样本数量不一致; select_samples 是 True 才有用; 每个 client 选择的样本数目, 应该是 min{select_samples_num, 每个拥有的样本数目}


    # 只有在 corruption_type 不为 none 时, 才会有下面两个参数
    parser.add_argument('--corruption_type', type=str, default=None, help='None, uniform, flip1, flip2')
    parser.add_argument('--corrupt_num', type=int, default=-1)          # 有多少 client 的数据是腐败的; -1 表示对所有的数据都引入腐败
    parser.add_argument('--corruption_prob', type=float, default=0.4, help='label noise')     # corruption_ratio 是 noisy data 的概率, 控制这个 client 中标签被腐败的程度, 从而引入噪声数据


    # iid or non-iid
    parser.add_argument('--noniid_ratio', type=float, default=-1,
                        # 只有一个类是 noniid 的; noniid 是指分配的样本是否包含所有的类, 包含所有的类的是 iid, 不包含所有类的是 noniid
                        help='class ratio of each client in non-iid situation')  # noniid_ratio 乘上 总的样本数 = 主要客户端的样本数; 其他客户端平分这个类剩下的样本数; -1 表示没有
    parser.add_argument('--noniid_class_ratio', type=float, default=-1,  # 有多少个类是 noniid 的
                        help='class ratio of each client in non-iid situation')




    # parser.add_argument('--imbalanced_server', type=float, default=None)  # ?没看到哪里用到呀


    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')  #
    parser.add_argument('--local_epochs', default=5, type=int, help='number of local epochs')  # 每个 client 训练的次数

    parser.add_argument('--batch_size', default=100, type=int,  # batch_size 决定每个批次中的样本数量
                        help='mini-batch size (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)


    parser.add_argument('--target_accuracy', type=int, default=94, help=('mnist:94.0; cifar-10: 50.0'))

    # For alation study
    parser.add_argument('--dictionary_corruption_prob', type=float, default=0.2, help=('the noise ratio of the dictionary'))


    # for sensitivity analysis
    parser.add_argument('--gaussian', type=float, default=0, help='0 means no gaussian')     # ?通过高斯噪声控制DP
    parser.add_argument('--imbalanced_factor', type=int, default=1,
                        help='1 means no imbalance, >1 means to increase the imbalance degree')  # 根据 imbalanced_factor (>1) 来调整每个类的样本数量 (imbalanced_factor 增大, 样本数量减少); imbalanced_factor=0 表示不存在不平衡



    # Used for the FedFUEl method.
    parser.add_argument('--warmup_epochs', default=5, type=int, help='Warmup with standard training')
    parser.add_argument('--reward_data_size', type=int, default=500,
                        help='Number of dictionary size of R')  # 和 validation_num 一个意思
    parser.add_argument('--meta_momentum', type=float, default=0.9, help='Meta momentum')



    # Used for the validation dependency method (FedCSS, FLRD).
    parser.add_argument('--validation_noise_ratio', type = float, default=0.2, help = 'the nosie ratio of validation dataset (or reward dataset)')     # validation dataset 的噪声



    # Used for the SFedAvg
    #   下面这几个好像都用不到, 先不用管
    # parser.add_argument('--iid', type=int, default=0, help='Default set to IID. Set to 0 for non-IID')  # 0 NonIID
    # parser.add_argument('--unequal', type=int, default=0, help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)')
    # '''
    #     0 - 0_NonIID
    #     1 - 2_LabelNoise
    #     2 - 4_DataNoise
    #     3 - 5_GradientNoise
    #
    #     NoiseWord = ["0_NonIID", "1_LongTail", "2_LabelNoise", "3_LabelNoise2", "4_DataNoise", "5_GradientNoise", "6_RandomAttack", "7_ReverseGradient", "8_ConstantAttack"]
    # '''
    # parser.add_argument('--noise', type=int, default=2, help='init avg beta')
    # parser.add_argument('--noiselevel', type=float, default=0, help='gradient noiselevel / Level of noise injected. Default is 0')
    # parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")




    # Used for the FedCSS method.
    parser.add_argument('--validation_num', type=int, default=1000, help='number of server validation set')  # 应该就是 meta data (reward data)




    # User for FedDELTA
    parser.add_argument('--csd_importance', type = float, default=0, help = 'control the weight of csd loss')




    # Used for FedCorr
    parser.add_argument('--level_n_system', type=float, default=0.5, help="fraction of noisy clients")
    parser.add_argument('--non_iid_prob_class', type=float, default=0, help="non iid sampling prob for class")
    parser.add_argument('--max_beta', type=float, default=20, help="beta for coresloss，2 for mnist/cifar10,  20 for cifar100, 2.8 for clothing1M")
    parser.add_argument('--iid', type=int, default=0, help='Default set to IID. Set to 0 for iid')


    parser.add_argument('--alpha_dirichlet', type=float, default=10)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--beta', type=float, default=5, help="coefficient for local proximal，0 for fedavg, 1 for fedprox, 5 for noise fl")
    parser.add_argument('--begin_sel', type=float, default=1, help="which rounds to begin select clean samples")
    parser.add_argument('--without_CR', action='store_true', help='whether with CR loss')
    parser.add_argument('--without_alternative_update', action='store_true', help='whether with alternative update')
    parser.add_argument('--plr', help="--personal_learning_rate", type=str, default=0.1)
    parser.add_argument('--iteration1', type=int, default=1, help="enumerate iteration in preprocessing stage")


    parser.set_defaults(augment=True)  # 为命令行参数设置默认值

    return parser.parse_args()         # 解析命令行参数