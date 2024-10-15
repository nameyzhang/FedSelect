import torch
import numpy as np
import datetime
import wandb
import os
import logging

from utils.options import args_parser
from utils.utils import logging_config

from algorithm.FedAvg import FedAvg
from algorithm.FedBary import FedBary
from algorithm.FedCSS import FedCSS
from algorithm.FedDELTA import FedDELTA
from algorithm.FedFUEL import FedFUEL
from algorithm.FedGradient import FedGradient
from algorithm.FedLoss import FedLoss
from algorithm.FedMR import FedMR
from algorithm.SFedAvg import SFedAvg
from algorithm.FedCorr import FedCorr
from algorithm.FedFUEL_dictionary_noise import FedFUEL_dictionary_noise

from algorithm.FedFUEL_Aba_Loss import FedFUEL_Aba_Loss
from algorithm.FedFUEL_Aba_random import FedFUEL_Aba_random
from algorithm.FedFUEL_Aba_AUM import FedFUEL_Aba_AUM
from algorithm.FedFUEL_Aba_grad import FedFUEL_Aba_grad
from algorithm.FedFUEL_all_meta_model import FedFUEL_all_meta_model

from algorithm.FedFUEL_without_meta import FedFUEL_without_meta


def main(args, logger):
    torch.manual_seed(args.seed)         # 设置 PyTorch 的随机数生成器种子
    np.random.seed(args.seed)            # 设置 Numpy 的

    torch.backends.cudnn.deterministic = True    # 确定性：设置 torch.backends.cudnn.deterministic = True 确保结果在相同条件下可重复。
    torch.backends.cudnn.benchmark = True        # 性能：设置 torch.backends.cudnn.benchmark = True 可以提高训练速度，前提是输入大小是固定的。


    eval(args.algorithm)(args, logger)                  #  例如, 等同于 FedVag(args)
    torch.cuda.empty_cache()                            #  用于释放未被分配的显存空间; 它可以清理CUDA缓存，减少显存占用




if __name__ == '__main__':

    # parse args
    args = args_parser()


    # wandb 初始化
    wandb.login(key= '0e6e68b06f47fc7a5ce1acafc61cd724a307be96')

    run = wandb.init(
        project='Fed_DataSelection_accuracy',
        name= args.algorithm + '-' + args.dataset_name + '-' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        config={
            'algorithm': args.algorithm,
            'dataset_name': args.dataset_name,
            'model': args.model,

            'client_num': args.client_num,
            'select_client': args.select_client,
            'select_ratio': args.select_ratio,

            'select_samples': args.select_samples,
            'select_samples_ratio': args.select_samples_ratio,

            'noniid_ratio': args.noniid_ratio,
            'noniid_class_ratio': args.noniid_class_ratio,

            'corruption_type': args.corruption_type,
            'corrupt_num': args.corrupt_num,
            'corruption_prob': args.corruption_prob,

            'imbalanced_factor': args.imbalanced_factor,
            'gaussian': args.gaussian,

            'epochs': args.epochs,
            'local_epochs': args.local_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epoch': args.epochs,
            'momentum': args.momentum,
            'seed': args.seed,

            'validation_num': args.validation_num,

            'warmup_epochs': args.warmup_epochs,
            'reward_data_size': args.reward_data_size,
            'meta_momentum': args.meta_momentum
        }
    )



    # logger
    os.makedirs(f"./logs/{args.dataset_name}/{args.algorithm}", exist_ok=True)  # 创建日志目录
    # os.makedirs("./logs/{}/{}".format(args.dataset_name, args.algorithm), exist_ok=True)  # 创建日志目录

    logger = logging.getLogger('test_logger')  # 获取一个名为 test_logger 的日志记录器对象; 日志记录器是用于记录日志信息的对象, 通过它可以控制日志的级别、格式、输出位置等.
    logger.setLevel(
        logging.DEBUG)  # 设置日志记录器的日志级别 DEBUG, 这意味着所有级别大于等于 DEBUG 的日志消息 (包括 DEBUG, INFO, WARNING, ERROR, CRITICAL)都会被记录

    test_log = logging.FileHandler(f'./logs/{args.dataset_name}/{args.algorithm}/{args.algorithm}.log', 'a',
                                   encoding='utf-8')  # 创建一个文件处理器, 用于将日志写入文件, model: 'a' 表示文件打开模式, 这里是附加模式 (append), 即日志消息会追加到文件末尾.
    test_log.setLevel(logging.DEBUG)  # 设置文件处理器的日志级别
    formatter = logging.Formatter('')  # 创建一个格式化器对象, 用于指定日志消息的输出格式. 这里的格式化器没有指定任何格式, 这意味着日志消息将使用默认格式.
    test_log.setFormatter(formatter)  # 将格式化器对象设置到文件处理器, 这意味着处理器将使用指定的格式化器来格式化日志消息.

    logger.addHandler(test_log)  # 将文件处理器添加到日志记录器, 这意味着日志记录器将使用这个处理器来处理日志消息, 即将日志消息写入指定的文件.

    KZT = logging.StreamHandler()  # 创建一个日志流处理器对象, 用于将日志消息输出到控制台(标准输出)
    KZT.setLevel(logging.DEBUG)
    formatter = logging.Formatter('')
    KZT.setFormatter(formatter)  # 将格式化器对象设置到流处理器

    logger.addHandler(KZT)  # 将流处理器添加到日志记录器

    # log config
    logging_config(logger, args)  # 用logging_config来配置logger


    main(args, logger)


