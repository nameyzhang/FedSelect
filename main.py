import torch
import numpy as np
import datetime
import wandb
import os
import logging
import random

from utils.options import args_parser
from utils.utils import logging_config


from algorithm.FedSelect import FedSelect


def main(args, logger):
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    eval(args.algorithm)(args, logger)
    torch.cuda.empty_cache()




if __name__ == '__main__':

    # parse args
    args = args_parser()


    # if bool(args.is_wandb) is True:
    #     # os.environ["WANDB_MODE"] = "offline"
    #
    #     #     wandb.login(key= 'XXX')
    #
    #     run = wandb.init(
    #         project=args.wandb_name,
    #         mode="offline",
    #         # name= args.algorithm + '-' + args.dataset_name + '-' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    #         name=args.algorithm + '_' + str(args.seed),
    #         config={
    #             'algorithm': args.algorithm,
    #             'dataset_name': args.dataset_name,
    #             'model': args.model,
    #
    #             'client_num': args.client_num,
    #             'select_client': args.select_client,
    #             'select_ratio': args.select_ratio,
    #
    #             'select_samples': args.select_samples,
    #             'select_samples_ratio': args.select_samples_ratio,
    #
    #             'noniid_ratio': args.noniid_ratio,
    #             'noniid_class_ratio': args.noniid_class_ratio,
    #
    #             'corruption_type': args.corruption_type,
    #             'corrupt_num': args.corrupt_num,
    #             'corruption_prob': args.corruption_prob,
    #
    #             'imbalanced_factor': args.imbalanced_factor,
    #             'gaussian': args.gaussian,
    #
    #             'epochs': args.epochs,
    #             'local_epochs': args.local_epochs,
    #             'batch_size': args.batch_size,
    #             'learning_rate': args.lr,
    #             'momentum': args.momentum,
    #             'seed': args.seed,
    #
    #             'warmup_epochs': args.warmup_epochs,
    #             'reward_data_size': args.reward_data_size,
    #             'meta_momentum': args.meta_momentum,
    #
    #             'validation_num': args.validation_num,
    #             'validation_noise_ratio': args.validation_noise_ratio,
    #
    #             'level_n_system': args.level_n_system,
    #             'non_iid_prob_class': args.non_iid_prob_class,
    #             'max_beta': args.max_beta,
    #             'iid': args.iid
    #         }
    #     )


    # logger
    # os.makedirs(f"./logs/{args.dataset_name}/{args.algorithm}", exist_ok=True)
    os.makedirs(args.main_path, exist_ok=True)

    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)


    log_path = os.path.join(args.main_path, f'{args.dataset_name}.log')
    test_log = logging.FileHandler(log_path, 'a', encoding='utf-8')


    test_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('')
    test_log.setFormatter(formatter)

    logger.addHandler(test_log)

    KZT = logging.StreamHandler()
    KZT.setLevel(logging.DEBUG)
    formatter = logging.Formatter('')
    KZT.setFormatter(formatter)

    logger.addHandler(KZT)

    # log config
    logging_config(logger, args)


    main(args, logger)


