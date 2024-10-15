import torch
import datetime
import os
import logging

def get_device(args):
    device = torch.device(                      # torch.device 是 pytorch 提供的用于指定设备的函数, 它接受一个字符串参数来指定设备类型和编号
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu'
    )

    return device


def logging_config(logger, args):
    logger.info("--------------------------------------------------------")
    logger.info(f"Config:")
    logger.info(f"\ttraining_time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info(f"\talgorithm: {args.algorithm}")
    logger.info(f"\tdataset_name: {args.dataset_name}")
    logger.info(f"\tmodel: {args.model}")

    logger.info(f"\tselect_client: {args.select_client}")
    logger.info(f"\tclient_num: {args.client_num}")
    logger.info(f"\tclient_select_ratio: {args.select_ratio}")                     # 选择的 client 的比率

    logger.info(f"\tselect_samples: {args.select_samples}")
    logger.info(f"\tsample_select_ratio:{args.select_samples_ratio}")          # select_samples 为 True 才有用; 每个 client 选择的样本数量


    logger.info(f"\tcorruption_type: {args.corruption_type}")
    logger.info(f"\tcorrupt_num: {args.corrupt_num}")
    logger.info(f"\tcorruption_prob: {args.corruption_prob}")


    logger.info(f"\tnoniid_ratio: {args.noniid_ratio}")
    logger.info(f"\tnoniid_class_ratio: {args.noniid_class_ratio}")

    logger.info(f"\timbalanced_factor: {args.imbalanced_factor}")
    logger.info(f"\tgaussian: {args.gaussian}")

    logger.info(f"\tepochs: {args.epochs}")
    logger.info(f"\tlocal_epochs: {args.local_epochs}")
    logger.info(f"\tbatch_size: {args.batch_size}")
    logger.info(f"\tlr: {args.lr}")
    logger.info(f"\tmomentum: {args.momentum}\n")
    logger.info(f"\tseed: {args.seed}")


    logger.info(f"\tUsed for FedFUEl")
    logger.info(f"\twarmup_epochs: {args.warmup_epochs}")
    logger.info(f"\treward_data_size: {args.reward_data_size}")
    logger.info(f"\tmeta_momentum: {args.meta_momentum}")


    logger.info(f"\tUsed for the validation dependency method (FedCSS, FLRD)")
    logger.info(f"\tvalidation_noise_ratio: {args.validation_noise_ratio}")


    # logger.info(f"\tUsed for SFedAvg")
    # logger.info(f"\tiid: {args.iid}")
    # logger.info(f"\tunequal: {args.unequal}")
    # logger.info(f"\tnoise: {args.noise}")
    # logger.info(f"\tnoiselevel: {args.noiselevel}")
    # logger.info(f"\toptimizer: {args.optimizer}")


    logger.info(f"\tUsed for FedCSS")
    logger.info(f"\tvalidation_num: {args.validation_num}")


    logger.info(f"\tUsed for FedDELTA")
    logger.info(f"\tcsd_importance: {args.csd_importance}")



    logger.info(f"\tUsed for FedCorr")
    logger.info(f"\tiid: {args.iid}")
    logger.info(f"\tlevel_n_system: {args.level_n_system}")
    logger.info(f"\talpha_dirichlet: {args.alpha_dirichlet}")
    logger.info(f"\tmixup: {args.mixup}")
    logger.info(f"\tbeta: {args.beta}")
    logger.info(f"\tbegin_sel: {args.begin_sel}")
    logger.info(f"\twithout_CR: {args.without_CR}")
    logger.info(f"\twithout_alternative_update: {args.without_alternative_update}")
    logger.info(f"\tmax_beta: {args.max_beta}")
    logger.info(f"\tplr: {args.plr}")
    logger.info(f"\tnon_iid_prob_class: {args.non_iid_prob_class}")
    logger.info(f"\titeration1: {args.iteration1}")















def normalize_tensor(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

