import torch
import datetime
import os
import logging

def get_device(args):
    device = torch.device(
        'cuda:{}'.format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else 'cpu'
    )

    return device


def logging_config(logger, args):
    logger.info("--------------------------------------------------------")
    logger.info(f"Config:")
    logger.info(f"\ttraining_time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info(f"\tgpu: {args.gpu}")
    logger.info(f"\talgorithm: {args.algorithm}")
    logger.info(f"\tdataset_name: {args.dataset_name}")
    logger.info(f"\tmodel: {args.model}")

    logger.info(f"\tselect_client: {args.select_client}")
    logger.info(f"\tclient_num: {args.client_num}")
    logger.info(f"\tclient_select_ratio: {args.select_ratio}")

    logger.info(f"\tselect_samples: {args.select_samples}")
    logger.info(f"\tsample_select_ratio:{args.select_samples_ratio}")


    logger.info(f"\tcorruption_type: {args.corruption_type}")
    logger.info(f"\tcorrupt_num: {args.corrupt_num}")
    logger.info(f"\tcorruption_prob: {args.corruption_prob}")


    logger.info(f"\tnoniid_ratio: {args.noniid_ratio}")
    logger.info(f"\tnoniid_class_ratio: {args.noniid_class_ratio}")

    logger.info(f"\tepochs: {args.epochs}")
    logger.info(f"\tlocal_epochs: {args.local_epochs}")
    logger.info(f"\tbatch_size: {args.batch_size}")
    logger.info(f"\tlr: {args.lr}")
    logger.info(f"\tmomentum: {args.momentum}\n")
    logger.info(f"\tseed: {args.seed}")

    logger.info(f"\twarmup_epochs: {args.warmup_epochs}")
    logger.info(f"\treward_data_size: {args.reward_data_size}")
    logger.info(f"\tmeta_momentum: {args.meta_momentum}")
    logger.info(f"\tmain_path: {args.main_path}")




def normalize_tensor(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

