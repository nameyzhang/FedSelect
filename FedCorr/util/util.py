import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial.distance import cdist
from utils.utils import get_device

def add_noise(args, y_train, dict_users):   # 分别为: 参数对象; 训练数据的标签; 每个用户对应的训练数据索引字典
    # np.random.seed(args.seed)

    #         np.random.binomial(n, p, size), 其中 n 表示一次训练的样本数, p 表示事件发生的概率, size: 限定了返回值的形式和试验次数
    gamma_s = np.random.binomial(1, args.level_n_system, args.client_num)       # level_n_system: fraction of noisy clients; 生成一个长度为 args.num.users 的数组 gamma_s, 每个元素是一个伯努利随机变量, 成功概率为 args.level_n_system; 这表示每个客户端是否收到系统级噪声影响
    gamma_c_initial = np.random.rand(args.client_num)         # 生成一个长度为 args.client_num 的随机数组 gamma_c_initial, 初始值在 [0, 1] 范围内
    gamma_c_initial = (1 - args.corruption_prob) * gamma_c_initial + args.corruption_prob       # 将初始值调整为在 [ args.level_n_lowerb, 1) 范围内, 确保噪声水平不低于某个下界
    gamma_c = gamma_s * gamma_c_initial        # 每个客户端的最终噪声水平: 系统级噪声(有多少client受到噪声干扰) * 客户端级噪声初始值(每个client的干扰程度, 大于level_n_lowerb)
    # level_n_system=0,


    y_train_noisy = copy.deepcopy(y_train)     # 在不修改原始标签数据的情况下添加噪声

    real_noise_level = np.zeros(args.client_num)    # 创建一个长度为 args.client_num 的数组 real_noise_level, 用于存储每个客户端实际添加的噪声比例

    for i in np.where(gamma_c > 0)[0]:         # 为每一个受影响的客户端添加噪声 (遍历所有噪声水平大于0的客户端); np.where(gamma_c > 0) 返回一个元组, 其中包含满足条件 gamma_c > 0 的索引; [0] 用来提取元组中的第一个元素, 即包含满足条件的索引的数组;
        sample_idx = np.array(list(dict_users[i]))      # 获取客户端 i 的样本索引
        prob = np.random.rand(len(sample_idx))          # 生成一个长度为 sample_idx 数组的随机数数组 prob, 其中每个元素在 [0, 1) 范围内

        noisy_idx = np.where(prob <= gamma_c[i])[0]     # 找到满足 prob <= gamma_c[i] 条件的索引 noisy_idx, 这些索引对应的样本将添加噪声

        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))    # 对于 noisy_idx 中的样本，将其标签替换为 [0, 10) 之间的随机整数
        noise_ratio = np.mean( y_train[sample_idx] != y_train_noisy[sample_idx] )          # y_train[sample_idx] != y_train_noisy[sample_idx]: 生成一个布尔数组, 表示原始标签和噪声标签不想等的位置; np.mean() 计算这些位置的平均值, 即噪声比例 noise_ratio

        # i 客户端的编号; 客户端 i 的噪声水平; 调整后的噪声水平（假设乘以0.9）; 实际噪声比例
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))

        real_noise_level[i] = noise_ratio
    # By comparing the labels of the dataset before and after adding noise
    # we can determine which samples have been affected by the noise
    noisy_samples_idx = np.where(y_train != y_train_noisy)[0]

    print("add_noise ok")
    # 添加噪声后的的所有标签数据 y_train_noisy;
    # 系统级噪声标记(哪些客户点受噪声影响) gamma_s, 是一个数组;
    # 实际噪声水平(客户级) real_noise_level, 是一个数字;
    # 被噪声影响的样本索引 noisy_samples_idx;
    return y_train_noisy, gamma_s, real_noise_level, noisy_samples_idx


def get_output(loader, net, args, latent=False, criterion=None):


    device = get_device(args)

    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.long()
            if latent == False:
                # outputs, _ = net(images)
                outputs = F.softmax(net(images), dim=1)
            else:
                outputs, _ = net(images, True)
            loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    distances_ = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids

#
# if rnd == 1 or rnd == 20 or rnd == 50 or rnd == (args.rounds2 - 1):
#     # Record the loss for clean and noisy samples separately
#     clean_loss_s, noisy_loss_s = get_clean_noisy_sample_loss(
#         model=netglob.to(args.device),
#         dataset=dataset_train,
#         noisy_sample_idx=noisy_sample_idx,
#         round=rnd,
#         device=args.device,
#         beta=Beta
#     )
#     print(f"{loss_fn.__class__.__name__}:")
#     print("clean_loss:")
#     print(clean_loss_s)
#     print("noisy_loss:")
#     print(noisy_loss_s)