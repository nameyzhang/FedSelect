from torch import nn
from sklearn.metrics import f1_score


def accuracy(output, target, topk=(1,)):     # topk: 一个元祖, 指定要计算的 k 值, 默认值为 (1, ), 表示计算 top-1 精度
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)             # 获取 topk 中的最大值 k, 用于确定需要选择的最大预测数目;
    bs = target.size(0)          # 获取批次大小 (即样本数量); target.size(0) 返回目标张量的第一个维度大小;

    _, pred = output.topk(maxk, 1, True, True)                # 从 output 中获取每行的前 maxk 个最大的值及其索引; maxk 要获取的最大值的个数; 1 表示要沿着第1个维度 (每行) 进行操作; True 表示返回排序后的值; True 表示返回排序后的索引;
    pred = pred.t()                                           # 对预测的索引进行转置, 使其形状从 [bs, maxk] 变为 [maxk, bs]
    correct = pred.eq(target.view(1, -1).expand_as(pred))     # 将 target 转换为 [1, bs] 形状, 并扩展为与 pred 形状相同的张量 [maxk, bs]; pred.eq() 比较预测的索引 pred 和目标标签 target, 生成一个布尔张量 correct, 表示预测是否正确;

    res = []        # 初始化一个空列表, 用于存储每个 k 的精度;
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)       # correct[:k] 获取前 k 行的预测结果; view(-1) 将结果展平为一维张量; sum(0) 计算所有正确预测的数量;
        res.append(correct_k.mul_(100.0 / bs))                # 将正确的数量除以批次大小 bs, 得到正确预测的比例并乘以100, 转换为百分比后, 添加到结果列表 res 中;
    return res





def f1_score(output, target, threshold=0.5):
    """
    Computes the F1 score for binary classification.

    Args:
        output (torch.Tensor): The output predictions from the model (probabilities).
        target (torch.Tensor): The ground truth labels.
        threshold (float): The threshold for converting probabilities to binary predictions.

    Returns:
        float: The F1 score.
    """
    output = output.to(target.device)

    # Convert probabilities to binary predictions
    pred = (output > threshold).float()

    # True Positives (TP), False Positives (FP), False Negatives (FN)
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()

    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1








