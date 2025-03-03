from torch import nn
from sklearn.metrics import f1_score


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    bs = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / bs))
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








