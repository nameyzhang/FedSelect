from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
# user model to calc loss of samples in dataset by loss_fn
# def get_loss(model, loss_fn, dataset, *args, **kwargs):
#     loss_list = []
#     data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
#
#     for batch in data_loader:
#         features, labels = batch
#         output = model(features)[0]
#         loss = loss_fn(output, labels, *args, **kwargs)
#         loss_list += loss.tolist()
#
#     return loss_list


def get_loss(model, dataset, device, *args, **kwargs):
    loss_list = []
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    for batch in data_loader:
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        output, _ = model(features)
        loss = F.cross_entropy(output, labels, reduction='none')
        loss_ = -torch.log(F.softmax(output, dim=1) + 1e-8)
        loss_sel = loss - torch.mean(loss_, 1)
        loss_list += loss_sel.tolist()
    return loss_list


# noisy_sample_idx is a 2d list
# noisy_sample_idx[i] is the noisy sample idx of user <i>, e.g. [0,1,2,8] 
def split_clean_noisy_loss(loss_s, noisy_sample_idx):
    clean_loss_s = []
    noisy_loss_s = []

    # Iterate through each sample in loss_s
    for sample in range(len(loss_s)):
        if sample in noisy_sample_idx:
            noisy_loss_s.append(loss_s[sample])
        else:
            clean_loss_s.append(loss_s[sample])

    return clean_loss_s, noisy_loss_s


def get_clean_noisy_sample_loss(model, dataset, noisy_sample_idx, round, device, *args, **kwargs):
    loss_s = get_loss(model, dataset, device, *args, **kwargs)
    clean_loss_s, noisy_loss_s = split_clean_noisy_loss(loss_s, noisy_sample_idx)

    return clean_loss_s, noisy_loss_s
