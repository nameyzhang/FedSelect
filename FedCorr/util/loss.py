from typing import Optional
from torch import Tensor
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
from FedCorr.util.optimizer import filter_noisy_data, f_beta
from torch.nn import CrossEntropyLoss


class CORESLoss(CrossEntropyLoss):
    r"""
    Examples::
        >>> # Example of target with class indices
        >>> loss = CORESLoss()
        >>> beta = 0
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss.forward(input, target, beta)
        >>> output.backward()
        >>>
        >>> # Example of target with class probabilities
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).softmax(dim=1)
        >>> output = loss.forward(input, target, beta)
        >>> output.backward()
    """
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, beta, noise_prior=None) -> Tensor:
        # beta = f_beta(epoch)
        # if epoch == 1:
            # print(f'current beta is {beta}')
        loss = F.cross_entropy(input, target, reduction=self.reduction) # crossentropy loss
        loss_ = -torch.log(F.softmax(input, dim=1) + 1e-8)
        if noise_prior is None:
            loss = loss - beta * torch.mean(loss_, 1)  # CORESLoss
        else:
            loss = loss - beta * torch.sum(torch.mul(noise_prior, loss_), 1)
        loss_ = loss
        return loss_


class CrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super(CrossEntropyLoss, self).__init__(*args, **kwargs)
    def forward(self, input, target, *args, **kwargs):
        return super(CrossEntropyLoss, self).forward(input, target)

class FedTwinCRLoss(CrossEntropyLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input_p, input_g, target, rounds, epoch, args, noise_prior=None):
        if args.without_CR:
            loss = CrossEntropyLoss(reduction='none')
        else:
            loss = CORESLoss(reduction='none')
        Beta = f_beta(rounds * args.local_epochs + epoch, args)
        if rounds <= args.begin_sel:  # 如果在前30epoch集中式，对应联邦应该是30/local_epoch
            loss_p_update = loss(input_p, target, Beta, noise_prior)
            loss_g_update = loss(input_g, target, Beta, noise_prior)

            ind_g_update = Variable(torch.from_numpy(np.ones(len(loss_p_update)))).bool()
        else:
            ind_p_update = filter_noisy_data(input_p, target)
            ind_g_update = filter_noisy_data(input_g, target)
            if args.without_alternative_update:
                ind_p_update, ind_g_update = ind_g_update, ind_p_update
        
            loss_p_update = loss(input_p[ind_g_update], target[ind_g_update], Beta, noise_prior)
            loss_g_update = loss(input_g[ind_p_update], target[ind_p_update], Beta, noise_prior)
        loss_batch_p = loss_p_update.data.cpu().numpy() # number of batch loss1
        loss_batch_g = loss_g_update.data.cpu().numpy()  # number of batch loss1
        if len(loss_batch_p) == 0.0:
            loss_p = loss(input_p, target, Beta, noise_prior)
            loss_p = torch.mean(loss_p) / 100000000
        else:
            loss_p = torch.sum(loss_p_update) / len(loss_batch_p)
        if len(loss_batch_g) == 0.0:
            loss_g = loss(input_g, target, Beta, noise_prior)
            loss_g = torch.mean(loss_g) / 100000000
        else:
            loss_g = torch.sum(loss_g_update) / len(loss_batch_g)
        return loss_p, loss_g, len(loss_batch_p), len(loss_batch_g)
