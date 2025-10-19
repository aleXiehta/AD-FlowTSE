from asteroid.losses import singlesrc_neg_sisdr
import torch


def neg_sisdr_loss_wrapper(est_targets, targets):
    return singlesrc_neg_sisdr(est_targets, targets).mean()

def mse_loss(est_targets, targets):
    return torch.mean((est_targets - targets) ** 2)