import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import math
import torch.optim as optim

def normalize_perturbation(perturbation, p):
    if p in ['inf', 'linf', 'Linf']:
        return perturbation.sign()
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        bs = perturbation.shape[0]
        pert_flat = perturbation.view(bs, -1)
        pert_normalized = F.normalize(pert_flat, p=2, dim=1)
        return pert_normalized.view_as(perturbation)
    else:
        raise NotImplementedError('Normalization only supports l2 and inf norm')


def project_perturbation(perturbation, eps, p):
    if p in ['inf', 'linf', 'Linf']:
        pert_normalized = torch.clamp(perturbation, -eps, eps)
        return pert_normalized
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        pert_normalized = torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
        return pert_normalized
    else:
        raise NotImplementedError('Projection only supports l2 and inf norm')


def reduce(loss, reduction) :
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError('reduction not supported')

#############################################iterative PGD attack
def logits_diff_loss(out, y_oh, reduction='mean'):
    #out: model output
    #y_oh: targets in one hot encoding
    #confidence:
    out_real = torch.sum((out * y_oh), 1)
    out_other = torch.max(out * (1. - y_oh) - y_oh * 1e13, 1)[0]

    diff = out_other - out_real

    return reduce(diff, reduction)

def conf_diff_loss(out, y_oh, reduction='mean'):
    #out: model output
    #y_oh: targets in one hot encoding
    #confidence:
    confidences = F.softmax(out, dim=1)
    conf_real = torch.sum((confidences * y_oh), 1)
    conf_other = torch.max(confidences * (1. - y_oh) - y_oh * 1e13, 1)[0]

    diff = conf_other - conf_real

    return reduce(diff, reduction)


###################################
def create_early_stopping_mask(out, y, conf_threshold, targeted):
    finished = False
    conf, pred = torch.max(torch.nn.functional.softmax(out, dim=1), 1)
    conf_mask = conf > conf_threshold
    if targeted:
        correct_mask = torch.eq(y, pred)
    else:
        correct_mask = (~torch.eq(y, pred))

    mask = 1. - (conf_mask & correct_mask).float()

    if sum(1.0 - mask) == out.shape[0]:
        finished = True

    mask = mask[(..., ) + (None, ) * 3]
    return finished, mask
