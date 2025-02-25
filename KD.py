from torch.nn.modules import loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
import math


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss


def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

# def KD_loss_func(criterion, teacher_logits, student_logits, labels, alpha, temp, mode):
#     if mode == 'cse':
#         teacher_dist = F.log_softmax(teacher_logits / temp, dim=1)
#         student_dist = F.softmax(student_logits / temp, dim=1).detach()
#         soft_loss = -torch.mean(torch.sum(teacher_dist * student_dist, dim=1))
#     elif mode == 'mse':
#         teacher_dist = F.log_softmax(teacher_logits / temp, dim=1)
#         student_dist = F.softmax(student_logits / temp, dim=1).detach()
#         soft_loss = nn.MSELoss()(teacher_dist, student_dist) / 2

#     soft_loss = soft_loss * temp ** 2
#     hard_loss = criterion(student_logits, labels)
#     total_loss = alpha * soft_loss + (1 - alpha) * hard_loss

#     return total_loss


def KD_loss_func(criterion, logits, target_logits, labels, alpha, temp, mode):
    if mode == 'cse':
        teacher_dist = F.log_softmax(logits / temp, dim=1)
        student_dist = F.softmax(target_logits / temp, dim=1).detach()
        soft_loss = -torch.mean(torch.sum(teacher_dist * student_dist, dim=1))
    elif mode == 'mse':
        teacher_dist = F.log_softmax(logits / temp, dim=1)
        student_dist = F.softmax(target_logits / temp, dim=1).detach()
        soft_loss = nn.MSELoss()(teacher_dist, student_dist) / 2

    soft_loss = soft_loss * temp ** 2
    hard_loss = criterion(target_logits, labels)
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss

    return total_loss


def JSD_KD_loss_func(criterion, logits, target_logits, labels, alpha, temp, mode):
    M = (logits + target_logits) / 2

    if mode == 'cse':
        dist = F.log_softmax(logits / temp, dim=1)
        target_dist = F.log_softmax(target_logits / temp, dim=1)
        middle_dist = F.softmax(M / temp, dim=1).detach()

        soft_loss_1 = -torch.mean(torch.sum(dist * middle_dist, dim=1))
        soft_loss_2 = -torch.mean(torch.sum(target_dist * middle_dist, dim=1))
    elif mode == 'mse':
        dist = F.log_softmax(logits / temp, dim=1)
        target_dist = F.log_softmax(target_logits / temp, dim=1)
        middle_dist = F.softmax(M / temp, dist=1)

        soft_loss_1 = nn.MSELoss()(dist, middle_dist) / 2
        soft_loss_2 = nn.MSELoss()(target_dist, middle_dist) / 2

    # soft_loss_1 = soft_loss_1 * temp ** 2
    # soft_loss_2 = soft_loss_2 * temp ** 2
    soft_loss = (soft_loss_1 + soft_loss_2) * temp ** 2

    hard_loss = criterion(M, labels)
    # loss_1 = alpha * soft_loss_1 + (1 - alpha) * hard_loss
    # loss_2 = alpha * soft_loss_2 + (1 - alpha) * hard_loss
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss

    # total_loss = loss_1 + loss_2

    return total_loss


def JSD(p, q):
    kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    m = (0.5 * (p + q)).log()
    return 0.5 * (kl(m, p.log()) + kl(m, q.log()))


# class JSD(nn.Module):
#     def __init__(self):
#         super(JSD, self).__init__()
#         self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

#     def forward(self, p: torch.tensor, q: torch.tensor):
#         p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
#         m = (0.5 * (p + q)).log()
#         return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


def attention_loss_function(attention1, attention2):
    attention_minus = (attention1 - attention2)
    attention_sq = torch.mul(attention_minus, attention_minus)
    attention_total = torch.abs(attention_sq).sum()/1024
    return attention_total


class DistributionLoss(loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher output.

    output must be a pair of (model_output, real_output), both NxC tensors.
    The rows of real_output must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, model_output, real_output):

        self.size_average = True

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        if real_output.requires_grad:
            raise ValueError(
                "real network output should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        real_output_soft = F.softmax(real_output, dim=1)
        del model_output, real_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = - \
            torch.bmm(real_output_soft, model_output_log_prob)
        if self.size_average:
            cross_entropy_loss = cross_entropy_loss.mean()
        else:
            cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        # model_output_log_prob = model_output_log_prob.squeeze(2)
        return cross_entropy_loss
