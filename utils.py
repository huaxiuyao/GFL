from typing import Tuple, List

import torch
import torch.nn.functional as F
import ipdb
from sklearn.metrics import f1_score

def pairwise_loss(u_batch, i_batch_p, i_batch_n, hid_d):
    u_batch = u_batch.view(len(u_batch), 1, hid_d)
    i_batch_p = i_batch_p.view(len(i_batch_p), hid_d, 1)
    i_batch_n = i_batch_n.view(len(i_batch_n), hid_d, 1)

    out_p = torch.bmm(u_batch, i_batch_p)
    out_n = - torch.bmm(u_batch, i_batch_n)

    # sum_p = F.logsigmoid(out_p)
    # sum_n = F.logsigmoid(out_n)
    # loss_sum = - (sum_p + sum_n)

    loss_sum = - F.logsigmoid(out_p + out_n)
    loss_sum = loss_sum.sum() / len(loss_sum)

    return loss_sum


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def F1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    macro_f1 = torch.tensor(f1_score(labels.cpu().data.numpy(), preds.cpu().data.numpy(), average='macro')).cuda()
    micro_f1 = torch.tensor(f1_score(labels.cpu().data.numpy(), preds.cpu().data.numpy(), average='micro')).cuda()
    return micro_f1

def accuracy_nn(preds, labels):
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train_accuracy_multilabel(output, labels, idx):
    preds = output.max(1)[1]
    correct = 0
    total_num = 0
    for i in range(len(idx)):
        total_num += 1
        index = idx[i]
        if preds[index] == labels[i]:
            correct += 1

    return correct, total_num


def accuracy_multilabel(output, labels, idx):
    # print output.size()
    preds = output.max(1)[1]

    correct = 0
    total_num = 0
    for i in range(len(idx)):
        total_num += 1
        index = idx[i]
        if preds[index] in labels[i]:
            correct += 1

    return correct, total_num
