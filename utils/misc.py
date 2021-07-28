# -*- coding: utf-8 -*-
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from fvcore.nn import sigmoid_focal_loss
from datetime import timedelta
from sklearn.metrics import precision_recall_curve

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def max_f_measure(y_true, y_scores):
  precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
  
  f_measure = (2 * (precision * recall) / (precision + recall + 1e-5)).max()
  
  return f_measure

def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
  
class Focal_loss(nn.Module):
    """
    Pytorch implementation from https://github.com/richardaecn/class-balanced-loss
    Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
        labels: A float32 tensor of size [batch, num_classes].
        logits: A float32 tensor of size [batch, num_classes].
        alpha: A float32 tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
        gamma: A float32 scalar modulating loss from hard and easy examples.
    Returns:
        focal_loss: A float32 scalar representing normalized total loss.
    """
    def __init__(self, gamma=0):
        super().__init__()
        self.cross_entropy = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma
    def forward(self, logits, labels, pos_weight=1, neg_weight=1):
        ce =  self.cross_entropy(logits, labels)   
        alpha = labels*pos_weight + (1-labels)*neg_weight
        # A numerically stable implementation of modulator.
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels * logits - self.gamma *
                                  torch.log1p(torch.exp(-1.0 * logits)))

        loss = modulator * ce

        weighted_loss = alpha * loss
        focal_loss = torch.mean(weighted_loss)
        # Normalize by the total number of positive samples.
        #focal_loss /= torch.sum(labels) # Da divisiones por cero
        
        return focal_loss

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, train_rule=None, s=30):
        super(LDAMLoss, self).__init__()
        self.cls_num_list = cls_num_list
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.train_rule = train_rule
        
        #self.cross_entropy = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, x, target, epoch):
        #index = torch.cat([1-target, target], dim=1)
        index = target
        index = index.to(dtype=torch.uint8)
        # index = torch.zeros_like(x, dtype=torch.uint8)
        # index.scatter_(1, target.data.view(-1, 1), 1)
        
        # index_float = index.type(torch.cuda.FloatTensor)
        # batch_m = torch.matmul(self.m_list[None, :].transpose(0,1), index_float.transpose(0,1))
        # batch_m = batch_m.view((-1, 1)).transpose(0,1)
        # x_m = x - batch_m
        
        # index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index.float().transpose(0,1))
        batch_m = batch_m.view((-1, 1)).type(x.dtype)
        x_m = x - batch_m
        
        weight = define_weights(self.cls_num_list, epoch, self.train_rule).cuda()
        weight = weight.to(dtype=torch.float16)
        
        output = torch.where(index, x_m, x)
        
        return F.cross_entropy(self.s*output, target[:,1].squeeze_().long(), weight=weight)
    
def define_weights(cls_num_list, epoch=None, train_rule=None):
    if train_rule == 'None':
        train_sampler = None  
        per_cls_weights = None 
    elif train_rule == 'Resample':
        train_sampler = ImbalancedDatasetSampler(train_dataset)
        per_cls_weights = None
    elif train_rule == 'Reweight':
        train_sampler = None
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    elif train_rule == 'DRW':
        train_sampler = None
        idx = int(epoch >= 10) 
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    
    return per_cls_weights
        
class Logger():
    def __init__(self, file_name):
        self.file_name = file_name
        self.start = time.time()
    
    def write(self, message, mode='a'):
        elapsed = timedelta(seconds=time.time() - self.start)
        message = str(message) + f'\t| Elapsed time: {elapsed}\n'
        with open(self.file_name, mode) as  file:
            file.write(message)
            
def compute_overfit(val_loss, train_loss, prev_val_loss, prev_train_loss):
        new_O = val_loss - train_loss
        prev_O = prev_val_loss - prev_train_loss
        return new_O - prev_O

def compute_gen(val_loss, prev_val_loss):
    return val_loss - prev_val_loss

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

    
class LossConstructor(nn.Module):
    def __init__(self, name='bce', args_dict={}):
        super().__init__()
        self.args_dict = args_dict 
        self.name = name
        if self.name == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=eval(self.args_dict['pos_weight'])) #torch.tensor(758/72)) o 793/77
        elif (name == 'focal') or (name == 'effective'):
            self.criterion = sigmoid_focal_loss
        elif name == 'ldam':
            self.criterion = LDAMLoss(cls_num_list=self.args_dict['cls_num_list'],
                                      max_m=self.args_dict['max_m'], 
                                      s=self.args_dict['s'],
                                      train_rule=self.args_dict['train_rule'])
    
    def forward(self, output, target, epoch=None):
        if (self.name == 'bce'):
            loss = self.criterion(output, target)  
        if (self.name == 'ldam'):
            loss = self.criterion(output, target, epoch)     
        elif self.name == 'focal':
            loss = self.criterion(output, target, alpha=self.args_dict['alpha'],
                                  gamma=self.args_dict['gamma'], 
                                  reduction=self.args_dict['reduction'])
        elif self.name == 'effective':
            #DRW optimizing scheduler from LDAM loss paper
            train_sampler = None
            idx = epoch // 10
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], 
                                           eval(self.args_dict['cls_num_list']))
            weights = (1.0 - betas[idx]) / np.array(effective_num)
            alpha = weights[0] / np.sum(weights)
            
            #Effective number of samples
            # effective_num = 1.0 - np.power(self.args_dict['beta'], np.array([72, 758]))
            # weights = (1.0 - self.args_dict['beta']) / np.array(effective_num)
            # alpha = weights[0] / np.sum(weights)
            
            #Loss
            loss = self.criterion(output, target, alpha=alpha,
                                  gamma=self.args_dict['gamma'], 
                                  reduction=self.args_dict['reduction'])
        
        return loss

class OptimizerConstructor(nn.Module):
    def __init__(self, name='Adam', args_dict={}):
        super().__init__()
        self.args_dict = args_dict 
        self.name = name
    
    def forward(self, params):
        if self.name == 'Adam':
            optimizer = optim.Adam(params, lr=eval(self.args_dict['lr']),
                                weight_decay=self.args_dict['weight_decay'], 
                                amsgrad=self.args_dict['amsgrad'])
        if self.name == 'SGD':
            optimizer = optim.SGD(params, lr=eval(self.args_dict['lr']), 
                                  weight_decay=self.args_dict['weight_decay'], 
                                  momentum=self.args_dict['momentum'])
        return optimizer
               
if __name__ == '__main__':
    out = torch.tensor([-1.45, 0.6, -2000., 1.])
    target = torch.tensor([1., 1., 1., 1.])
    loss_class = Focal_loss(gamma=0.5)
    loss = loss_class(out, target)
    print(loss)