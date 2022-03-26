from ast import parse
import os
import pickle
import argparse

import logging

import shutil
from numpy import size
from sklearn.covariance import empirical_covariance
import torch.nn as nn 
import torch

from tensorboard_logger import tensorboard_logger

from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader 
from tqdm import tqdm, trange

from dataset import KnowledgeGraphDataset, collate_valid
import dataset
from model import ConvE
from util import AttributeDict

logger = logging.getLogger(__file__)

class StableBCELoss(nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()
    
    # 这里使用的BCELoss二分类交叉熵损失函数
    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()        


def train(epoch, data, conv_e, criterion, optimizer, args):
    train_set = DataLoader(
        KnowledgeGraphDataset(data.x, data.y, e2index=data.e2index, r2index=data.r2index),
        collate_fn=dataset.collate_train, batch_size=args.batch_size, num_workers=4, shuffle=True
    )

    progress_bar = tqdm(iter(train_set))
    moving_loss = 0
    
    conv_e.train(True)
    y_multihot = torch.LongTensor(args.batch_size, len(data.e2index))
    for s, r, os in progress_bar:
        s, r = Variable(s).cuda(), Variable(r).cuda()
        
        if s.size()[0] != args.batch_size:
            y_multihot = torch.LongTensor(s.size()[0], len(data.e2index))  
        
        y_multihot.zero_()
        
        y_multihot = y_multihot.scatter_(1, os, 1)
        y_smooth = (1 - args.label_smooth) * y_multihot.float() +  args.label_smooth / len(data.e2index)

        targets = Variable(y_smooth, requires_grad=False).cuda()
        
        output = conv_e(s, r)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        conv_e.zero_grad()
        
        if moving_loss == 0:
            moving_loss = loss.item()
        else:
            moving_loss = moving_loss * 0.9 + loss.item() * 0.1
        
        progress_bar.set_description(
            'Epoch: {}; Loss: {:.5f}; Avg: {:.5f}'.format(epoch + 1, loss.item(), moving_loss)
        )
    
    logger.info('Epoch: {}; Loss: {:.5f}; Avg: {:.5f}'.format(epoch + 1, loss.item(), moving_loss))

    tensorboard_logger.log_value('avg_loss', moving_loss, epoch + 1)
    tensorboard_logger.log_value('loss', loss.item(), epoch + 1)

def valid(epoch, data, conv_e, batch_size, log_decs):
    dataset = KnowledgeGraphDataset(data.x, data.y, e2index=data.e2index, r2index=data.r2index)
    valid_set = DataLoader(dataset, collate_fn=collate_valid, batch_size=batch_size, num_workers=4, shuffle=True)

    conv_e.train(True)
    ranks = list()
    
    for s, r, os in tqdm(iter(valid_set)):
        s, r = Variable(s).cuda(), Variable(r).cuda()
        
        output = conv_e.test(s, r)
        # 这里就和train不同一些，train要进行loss的计算，所以还要处理objects数据集，并将output预测结果和objects对比

        # 下面就是Hitk的计算，将outputs中和objects答案对比，有多少命中的
        for i in range(min(batch_size, s.size()[0])):
            _, top_indices = output[i].topk(output.size()[1])
            for o in os[i]:
                _, rank = (top_indices == o).max(dim=0)
                ranks.append(rank.item() + 1)
        
    ranks_t = torch.FloatTensor(ranks)
    mr = ranks_t.mean()
    mrr = (1 / ranks_t).mean()
    
    logger.info(log_decs + 'MR: {:.3f}, MRR: {:.10f}'.format(mr, mrr))
    tensorboard_logger.log_value(log_decs + ' mr', mr, epoch + 1) 
    tensorboard_logger.log_value(log_decs + ' mrr', mrr, epoch + 1)
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='ConvE Training by PyTorch')
    parser.add_argument('--train_path', type=str, help='Path to traing file(pkl file) produced by preprocess.py')    
    parser.add_argument('--valid_path', type=str, help='Path to valid/test file(pkl file) produced by preprocess.py')
    parser.add_argument('--name', type=str, default='', help='name of the saved mode, used to save checkpoints')
    parser.add_argument('--batch-size', type=int, default=256, dest='batch_size')
    parser.add_argument('--epochs', type=int, default=100, dest='epochs')
    parser.add_argument('--label-smooth', type=float, default=0.1, dest='label_smoothing')
    parser.add_argument('--log-file', type=str)

    return parser.parse_args()

def setup_logger(args):
    log_file = args.log_file
    tensorboard_log_dir = 'tensorboard_' + args.name
    shutil.rmtree(tensorboard_log_dir)
    if args.log_file is None:
        if args.name == '':
            log_file = 'train.log'
        else:
            log_file = args.name + '.log'
        
    print('Logging to: ' + log_file)
    
    logging.basicConfig(filename=log_file, level=logging.INFO)
    tensorboard_logger.configure(tensorboard_log_dir)

def main():
    
    args = parse_args()
    setup_logger(args)
    
    checkpoint_path = 'checkpoint-{}'.format(args.name)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    with open(args.train_path, 'rb') as f:    
        train_data = AttributeDict(pickle.load(f))
    with open(args.valid_path, 'rb') as f:
        valid_data = AttributeDict(pickle.load(f))
        
    valid_data.e2index = train_data.e2index
    valid_data.index2e = train_data.index2e
    valid_data.r2index = train_data.r2index
    valid_data.index2r = train_data.index2r
    
    conv_e = ConvE(num_e=len(train_data.e2index), num_r=len(train.r2index)).cuda()
    criterion = StableBCELoss()
    optimizer = optim.Adam(conv_e.parameters(), lr = 0.003)
    
    for epoch in trange(args.epochs):
        train(epoch, train_data, conv_e, criterion, optimizer, args)
        valid(epoch, train_data, conv_e, args.batch_size, 'train')
        valid(epoch, valid_data, conv_e, args.batch_size, 'valid')
        
        with open('{}/checkpoint_{}.model'.format(checkpoint_path, str(epoch + 1).zfill(2)), 'wb') as f:
            torch.save(conv_e, f)        
    
if __name__ == '__mian__':
    main()        
    
    
    
    
    
    
    







