from itertools import repeat
from re import X
#from typing_extensions import Self

import torch
from torch.utils.data import Dataset

'''
这里的torch.utils.data.Dataset，这个类就是一个abstract class，就是用来被继承的
我们都需要构建自己的Dataset类，都需要继承上面的Dataset类，并且常常需要重写 __len__()和__getitem__()两个函数
关于overwrite这两个函数，是官方文档里建议的
所以重写的Dateset class里基本要实现 __init__() __len__() __getitem__()这三个函数 
'''

class KnowledgeGraphDataset(Dataset):
    def __init__(self, x, y, e2index, r2index):
        self.x = x
        self.y = y
        self.e2index = e2index
        self.r2index = r2index
        
        assert len(x) == len(y)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, item):
        s, r = self.x[item]


# About the collate_fn: 
# merges a list of samples to form a mini-batch of Tensor(s).  Used when using batched loading from a map-style dataset.
def collate_train(batch):
    max_len = max(map(lambda x: len(x[2]), batch))
    
    for _, _, indices in batch:
        indices.extend(repeat(indices[0], max_len - len(indices)))
    
    s, o, i = zip(*batch)
    return torch.LongTensor(s), torch.LongTensor(o), torch.LongTensor(i)
    
def collate_valid(batch):
    s, o, i = zip(*batch)
    return torch.LongTensor(s), torch.LongTensor(o), torch.LongTensor(i)

