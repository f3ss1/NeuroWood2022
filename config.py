import torch

class Config:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 3
    SIZE = 512