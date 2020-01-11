import logging

import numpy as np
from utils.torchsummary import summary
import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Nbr of trainable parameters: {0}'.format(nbr_params))

    @property
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + '\nNbr of trainable parameters: {0}'.format(nbr_params)
        #return summary(self, input_shape=(2, 3, 224, 224))
        
    def save_network(self, network, network_label, iter_label):
        save_filename = './checkpoints/{}_{}.pth'.format(iter_label, network_label)
        if not os.path.exists('./checkpoints/'):
            os.makedirs('./checkpoints/')
            
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
