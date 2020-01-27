# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:46:56 2020

@author: kasy
"""

import os
import json
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import cv2

import random

from loss import dice, cross_entropy
from loss import focal_loss
from loss import weighted_dice, multi_dice
from loss import DiceLoss

from data import DR_EX
from data import DR_EX_test

from torch.utils.data import DataLoader

from models import UNet, FCN8
from models import ENet, PSPNet

from models import UperNet

from models.a_unet import A_UNet

from evaluate import predict_model
from evaluate import cal_f1

#from models import Seg_Model
#from models import CC_UNet

config_path = './config.json'
config = json.load(open(config_path))



data_dir = config['dataset']['train_data']
label_dir = config['dataset']['train_label']

model_name = config['trainer']['model_name']
epoch_num = config['trainer']['epoch']
device_num = config['trainer']['device']
crop_size = config['trainer']['crop_size']


batch_size = config['trainer']['batch_size']
reuse_model = config['trainer']['reuse']
reuse_num = config['trainer']['reuse_num']
reuse_path = './checkpoints/{0}_{1}.pth'.format(model_name, reuse_num)

train_mode = config['trainer']['train_mode']
train_class = config['trainer']['train_class']


device = torch.device('cuda:{}'.format(device_num))

DR_dataset = DR_EX(data_dir, label_dir, train_mode, crop_size)

test_data_dir = config['dataset']['test_data']
test_label_dir = config['dataset']['test_label']

DR_test_dataset = DR_EX_test(test_data_dir, test_label_dir)
DR_test_loader = DataLoader(DR_test_dataset, batch_size=1, shuffle=False, num_workers=6)

#print('Len ', len(DR_dataset))

if config['trainer']['train_mode'] == 'fast':
    print('Load data mode ', config['trainer']['train_mode'])
    DR_loader = DataLoader(DR_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
else:
    DR_loader = DataLoader(DR_dataset, batch_size=batch_size, shuffle=True, num_workers=6)


model = ''
if model_name == 'UNet':
    model = UNet(2, in_channels=3)

if model_name == 'FCN8':
    model = FCN8(2)
    
if model_name == 'PSPNet':
    model = PSPNet(2)
    
if model_name == 'UperNet':
    model = UperNet(2)
    
#if model_name == 'CCNet':
#    model = Seg_Model(2)
#    
#if model_name == 'CC_UNet':
#    model = CC_UNet(2)


if model_name == 'A_UNet':
    model = A_UNet(2)


assert model != ''

start_epoch = 0
current_step = 0


if reuse_model == True:
    state_dict = torch.load(reuse_path)
    model.load_state_dict(state_dict)
    num_data = len(DR_dataset)
    batch_idx = num_data//batch_size
    start_epoch = reuse_num // batch_idx
    current_step = reuse_num
    print('Reuse model :', reuse_path)

model.to(device)

#loss_fn = torch.nn.MSELoss()
#loss_fn = weighted_dice
loss_fn = nn.CrossEntropyLoss(\
    weight=torch.FloatTensor([0.1, 1]).to(device))
#loss_fn = focal_loss
#loss_fn = multi_dice

#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
base_lr = config['trainer']['lr']
# optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.5, 0.999))
# optimizer.zero_grad()
optimizer = torch.optim.SGD(model.parameters(),
                        lr=base_lr,
                        momentum=0.9,
                        weight_decay=0.0005)

model.train()
#torch.set_grad_enabled(True)



def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))
            
def adjust_learning_rate(optimizer, learning_rate, i_iter, max_iter, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

print('Train {0} in {1} type!'.format(model_name, train_mode))

for i in range(start_epoch, epoch_num):
    
    #lr = adjust_learning_rate(optimizer, base_lr, i, epoch_num, power=0.9)
    
    for idx, (data, label) in enumerate(DR_loader):
        current_step += 1
        
        
        data_shape = data.shape #[20, 128, 128, 3]
        label_shape = label.shape # [ 20, 128, 128, 1]

        #print(label.max())
        data = np.transpose(data, [0, 3, 1, 2])
        data = np.reshape(data, [-1, 3, crop_size, crop_size]) # [4*30, 3, 128, 128]
        label = label.view((-1, crop_size, crop_size)) # [4*30, 128, 128]

        #print('data max ', data.max().item(), ' label max ', label.max().item())
        
        data = data.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.float)
        
        #print(data.shape)
        
        if model_name == 'PSPNet':
            output = model(data)[0]

        else:
            output = model(data)

        
        if model_name == 'CC_UNet':
            #loss_v = 0.7*loss_cc(output[0], label) + 0.3 * loss_cc(output[1], label)
            pass
        else:
            label_flat = label.reshape(-1).long()
            pred_model = output.permute(0, 2, 3, 1)
            pred_flat = pred_model.reshape(-1, pred_model.shape[-1])
            loss_v = loss_fn(pred_flat, label_flat)

        optimizer.zero_grad()
        loss_v.backward()
        optimizer.step()
        
        if current_step % 100 == 0:
            print('Epoch {0} Step {1} loss {2:.4f}'.format(i, current_step, loss_v.item()))
            predict_model(model, device, DR_test_loader)
            f1 = cal_f1()
            print('f1 score ', f1)
            
        #print('Output max {0} min {1} ; Label max {2} min{3} '.format(output_v.max(), output_v.min(), label.max(), label.min()))
        
        if (current_step)%1000 == 0:
            print('Saving model and parameters')
            save_path = './checkpoints/{0}_{1}_{2}.pth'.format(model_name, train_class,current_step)
            if not os.path.exists('./checkpoints/'):
                os.makedirs('./checkpoints/')
            state_dict = model.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()
            torch.save(state_dict, save_path)
        


save_path = './checkpoints/{0}_last.pth'.format(model_name)
state_dict = model.state_dict()
torch.save(state_dict, save_path)

