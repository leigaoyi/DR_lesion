# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 09:34:55 2020

@author: kasy
"""

import os
import json
import argparse
import torch
import numpy as np

import torch.nn.functional as F
from data import DR_EX_test

import cv2

from torch.utils.data import DataLoader

from models import UNet, FCN8, PSPNet
from models import UperNet

from models.a_unet import A_UNet


config_path = './config.json'
config = json.load(open(config_path))


class_list = ['EX', 'HE', 'MA', 'SE']


model_name = config['predict']['pred_model']
step_num = config['predict']['reuse_step']


class_name = class_list[0]

print('Predict {0} with epoch {1}'.format(model_name, step_num))
result_dir = './result/{0}/{1}/'.format(model_name, class_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

if model_name == 'UNet':
    model = UNet(2, in_channels=3)

if model_name == 'FCN8':
    model = FCN8(2)
    
if model_name == 'PSPNet':
    model = PSPNet(2)
    
if model_name == 'UperNet':
    model = UperNet(2)
    
if model_name == 'CC_UNet':
    model = CC_UNet(2)

if model_name == 'A_UNet':
    model = A_UNet(2)

device = torch.device('cuda:0')

state_dict = torch.load('./checkpoints/{0}_{1}.pth'.format(model_name, step_num))

model.load_state_dict(state_dict)
model.to(device)
model.eval()
print('Load over!')

data_dir = config['predict']['data_dir']
label_dir = config['predict']['label_dir']

pred_simple = config['predict']['simple_type']

DR_dataset = DR_EX_test(data_dir, label_dir)
DR_loader = DataLoader(DR_dataset, batch_size=1, shuffle=False, num_workers=5)

if not os.path.exists('./result/'):
    os.makedirs('./result/')

def recover(fig_list, shape, crop_size=512):
    w_int, h_int = shape[0]//crop_size, shape[1]//crop_size
    
    fig_zeros = np.zeros([(w_int+1)*crop_size, (h_int+1)*crop_size])
    
    for i in range(w_int+1):
        for j in range(h_int+1):
            fig_zeros[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size] = fig_list[i*(h_int+1)+j]
    fig_output = fig_zeros[:shape[0], :shape[1]]
    return fig_output


current_step = 0
for idx, (data, label, shape, label_name) in enumerate(DR_loader):
    current_step += 1
    
    shape = (shape[0].cpu().item(), shape[1].cpu().item())
    #print(shape)
    print(label_name[0])
    data = np.transpose(np.squeeze(data), [0, 3, 1, 2])
    label = np.squeeze(label)
    
    
    data = data.to(device, dtype=torch.float)
    label = label.to(device, dtype=torch.float)
    #print(data.shape)
    with torch.no_grad():
        output_list = []
        for i in range(data.shape[0]):
            
            output = model(data[i, ...][np.newaxis, ...])
            if model_name == 'CC_UNet':
                output = F.softmax(output[0], dim=1)
            else:
                output = F.softmax(output, dim=1)
            #print(output.shape)
            output_fig = output.cpu().numpy()
            
            output_fig = output_fig[:, 1, :, :] * 255.
            output_fig = np.squeeze(output_fig)
            output_list.append(output_fig)
        
    output_all = recover(output_list, shape)
    label_all = recover(label.cpu().numpy(), shape)
    
    cv2.imwrite(os.path.join(result_dir, 'label_{0}.jpg'.format(label_name[0][:-4])), label_all*255.)        
    cv2.imwrite(os.path.join(result_dir, 'pred_{0}.jpg'.format(label_name[0][:-4])), output_all)
    #print(current_step)

    if pred_simple:
        if current_step == 4:
            break

