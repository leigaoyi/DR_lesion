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


from loss import dice, cross_entropy
from loss import focal_loss
from loss import weighted_dice
from loss import DiceLoss

from data import DR_EX

from torch.utils.data import DataLoader

from models import UNet


data_dir = './data/images/'
label_dir = './data/ground_truths/EX/'

model_name = 'UNet'
epoch_num = 1000
device = torch.device('cuda:0')


DR_dataset = DR_EX(data_dir, label_dir)
DR_loader = DataLoader(DR_dataset, batch_size=4, shuffle=True, num_workers=4)

model = UNet(2, inchannels=3)
model.to(device)

#loss_fn = torch.nn.MSELoss()
loss_fn = weighted_dice
#loss_fn = cross_entropy
#loss_fn = focal_loss

#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.zero_grad()


model.train()
torch.set_grad_enabled(True)
current_step = 0

sample_data = cv2.imread('./tmp/sample_data.jpg')
sample_data = sample_data/255.
sample_data = np.transpose(sample_data, [2, 0, 1])[np.newaxis, ...]
sample_data = torch.from_numpy(sample_data).to(device, dtype=torch.float)

for i in range(epoch_num):
    
    
    for idx, (data, label) in enumerate(DR_loader):
        current_step += 1
        
        
        data_shape = data.shape #[4, 30, 128, 128, 3]
        label_shape = label.shape # [ 4, 30, 128, 128, 1]
        
        data = np.transpose(data, [0, 1, 4, 2, 3])
        label = np.transpose(label, [0, 1, 4, 2, 3])
        data_list = []
        label_list = []
        
        for i_idx in range(data.shape[0]):
            for j_idx in range(data.shape[1]):
                data_list.append(data[i_idx, j_idx, ...].numpy())
                label_list.append(label[i_idx, j_idx, ...].numpy())
        #print(data_list)
        #data_test = data[1, 0, ...]
        #label_test = label[1, 0, ...]
        #cv2.imwrite('./result/test_data_{0}.png'.format(current_step), data_test.numpy()*255.)
        #cv2.imwrite('./result/test_label_{0}.png'.format(current_step), label_test.numpy()*255.)
        data_list = np.asarray(data_list)
        data = np.reshape(data_list, [-1, 3, 128, 128]) # [4*30, 3, 128, 128]
        label = np.reshape(np.asarray(label_list), [-1, 128, 128]) # [4*30, 128, 128]
        
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        
        data = data.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        
        output = model(data)
        #print(output.shape)
        output_v = output.detach().cpu().numpy()
        #print(np.argmax(output_v, axis=1))
        #print(np.max(np.argmax(output_v, axis=1)))
        

        optimizer.zero_grad()
        
        loss_v = loss_fn(output, label)
        
        
        loss_v.backward()
        optimizer.step()
        
        print('Epoch {0} Step {1} loss {2:.4f}'.format(i, current_step, loss_v.item()))
        #print('Output max {0} min {1} ; Label max {2} min{3} '.format(output_v.max(), output_v.min(), label.max(), label.min()))
        
    if (i+1)%5 == 0:
        print('Saving model and parameters')
        save_path = './checkpoints/{0}_{1}.pth'.format(model_name, i+1)
        if not os.path.exists('./checkpoints/'):
            os.makedirs('./checkpoints/')
        state_dict = model.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
        
    if (i+1)%1 == 0:
        sample_out = model(sample_data).detach().cpu()
        sample_softmax = F.softmax(sample_out, dim=1).numpy()[:, 1, :, :]
        #sample_out = np.argmax(sample_out, axis=1)
        sample_out = np.squeeze(sample_softmax, axis=0)
        #sample_out = np.transpose(sample_out, [1, 2, 0])
        
        cv2.imwrite('./tmp/sample_test_{0}.jpg'.format(i+1), sample_out*255.)


save_path = './checkpoints/{0}_last.pth'.format(model_name)
state_dict = model.state_dict()
torch.save(state_dict, save_path)



