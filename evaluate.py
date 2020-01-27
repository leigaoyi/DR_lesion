# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:21:20 2020

@author: kasy
"""

import numpy as np
import cv2

import glob
import os
import json

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F



import re
from tqdm import tqdm

config_path = './config.json'
config = json.load(open(config_path))

def single_auc(pred_path, label_path):
    pred_fig = cv2.imread(pred_path)[..., 0]/255.
    label_fig = cv2.imread(label_path)[..., 0]/255
    
    pred_fig = np.reshape(pred_fig, [-1])
    label_fig = np.reshape(label_fig, [-1])
    label_fig = np.asarray(label_fig, dtype=np.int)
    
    auc = roc_auc_score(label_fig, pred_fig)
    #print(auc)
    return auc


def dataset_auc(dataset_path):
    pred_list = glob.glob(os.path.join(dataset_path, 'pred*.jpg'))
    label_list = glob.glob(os.path.join(dataset_path, 'label*.jpg'))
    
    pred_list = sorted(pred_list, key=lambda str_i: int(str_i[-9:-7]))
    label_list = sorted(label_list, key= lambda str_i: int(str_i[-9:-7]))
    
    #print(pred_list)
    auc_list = []
    len_pred = len(pred_list)
    for i in tqdm(range(len_pred)):
        auc = single_auc(pred_list[i], label_list[i])
        auc_list.append(auc)
        
    return np.mean(auc_list)



def single_f1(pred_path, label_path):
    pred_fig = cv2.imread(pred_path)[..., 0]/255.
    pred_fig[pred_fig>0.5] = 1
    pred_fig[pred_fig<=0.5] = 0
    label_fig = cv2.imread(label_path)[..., 0]/255
    
    pred_fig = np.reshape(pred_fig, [-1])
    label_fig = np.reshape(label_fig, [-1])
    label_fig = np.asarray(label_fig, dtype=np.int)
    
    #auc = roc_auc_score(label_fig, pred_fig)
    f1 = f1_score(label_fig, pred_fig)
    
    #print(auc)
    return f1


def dataset_f1(dataset_path):
    pred_list = glob.glob(os.path.join(dataset_path, 'pred*.jpg'))
    label_list = glob.glob(os.path.join(dataset_path, 'label*.jpg'))
    
    pred_list = sorted(pred_list, key=lambda str_i: int(re.findall(r'\d+', str_i)[0]))
    label_list = sorted(label_list, key= lambda str_i: int(re.findall(r'\d+', str_i)[0]))
    print(pred_list)
    print(label_list)
    #print(pred_list)
    f1_list = []
    len_pred = len(pred_list)
    for i in tqdm(range(len_pred)):
        f1 = single_f1(pred_list[i], label_list[i])
        f1_list.append(f1)
        
    return np.mean(f1_list)


model_name = config['trainer']['model_name']
class_name = config['trainer']['train_class']

def cal_auc():
    dataset_path = './result/{0}/{1}/'.format(model_name, class_name)
    auc = dataset_auc(dataset_path)
    print(model_name, class_name, auc)
    return auc

def cal_f1():
    dataset_path = './result/{0}/{1}/'.format(model_name, class_name)
    f1 = dataset_f1(dataset_path)
    return f1

class_list = ['EX', 'HE', 'MA', 'SE']


model_name = config['predict']['pred_model']
step_num = config['predict']['reuse_step']
pred_type = config['trainer']['train_mode']

class_name = class_list[0]

#print('Predict {0} with epoch {1}'.format(model_name, step_num))
result_dir = './result/{0}/{1}/'.format(model_name, class_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    

def recover(fig_list, shape, crop_size=512):
    w_int, h_int = shape[0]//crop_size, shape[1]//crop_size
    
    fig_zeros = np.zeros([(w_int+1)*crop_size, (h_int+1)*crop_size])
    
    for i in range(w_int+1):
        for j in range(h_int+1):
            fig_zeros[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size] = fig_list[i*(h_int+1)+j]
    fig_output = fig_zeros[:shape[0], :shape[1]]
    return fig_output

def predict_model(model, device, DR_loader):
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
    
        if pred_type == "fast":
            if current_step == 4:
                break
        
    return 0