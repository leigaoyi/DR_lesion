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

model_name = config['evaluate']['eval_model']
class_name = config['evaluate']['eval_class']

dataset_path = './result/{0}/{1}/'.format(model_name, class_name)
auc = dataset_auc(dataset_path)
print(model_name, class_name, auc)