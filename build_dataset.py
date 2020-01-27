# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:19:25 2020

@author: kasy
"""

import os
import cv2

import numpy as np
from tqdm import tqdm

import json

config_path = './config.json'
config = json.load(open(config_path))

crop_size = config['build']['crop_size']

data_dir = './data/images/'
label_dir = './data/ground_truths/EX/'

dest_data_dir = './data/train/images/'
dest_label_dir = './data/train/labels/EX/'

if not os.path.exists(dest_data_dir):
    os.makedirs(dest_data_dir)
    os.makedirs(dest_label_dir)

data_list = os.listdir(data_dir)
label_list = os.listdir(label_dir)

#print(label_list)
data_list = sorted(data_list, key = lambda str_n: int(str_n[6:8]))
label_list = sorted(label_list, key = lambda str_n: int(str_n[6:8]))

data_num = len(data_list)

def crop_img_label(img, label, crop_size = crop_size):
    h, w, c = img.shape
    h_num, w_num = h//crop_size, w//crop_size
    
    img_zeros = np.zeros([(h_num+1)*crop_size, (w_num+1)*crop_size, c])
    label_zeros = np.zeros([(h_num+1)*crop_size, (w_num+1)*crop_size])
    
    img_zeros[:h, :w, :] = img
    label_zeros[:h, :w] = label
    
    img_list = []
    label_list = []
    
    for i in range(h_num+1):
        for j in range(w_num+1):
            img_patch = img_zeros[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size, :]
            label_patch = label_zeros[i*crop_size:(i+1)*crop_size, j*crop_size:(j+1)*crop_size]
            
            img_list.append(img_patch)
            label_list.append(label_patch)
            
    return img_list, label_list

start_num = 0

for i in tqdm(range(data_num)):
    fig = cv2.imread(os.path.join(data_dir, data_list[i]))
    label = cv2.imread(os.path.join(label_dir, label_list[i]))[..., 2]
    
    assert fig.shape[:2] == label.shape[:2]
    
    img_crop, label_crop = crop_img_label(fig, label)
    
    img_num = len(img_crop)
    
    for j in range(img_num):
        
        fig_p_name = str(start_num) + '.png'
        label_p_name = str(start_num) + '.png'
        
        cv2.imwrite(os.path.join(dest_data_dir, fig_p_name), img_crop[j])
        cv2.imwrite(os.path.join(dest_label_dir, label_p_name), label_crop[j])
        
        start_num += 1
        
    