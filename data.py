# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:13:53 2020

@author: kasy
"""

import numpy as np
import os
import torch
import cv2
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms

import PIL
from PIL import ImageEnhance, Image

import random
import numpy as np


class DR_EX(Dataset):
    def __init__(self, data_dir, label_dir, crop_size=128):
        super(DR_EX, self).__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.crop_size = crop_size
        
        self.data_list = os.listdir(self.data_dir)
        self.label_list = os.listdir(self.label_dir)
        
        #print(self.label_list)
        self.data_list = sorted(self.data_list, key = lambda str_n: int(str_n[6:8]))
        self.label_list = sorted(self.label_list, key = lambda str_n: int(str_n[6:8]))
        
        print(self.data_list)
        print(self.label_list)

    def __len__(self):
        return len(self.data_list)
        #return 4
    
    def __getitem__(self, index):
        fig_name = os.path.join(self.data_dir, self.data_list[index])
        label_name = os.path.join(self.label_dir, self.label_list[index])
        assert self.data_list[index][6:8] == self.label_list[index][6:8]
        #fig = cv2.imread(fig_name)
        #label = cv2.imread(label_name)[..., 2]
        
        fig, label = self._enhance(fig_name, label_name)
        
        fig = fig/255.0
        
        crop_num = 20
        fig_crop, label_crop = self._random_crop(fig, label, crop_num)
        fig_crop = np.asarray(fig_crop)
        label_crop = np.asarray(label_crop)
        label_crop = np.reshape(label_crop, [crop_num, self.crop_size, self.crop_size, 1])
        
        return fig_crop, label_crop
        
    def _random_crop(self, fig, label, crop_num=10):
        w, h, c = fig.shape
        crop_fig_list = []
        crop_label_list = []
        
        for i in range(crop_num):
            x_start = np.random.randint(0, w-self.crop_size)
            y_start = np.random.randint(0, h-self.crop_size)
            
            fig_crop = fig[x_start:(x_start+self.crop_size), y_start:(y_start+self.crop_size)]
            label_crop = label[x_start:(x_start+self.crop_size), y_start:(y_start+self.crop_size)]
            
            crop_fig_list.append(fig_crop)
            crop_label_list.append(label_crop)
            
        return crop_fig_list, crop_label_list
    
    
    def _enhance(self, fig_path, annot_path):
        image = Image.open(fig_path)
        annotation = Image.open(annot_path)
        
        # light
        enh_bri = ImageEnhance.Brightness(image)
        brightness = round(random.uniform(0.8, 1.2), 2)
        image = enh_bri.enhance(brightness)
        
        # color
        enh_col = ImageEnhance.Color(image)
        color = round(random.uniform(0.8, 1.2), 2)
        image = enh_col.enhance(color)
        
        
        # contrast
        enh_con = ImageEnhance.Contrast(image)
        contrast = round(random.uniform(0.8, 1.2), 2)
        image = enh_con.enhance(contrast)
        #
        # enh_sha = ImageEnhance.Sharpness(image)
        # sharpness = round(random.uniform(0.8, 1.2), 2)
        # image = enh_sha.enhance(sharpness)
        
        method = random.randint(0, 7)
        # print(method)
        if method < 7:
            image = image.transpose(method)
            annotation = annotation.transpose(method)
        degree = random.randint(-5, 5)
        image = image.rotate(degree)
        annotation = annotation.rotate(degree)
        
        #image.save('test3.jpg')
        #annotation.save('test_label.tif')
        
        img_arr = np.array(image)
        annot_arr = np.array(annotation)
        
        return img_arr, annot_arr
    
    
class DR_EX_test(Dataset):
    def __init__(self, data_dir, label_dir, crop_size=512):
        super(DR_EX_test, self).__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.crop_size = crop_size
        
        self.data_list = os.listdir(self.data_dir)
        self.label_list = os.listdir(self.label_dir)
        
        #print(self.label_list)
        #print(self.label_list)
        self.data_list = sorted(self.data_list, key = lambda str_n: int(str_n[6:8]))
        self.label_list = sorted(self.label_list, key = lambda str_n: int(str_n[6:8]))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        fig_name = os.path.join(self.data_dir, self.data_list[index])
        label_name = os.path.join(self.label_dir, self.label_list[index])
        
        assert self.data_list[index][6:8] == self.label_list[index][6:8]
        
        fig = cv2.imread(fig_name)
        b,g,r = cv2.split(fig)
        fig_RGB = cv2.merge([r, g, b])
        label = cv2.imread(label_name)[..., 2]
        
        label = 1*(label>0.1)
        fig = fig_RGB/255.0
        
        fig_crop, label_crop = self._paint_crop(fig, label)
        fig_shape = fig.shape[:2]
        
        return fig_crop, label_crop, fig_shape, self.label_list[index]
    
    def _paint_crop(self, img, label):
        w, h, c = img.shape
        w_int, h_int = w//self.crop_size, h//self.crop_size
        
        img_paint = np.zeros([(w_int+1)*self.crop_size, (h_int+1)*self.crop_size, c])
        label_paint = np.zeros([(w_int+1)*self.crop_size, (h_int+1)*self.crop_size, 1])
        
        img_paint[:w, :h, :] = img
        label_paint[:w, :h, :] = label[..., np.newaxis]
        
        img_list = []
        label_list = []
        
        for i in range(w_int+1):
            for j in range(h_int+1):
                img_list.append(img_paint[i*self.crop_size:(i+1)*self.crop_size, j*self.crop_size:(j+1)*self.crop_size, :])
                label_list.append(label_paint[i*self.crop_size:(i+1)*self.crop_size, j*self.crop_size:(j+1)*self.crop_size, :])
                
        img_list = np.asarray(img_list)
        label_list = np.asarray(label_list)
        return img_list, label_list
         