# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:22:55 2020

@author: kasy
"""

import PIL
from PIL import ImageEnhance, Image

import random
import numpy as np
import cv2

fig_path = './data/images/IDRiD_03.jpg'
annot_path = './data/ground_truths/EX/IDRiD_03_EX.tif '

def _enhance(fig_path, annot_path):
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
    
#    image.save('test3.jpg')
#    annotation.save('test_label.tif')
    
    img_arr = np.array(image)
    annot_arr = np.array(annotation)
    
    return img_arr, annot_arr

img, annot = _enhance(fig_path, annot_path)
cv2.imwrite('test_label.jpg', annot*255.)