# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 18:58:59 2020

@author: kasy
"""

import cv2
import os

label_path = './data/ground_truths/EX/IDRiD_10_EX.tif'
data_path = './data/images/IDRiD_10.jpg'


label = cv2.imread(label_path)[ 537:(537+128), 1890:(1890+196),2]
data = cv2.imread(data_path)[537:(537+128), 1890:(1890+196), :]

if not os.path.exists('./tmp/'):
    os.makedirs('./tmp/')

cv2.imwrite('./tmp/sample_label.jpg', label)
cv2.imwrite('./tmp/sample_data.jpg', data)
