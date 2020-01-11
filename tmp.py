# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 14:11:03 2020

@author: kasy
"""

import os
import json
import argparse

config_path = './config.json'

config = json.load(open(config_path))
epoch_num = config['trainer']['epoch']
print(epoch_num)

