# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:27:12 2020

@author: 205557
"""
import os
import cv2

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
def convert_to_gray(input_folder, output_folder, file_type):
    piclist= [x for x in os.listdir(input_folder) if x[-3:]== file_type]
    for pic in piclist:
        image = cv2.imread(input_folder+pic)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_folder + pic, gray)