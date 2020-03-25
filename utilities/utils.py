# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:27:12 2020

@author: 205557
"""
import pandas as pd
import numpy as np
import sys
sys.path.append("D:/PROJECTS_ROOT/AIML_utilities/")
sys.path.append('D:/PROJECTS_ROOT/AIML_utilities\spivision/For_Object_Detection/')
import spivision
from spivision.For_Object_Detection import analysis_objectdetection
from spivision.For_Object_Detection import pascol_voc_converter_utils
from spivision.For_Object_Detection import plot_annotation_utils
from augmentation_code import *
from utilities.utils import create_folder
from sklearn.utils import shuffle
import os
import cv2
import shutil

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
def convert_to_gray(input_folder, output_folder, file_type):
    piclist= [x for x in os.listdir(input_folder) if x[-3:]== file_type]
    for pic in piclist:
        image = cv2.imread(input_folder+pic)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_folder + pic, gray)
        
def create_augmentation(input_img_dir,csv_path,out_dir,augment_list, aug_min, aug_interval, aug_max, convert_image_depth):
    data = pd.read_csv(csv_path)
    files= data[['path']].drop_duplicates()
    files = shuffle(files)
    train= files[:int(0.70*len(files))]
    valid= files[int(0.70*len(files)):int(0.85*len(files))]
    test= files[int(0.85*len(files)):]
    traincsv= pd.merge(data,train, on=['path'], how='inner')
    validcsv= pd.merge(data,valid, on=['path'], how='inner')
    testcsv= pd.merge(data,test, on=['path'], how='inner')
    traincsv.to_csv(out_dir+'train.csv', index= False)
    validcsv.to_csv(out_dir+'valid.csv', index= False)
    testcsv.to_csv(out_dir+'test.csv', index= False)
    traincsv_path= out_dir+'train.csv'
    validcsv_path= out_dir+'valid.csv'
    testcsv_path= out_dir+'test.csv'
    augment_image_bbox(input_img_dir,traincsv_path,img_format,out_dir+'train_aug/','train_aug.csv',augment_list,aug_min,aug_interval,aug_max)
    augment_image_bbox(input_img_dir,validcsv_path,img_format,out_dir+'valid_aug/','valid_aug.csv',augment_list,aug_min,aug_interval,aug_max)
    augment_image_bbox(input_img_dir,testcsv_path,img_format,out_dir+'test_aug/','test_aug.csv',augment_list,aug_min,aug_interval,aug_max)
    shutil.copy(out_dir+'train_aug/'+'train_aug.csv', out_dir+'train_aug.csv')
    shutil.copy(out_dir+'valid_aug/'+'valid_aug.csv',out_dir+'valid_aug.csv')
    shutil.copy(out_dir+'test_aug/'+'test_aug.csv',out_dir+'test_aug.csv')
    datalist=[train,valid,test]
    dirlist=['train_aug/','valid_aug/','test_aug/']
    for set1,dir1 in zip(datalist,dirlist):
        for j in set1['path']:
            shutil.copy(input_img_dir+j, out_dir+dir1)
        df1= pd.read_csv(out_dir+dir1.split("_")[0]+'.csv')
        df2= pd.read_csv(out_dir+dir1[:-1]+'.csv')
        df= pd.concat([df2,df1])
        df= df.reset_index()
        df.to_csv(out_dir+dir1[:-1]+'.csv', index= False)

