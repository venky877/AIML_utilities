# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:23:01 2020

@author: 205557
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append("D:/PROJECTS_ROOT/AIML_utilities/")
sys.path.append('D:/PROJECTS_ROOT/AIML_utilities\spivision/For_Object_Detection/')
import spivision
from spivision.For_Object_Detection import analysis_objectdetection
from spivision.For_Object_Detection import pascol_voc_converter_utils
from spivision.For_Object_Detection import plot_annotation_utils
from augmentation_code import *
from utilities.utils import *
from sklearn.utils import shuffle
help(pascol_voc_converter_utils)
help(plot_annotation_utils)

'''
converting the xml files into csv for object detection 
'''

xml_folder_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train/"
out_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"
pascol_voc_converter_utils.convert_pascol_voc_xml_to_csv(xml_folder_path, out_path=out_path, return_df=False)

xml_folder_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/test/"
out_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/test.csv"
pascol_voc_converter_utils.convert_pascol_voc_xml_to_csv(xml_folder_path, out_path=out_path, return_df=False)


''' Testing convert_csv_2_pascol_voc_xml. Create a folder dummy in the location mentioned below'''

xml_folder_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/dummy/"
create_folder(xml_folder_path)
csv_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"
image_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train/"

pascol_voc_converter_utils.convert_csv_2_pascol_voc_xml(csv_path, image_path, xml_folder_path,labelname='class')

''' Testing the plotting and annotation tools '''
csv_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"
annotated_files_out_folder_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/annotatedimages/"
original_images_input_folder_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train/"
create_folder(annotated_files_out_folder_path)
plot_annotation_utils.plot_annotation(csv_path, annotated_files_out_folder_path, original_images_input_folder_path, labelname='class')

'''testing the label wise annotations'''
plot_annotation_utils.plot_annotation_labelwise(csv_path, annotated_files_out_folder_path, original_images_input_folder_path,labelname='class', first_5_only=False)

''' testing annotations of both gt and predicted on the same image all labels '''
pred_csv_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"
gt_csv_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"

annotated_files_out_folder_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/annotatedimages/"
original_images_input_folder_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train/"
plot_annotation_utils.plot_annotation_gt_pred(pred_csv_path,gt_csv_path, annotated_files_out_folder_path, original_images_input_folder_path,colorgt=(255,0,0),colorpred=(0,0,255), labelname='class')


''' testing annotations of both gt and predicted on the same image label wise '''
pred_csv_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"
gt_csv_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv"

annotated_files_out_folder_path="D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/annotatedimages/"
original_images_input_folder_path= "D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train/"
plot_annotation_utils.plot_annotation_labelwise_gt_pred(pred_csv_path,gt_csv_path, annotated_files_out_folder_path, original_images_input_folder_path,colorgt=(255,0,0),colorpred=(0,0,255),labelname='class', first_5_only=False)













imagelib= 'D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train/'

bbox_df = pd.read_csv('D:/PROJECTS_ROOT/AIML_utilities/data_for_testing/objectdetection/train.csv')
#bbox_df['img_name'] = [i.split('/')[-1] for i in bbox_df['path']]
bbox_df['label_width']=bbox_df['xmax']-bbox_df['xmin']
bbox_df['label_height']=bbox_df['ymax']-bbox_df['ymin']
img_list= bbox_df['path'].unique()
all_df = pd.DataFrame()
for augment_name in augment_list:
    print(str(augment_name)+' in progress!!!')
    for i in tqdm(img_list):  
img_name,img1,bboxes1,label,width,height,orig_label_height,orig_label_width,no_of_labels = read_data(input_img_dir+i,bbox_df)


path= imagelib+img_list[0]
img = cv2.imread(path)
img_name = path.split('/')[-1]
group = bbox_df.groupby('path')
img_csv = group.get_group(img_name)  
img_csv.sort_values('label',ascending=False,inplace=True)
img_csv['height']= img.shape[0]
img_csv['width']= img.shape[1]
img_csv= img_csv[['path','label','width','height','xmin','ymin','xmax','ymax','label_width','label_height']]
bboxes = img_csv.to_numpy()
orig_label_height = bboxes[:,-1]
orig_label_width = bboxes[:,-2]
label = bboxes[:,1]
width = bboxes[:,2]
height = bboxes[:,3]
bboxes = bboxes[:,4:8]
#print(label)
#input("enter")
no_of_labels = np.size(bboxes,0)

img_name = img_name[:img_name.rfind(".")]

scale_x= 0.9
scale_y= 0.9

img_shape = img.shape
resize_scale_x = 1 + scale_x
resize_scale_y = 1 + scale_y
img=  cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)
img.shape
bboxes[:,:4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
canvas = np.zeros(img_shape, dtype = np.uint8)
canvas.shape
canvas.fill(255)
#y_lim = int(min(resize_scale_y,1)*img_shape[0])
#x_lim = int(min(resize_scale_x,1)*img_shape[1])
#canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]
img = canvas
bboxes = clip_box(bboxes, [0,0,1 + img_shape[1], img_shape[0]], 0.25)
cv2.imwrite(imagelib+ 'experiment1.jpg',img)

img=  cv2.resize(img, None, fx = 1/resize_scale_x, fy = 1/resize_scale_y)

cv2.imwrite(imagelib+ 'experiment1_1.jpg',img)

img.shape

from augmentation_code import *

input_img_dir = 'D:/PROJECTS_ROOT/DataServices/concise/jpg_xml_combined/'
csv_path = 'D:/PROJECTS_ROOT/DataServices/concise/base_data_text_scientific_forms.csv'
img_format = 'jpg'
out_dir ='D:/PROJECTS_ROOT/DataServices/concise/augmentexp/'
augment_list = ['scale','translate','trans_scale']
aug_min = 0.1
aug_interval = 0.05
aug_max = 0.20
convert_image_depth = 'no'

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





csv_path='D:/PROJECTS_ROOT/DataServices/concise/base_data_text_scientific_forms.csv'
annotated_files_out_folder_path='D:/PROJECTS_ROOT/DataServices/concise/annotatedimages/'
original_images_input_folder_path= 'D:/PROJECTS_ROOT/DataServices/concise/jpg_xml_combined/'
create_folder(annotated_files_out_folder_path)
plot_annotation_utils.plot_annotation(csv_path, annotated_files_out_folder_path, original_images_input_folder_path, labelname='label')

'''testing the label wise annotations'''
plot_annotation_utils.plot_annotation_labelwise(csv_path, annotated_files_out_folder_path, original_images_input_folder_path,labelname='label', first_5_only=False)


# testing the augmentation code

input_img_dir = 'D:/PROJECTS_ROOT/DataServices/concise/jpg_xml_combined/'
csv_path = 'D:/PROJECTS_ROOT/DataServices/concise/venkysample.csv'
img_format = 'jpg'
out_dir ='D:/PROJECTS_ROOT/DataServices/concise/augment_exp/'
augment_list = ['scale','trans','trans_scale']
aug_min = 0.1
aug_interval = 0.05
aug_max = 0.15
convert_image_depth = 'no'
splits=[0.85, 0.90]

create_augmentation(input_img_dir,csv_path,out_dir,augment_list, aug_min, aug_interval, aug_max, convert_image_depth, img_format, splits)

data=pd.read_csv(csv_path)
len(data['path'].unique())

x=[1,2,3,4]
x[3:]


"{0:.4f}".format(scale_list[])

str(scale_list[k])

