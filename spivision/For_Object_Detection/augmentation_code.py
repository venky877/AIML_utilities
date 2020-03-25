# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:37:30 2019

@author: 206011
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:15:47 2019

@author: 206011
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 19:25:46 2019

@author: developer
"""

'''This module can augment the image along with the generation of modified bounding box values.
Different augmentation attempted here are:
    1. Scaling
    2. Translation
    3. Translation followed by Scaling
    4. Rotation
    5. Translation followed by Rotation
    Refer 'data_aug.py' to understand about the augmentations in depth.
After augmentation, images are created in local file in specified image format and 
bounding box values along with labels and image name is written in csv file. 
Finaly single augmented csv file is created.

Additionally, conversion of bit depth of image is provided.
Generally, output images are saved in 24 bit depth format which occupies more space. 
In order to reduce the file size, bit depth of output image is converted according to the input image bit depth. 
Input image bit depth can be obtained using PIL.Image.mode function. 
The output image is converted to the mode of input image. 

'''

'''Requirements: DataAugmentationForObjectDetection package
   It can be downloaded from github repo.'''
   
'''Usage example:
    input_img_dir = '/home/developer/deep_learning/data/ecrash_OD/test_aug/'

    img_format = 'png'
    no_of_labels = 70
    out_dir = '/home/developer/deep_learning/data/ecrash_OD/test_aug/augment_img'


    augment_list = ['scale','translate']
    aug_min = 0.1
    aug_interval = 0.01
    aug_max = 0.9
    convert_image_depth = 'yes'

    augment_image_bbox(input_img_dir,img_format,no_of_labels,out_dir,augment_name,aug_min,aug_interval,aug_max,convert_image_depth)
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import pandas as pd
from tqdm import tqdm
from PIL import Image
import sys
sys.path.append('D:/PROJECTS_ROOT/AIML_utilities\spivision/For_Object_Detection/')
from data_aug import *
from bbox_util import *


'''This function can create output directory if they does not exist.

    Arguments
    .........
    out_dir:  string
                  Directory path where the augmented images and csv are saved.
  
    Returns
    .......
    Directory is created
    Nothing is returned to calling function.'''
   
def create_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Output directory is created!!!")
   
        
        
'''This function can read image and csv from the specified path and return to calling function.

    Arguments
    .........
    path: string
          Absolute path of the input image    
    bbox_df: Dataframe
             Dataframe containing image name, bounding box coordinates xmin ymin xmax ymax and label for each box.
    Returns:
    .......    
   img_name: string
             Basename of the input image
   img:  numpy.ndarray
         Numpy image
   bboxes: numpy.ndarray
           Numpy array containing bounding boxes of shape `N X 4` where N is the 
           number of bounding boxes and the bounding boxes are represented in the
           format `xmin ymin xmax ymax`
   label: numpy.ndarray
          Numpy array containing label of shape ' N X 1'.'''
   
def read_data(path, bbox_df):
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
    return(img_name,img,bboxes,label,width,height,orig_label_height,orig_label_width,no_of_labels)

    
'''This function can apply the transformation on image and return transformed values to calling function.

    Arguments
    .........
    transforms : data_aug.Sequence class
                Transformation to be performed on the input image
    img: numpy.ndarray
         Numpy image 
    bboxes: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `xmin ymin xmax ymax`
    
    Retruns
    .......
    img_new: numpy.ndarray
             Augmented image in the numpy format 
    bboxes_new: numpy.ndarray
                Tranformed bounding box co-ordinates of the format `N x 4` where N is 
                number of bounding boxes and 4 represents `xmin ymin xmax ymax` of the box.'''
   
def transform_data(transforms,img,bboxes):
    img_new,bboxes_new = transforms(img,bboxes)
    bboxes_new = bboxes_new.astype(int)
    return(img_new,bboxes_new)
    
    
'''This function can write image and csv in specified output directory if the number of labels in modified image is equal to the actual number of labels.

    Arguments
    .........
    img_new: numpy.ndarray
             Augmented image in the numpy format 
    bboxes_new: numpy.ndarray
                Tranformed bounding box co-ordinates of the format `N x 4` where N is 
                number of bounding boxes and 4 represents `xmin ymin xmax ymax` of the box
    label: numpy.ndarray
          Numpy array containing label of shape ' N X 1'
    no_of_labels: integer
                  Actual number of labels in ground truth
    mod_img_name: string
                  Name of the new augmented/modified image 
    out_dir:  string
                  Directory path where the augmented images and csv are saved

    Returns
    .......
    Augmented images are saved in output directory
    Nothing is returned to calling function.'''
   
def write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir):
    ann_df = pd.DataFrame(bboxes_new, columns = ['xmin','ymin','xmax','ymax'])  
    count = ann_df.count()
    if((count[1] == no_of_labels)):
        ann_df['path']=mod_img_name
        ann_df['label']=label
        ann_df['width']=width
        ann_df['height']=height
        ann_df.sort_values('label',ascending=False,inplace=True)
        ann_df['label_width']=ann_df['xmax']-ann_df['xmin']
        ann_df['label_height']=ann_df['ymax']-ann_df['ymin']
        ann_df['original_label_width']=orig_label_width
        ann_df['original_label_height']=orig_label_height
        ann_df['ratio_width']=ann_df['label_width']/ann_df['original_label_width']
        ann_df['ratio_height']=ann_df['label_height']/ann_df['original_label_height']
        width_mean = ann_df['ratio_width'].mean()
        height_mean = ann_df['ratio_height'].mean()
        img_wo = img_new
        # img_new = draw_rect(img_new, bboxes_new,color=[0,255,1])
        # cv2.putText(img_new, label, (int(bboxes_new[2] - 10), int(bboxes_new[3] + 10)), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
        mod_img_ann = 'ann'+mod_img_name
        if((width_mean>0.8)&(height_mean>0.8)):
            all_df=all_df.append(ann_df,ignore_index=True)
            cv2.imwrite(os.path.join(out_dir,mod_img_name),img_wo)
            # cv2.imwrite(os.path.join(out_dir,mod_img_ann),img_new)
    return(all_df)
    
'''
This function converts the bit depth of augmented images according to the bit depth of input image. 
Augmented images with altered bit depth are overwritten on the current augmented images.

    Arguments
    .........
    out_dir: String
             Directory path of augmented images whose bit depth has to be changed.
    img_format: String
                Format of the image
    bit_depth: String
               Bit depth/Mode of the image
               
    Returns
    .......  
    Augmented images with altered bit depth are overwritten on the old augmented images in the same         
    directory
               
'''

def convert_bit_depth(out_dir, img_format, bit_depth):
    print("Convertion in progress!!\n")
    img_ext = '*.'+img_format
    out_img_list=glob.glob(out_dir+img_ext) 
    for i in out_img_list:
        new_i = os.path.join(out_dir,os.path.basename(i))
        image =Image.open(i)
        MOD_IM=image.convert(bit_depth)
        MOD_IM.save(new_i,img_format)

'''This is core function that can perform augmentations on image and bounding boxes.

    Arguments
    .........
    input_img_dir: string
                   Input image directory which contain images and csv files.
                   (eg: '/home/developer/deep_learning/data/ecrash_OD/test_aug/')
    img_format: string
                Format of the input Image. (eg: png) 
    no_of_labels: integer
                  Actual number of labels in original image (eg: 70)                  

    out_dir:  string
                  Directory path where the augmented images and csv are saved. 
                  (eg:  '/home/developer/deep_learning/data/ecrash_OD/test_aug/')
    augment_list: list of augment name
                  List containing the name of the augmentation to be performed. 
                  (eg: 'scale' for scaling,

                       'translate' for translation,
                       'trans_scale' for translation followed by scaling,
                       'rotate' for rotation,
                       'trans_rotate' for translation followed by rotation
                        Default value is ['scale','translate'])

    For augmentation, three parameters are required. They are scale_min, scale_interval, scale_max.
    aug_min: float
             Minimum value to begin the augmentation. (Default: 0.01)
    aug_max: float 
             Maximum value to end the augmentation. (Default: 0.1)
    aug_interval: float
                  Step value to increment the current scale value. (Default: 0.01)  
    convert_image_depth: string
                         This take yes/no as input to choose conversion of image bit depth.
                         If you want to convert, provide 'yes', otherwise provide 'no'.
                         Default value is 'yes'

    Returns
    .......
    Augmented images and corresponding csv file containing image name, transformed bounding box values and   
    labels are created in the specified output directory

    Nothing is retured to calling function.'''
    
def augment_image_bbox(input_img_dir,csv_path,img_format,out_dir,final_csvname,augment_list=['scale'],aug_min=0.03,aug_interval=0.01,aug_max=0.1,convert_image_depth='no'):
    create_dir(out_dir) 
    #img_ext = '*.'+img_format
    #img_list = glob.glob(os.path.join(input_img_dir,img_ext))
    scale_list = np.arange(aug_min,aug_max,aug_interval)
    bbox_df = pd.read_csv(csv_path)
    #bbox_df['img_name'] = [i.split('/')[-1] for i in bbox_df['path']]
    bbox_df['label_width']=bbox_df['xmax']-bbox_df['xmin']
    bbox_df['label_height']=bbox_df['ymax']-bbox_df['ymin']
    img_list= bbox_df['path'].unique()
    PIL_image=Image.open(input_img_dir+img_list[0])
    image_mode =PIL_image.mode
    all_df = pd.DataFrame()
    print((augment_list))
    for augment_name in augment_list:
        print(str(augment_name)+' in progress!!!')
        for i in tqdm(img_list):  
            img_name,img1,bboxes1,label,width,height,orig_label_height,orig_label_width,no_of_labels = read_data(input_img_dir+i,bbox_df)
            img_name = img_name[:img_name.rfind(".")]
            if(augment_name=='scale'):
                for j in range(len(scale_list)):
                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomScale(scale_list[j],diff=False)])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_scale_false_'+"{0:.4f}".format(scale_list[j])+'.'+img_format
                    all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomScale(scale_list[j],diff=True)])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_scale_true_'+"{0:.4f}".format(scale_list[j])+'.'+img_format
                    all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

            elif(augment_name=='translate'): 
                 for j in range(len(scale_list)):
                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomTranslate(scale_list[j],diff=False)])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_trans_false_'+"{0:.4f}".format(scale_list[j])+'.'+img_format
                    all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomTranslate(scale_list[j],diff=True)])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_trans_true_'+"{0:.4f}".format(scale_list[j])+'.'+img_format
                    all_df= write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

            elif(augment_name=='rotate'):
                 for j in range(len(scale_list)):
                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomRotate(scale_list[j])])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_rotate_'+"{0:.4f}".format(scale_list[j])+'.'+img_format
                    all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

            elif(augment_name=='trans_scale'):
                for j in range(len(scale_list)):
                    for k in range(len(scale_list)):
                        img2=img1.copy()
                        bboxes2=bboxes1.copy()
                        transforms = Sequence([RandomTranslate(scale_list[k],diff=True),RandomScale(scale_list[j],diff=True)])
                        img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                        mod_img_name = img_name +'_trans_true_'+"{0:.4f}".format(scale_list[k])+'_scale_true_'+"{0:.4f}".format(scale_list[j])+'.'+img_format
                        all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

                        img2=img1.copy()
                        bboxes2=bboxes1.copy()
                        transforms = Sequence([RandomTranslate(scale_list[k],diff=True),RandomScale(scale_list[j],diff=False)])
                        img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                        mod_img_name = img_name +'_trans_true_'+"{0:.4f}".format(scale_list[k])+'_scale_false_'+"{0:.4f}".format(scale_list[j])+'.'+img_format
                        all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

                        img2=img1.copy()
                        bboxes2=bboxes1.copy()
                        transforms = Sequence([RandomTranslate(scale_list[k],diff=False),RandomScale(scale_list[j],diff=False)])
                        img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                        mod_img_name = img_name +'_trans_false_'+"{0:.4f}".format(scale_list[k])+'_scale_false_'+"{0:.4f}".format(scale_list[j])+'.'+img_format
                        all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

                        img2=img1.copy()
                        bboxes2=bboxes1.copy()
                        transforms = Sequence([RandomTranslate(scale_list[k],diff=False),RandomScale(scale_list[j],diff=True)])
                        img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                        mod_img_name = img_name +'_trans_false_'+"{0:.4f}".format(scale_list[k])+'_scale_true_'+"{0:.4f}".format(scale_list[j])+'.'+img_format
                        all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

            elif(augment_name=='trans_rotate'):
                for j in range(len(scale_list)):
                    for k in range(len(scale_list)):
                         img2=img1.copy()
                         bboxes2=bboxes1.copy()
                         img_name,img,bboxes,label = read_data(i,input_img_dir)
                         transforms = Sequence([RandomTranslate(scale_list[k],diff=False),RandomRotate(scale_list[j])])
                         img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                         mod_img_name = img_name +'_tr_false_'+"{0:.4f}".format(scale_list[k])+'_rot_'+"{0:.4f}".format(scale_list[j])+'.'+img_format
                         all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

                         img2=img1.copy()
                         bboxes2=bboxes1.copy()
                         transforms = Sequence([RandomTranslate(scale_list[k],diff=True),RandomRotate(scale_list[j])])
                         img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                         mod_img_name = img_name +'_tr_true_'+"{0:.4f}".format(scale_list[k])+'_rot_'+"{0:.4f}".format(scale_list[j])+'.'+img_format
                         all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)
                
            all_df=all_df[['path','xmin','ymin','xmax','ymax','label','width','height','label_width',
                   'label_height','original_label_width','original_label_height',
                   'ratio_width','ratio_height']]
            #final_csv_name='augment_all.csv'
            all_df.to_csv(os.path.join(out_dir,final_csvname),index=False)
            if(convert_image_depth=='yes'):
                convert_bit_depth(out_dir,img_format,image_mode)
    print("Augmentation is done successfully!!!")    
    

            
    
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
               


# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:37:30 2019

@author: 206011
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:15:47 2019

@author: 206011
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 19:25:46 2019

@author: developer
"""

'''This module can augment the image along with the generation of modified bounding box values.
Different augmentation attempted here are:
    1. Scaling
    2. Translation
    3. Translation followed by Scaling
    4. Rotation
    5. Translation followed by Rotation
    Refer 'data_aug.py' to understand about the augmentations in depth.
After augmentation, images are created in local file in specified image format and 
bounding box values along with labels and image name is written in csv file. 
Finaly single augmented csv file is created.

Additionally, conversion of bit depth of image is provided.
Generally, output images are saved in 24 bit depth format which occupies more space. 
In order to reduce the file size, bit depth of output image is converted according to the input image bit depth. 
Input image bit depth can be obtained using PIL.Image.mode function. 
The output image is converted to the mode of input image. 

'''

'''Requirements: DataAugmentationForObjectDetection package
   It can be downloaded from github repo.'''
   
'''Usage example:
    input_img_dir = '/home/developer/deep_learning/data/ecrash_OD/test_aug/'

    img_format = 'png'
    no_of_labels = 70
    out_dir = '/home/developer/deep_learning/data/ecrash_OD/test_aug/augment_img'


    augment_list = ['scale','translate']
    aug_min = 0.1
    aug_interval = 0.01
    aug_max = 0.9
    convert_image_depth = 'yes'

    augment_image_bbox(input_img_dir,img_format,no_of_labels,out_dir,augment_name,aug_min,aug_interval,aug_max,convert_image_depth)
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import pandas as pd
from tqdm import tqdm
from PIL import Image
import sys
sys.path.append('D:/PROJECTS_ROOT/AIML_utilities\spivision/For_Object_Detection/')
from data_aug import *
from bbox_util import *


'''This function can create output directory if they does not exist.

    Arguments
    .........
    out_dir:  string
                  Directory path where the augmented images and csv are saved.
  
    Returns
    .......
    Directory is created
    Nothing is returned to calling function.'''
   
def create_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Output directory is created!!!")
   
        
        
'''This function can read image and csv from the specified path and return to calling function.

    Arguments
    .........
    path: string
          Absolute path of the input image    
    bbox_df: Dataframe
             Dataframe containing image name, bounding box coordinates xmin ymin xmax ymax and label for each box.
    Returns:
    .......    
   img_name: string
             Basename of the input image
   img:  numpy.ndarray
         Numpy image
   bboxes: numpy.ndarray
           Numpy array containing bounding boxes of shape `N X 4` where N is the 
           number of bounding boxes and the bounding boxes are represented in the
           format `xmin ymin xmax ymax`
   label: numpy.ndarray
          Numpy array containing label of shape ' N X 1'.'''
   
def read_data(path, bbox_df):
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
    return(img_name,img,bboxes,label,width,height,orig_label_height,orig_label_width,no_of_labels)

    
'''This function can apply the transformation on image and return transformed values to calling function.

    Arguments
    .........
    transforms : data_aug.Sequence class
                Transformation to be performed on the input image
    img: numpy.ndarray
         Numpy image 
    bboxes: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `xmin ymin xmax ymax`
    
    Retruns
    .......
    img_new: numpy.ndarray
             Augmented image in the numpy format 
    bboxes_new: numpy.ndarray
                Tranformed bounding box co-ordinates of the format `N x 4` where N is 
                number of bounding boxes and 4 represents `xmin ymin xmax ymax` of the box.'''
   
def transform_data(transforms,img,bboxes):
    img_new,bboxes_new = transforms(img,bboxes)
    bboxes_new = bboxes_new.astype(int)
    return(img_new,bboxes_new)
    
    
'''This function can write image and csv in specified output directory if the number of labels in modified image is equal to the actual number of labels.

    Arguments
    .........
    img_new: numpy.ndarray
             Augmented image in the numpy format 
    bboxes_new: numpy.ndarray
                Tranformed bounding box co-ordinates of the format `N x 4` where N is 
                number of bounding boxes and 4 represents `xmin ymin xmax ymax` of the box
    label: numpy.ndarray
          Numpy array containing label of shape ' N X 1'
    no_of_labels: integer
                  Actual number of labels in ground truth
    mod_img_name: string
                  Name of the new augmented/modified image 
    out_dir:  string
                  Directory path where the augmented images and csv are saved

    Returns
    .......
    Augmented images are saved in output directory
    Nothing is returned to calling function.'''
   
def write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir):
    ann_df = pd.DataFrame(bboxes_new, columns = ['xmin','ymin','xmax','ymax'])  
    count = ann_df.count()
    if((count[1] == no_of_labels)):
        ann_df['path']=mod_img_name
        ann_df['label']=label
        ann_df['width']=width
        ann_df['height']=height
        ann_df.sort_values('label',ascending=False,inplace=True)
        ann_df['label_width']=ann_df['xmax']-ann_df['xmin']
        ann_df['label_height']=ann_df['ymax']-ann_df['ymin']
        ann_df['original_label_width']=orig_label_width
        ann_df['original_label_height']=orig_label_height
        ann_df['ratio_width']=ann_df['label_width']/ann_df['original_label_width']
        ann_df['ratio_height']=ann_df['label_height']/ann_df['original_label_height']
        width_mean = ann_df['ratio_width'].mean()
        height_mean = ann_df['ratio_height'].mean()
        img_wo = img_new
        # img_new = draw_rect(img_new, bboxes_new,color=[0,255,1])
        # cv2.putText(img_new, label, (int(bboxes_new[2] - 10), int(bboxes_new[3] + 10)), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
        mod_img_ann = 'ann'+mod_img_name
        if((width_mean>0.8)&(height_mean>0.8)):
            all_df=all_df.append(ann_df,ignore_index=True)
            cv2.imwrite(os.path.join(out_dir,mod_img_name),img_wo)
            # cv2.imwrite(os.path.join(out_dir,mod_img_ann),img_new)
    return(all_df)
    
'''
This function converts the bit depth of augmented images according to the bit depth of input image. 
Augmented images with altered bit depth are overwritten on the current augmented images.

    Arguments
    .........
    out_dir: String
             Directory path of augmented images whose bit depth has to be changed.
    img_format: String
                Format of the image
    bit_depth: String
               Bit depth/Mode of the image
               
    Returns
    .......  
    Augmented images with altered bit depth are overwritten on the old augmented images in the same         
    directory
               
'''

def convert_bit_depth(out_dir, img_format, bit_depth):
    print("Convertion in progress!!\n")
    img_ext = '*.'+img_format
    out_img_list=glob.glob(out_dir+img_ext) 
    for i in out_img_list:
        new_i = os.path.join(out_dir,os.path.basename(i))
        image =Image.open(i)
        MOD_IM=image.convert(bit_depth)
        MOD_IM.save(new_i,img_format)

'''This is core function that can perform augmentations on image and bounding boxes.

    Arguments
    .........
    input_img_dir: string
                   Input image directory which contain images and csv files.
                   (eg: '/home/developer/deep_learning/data/ecrash_OD/test_aug/')
    img_format: string
                Format of the input Image. (eg: png) 
    no_of_labels: integer
                  Actual number of labels in original image (eg: 70)                  

    out_dir:  string
                  Directory path where the augmented images and csv are saved. 
                  (eg:  '/home/developer/deep_learning/data/ecrash_OD/test_aug/')
    augment_list: list of augment name
                  List containing the name of the augmentation to be performed. 
                  (eg: 'scale' for scaling,

                       'translate' for translation,
                       'trans_scale' for translation followed by scaling,
                       'rotate' for rotation,
                       'trans_rotate' for translation followed by rotation
                        Default value is ['scale','translate'])

    For augmentation, three parameters are required. They are scale_min, scale_interval, scale_max.
    aug_min: float
             Minimum value to begin the augmentation. (Default: 0.01)
    aug_max: float 
             Maximum value to end the augmentation. (Default: 0.1)
    aug_interval: float
                  Step value to increment the current scale value. (Default: 0.01)  
    convert_image_depth: string
                         This take yes/no as input to choose conversion of image bit depth.
                         If you want to convert, provide 'yes', otherwise provide 'no'.
                         Default value is 'yes'

    Returns
    .......
    Augmented images and corresponding csv file containing image name, transformed bounding box values and   
    labels are created in the specified output directory

    Nothing is retured to calling function.'''
    
def augment_image_bbox(input_img_dir,csv_path,img_format,out_dir,final_csvname,augment_list=['scale'],aug_min=0.03,aug_interval=0.01,aug_max=0.1,convert_image_depth='no'):
    create_dir(out_dir) 
    #img_ext = '*.'+img_format
    #img_list = glob.glob(os.path.join(input_img_dir,img_ext))
    scale_list = np.arange(aug_min,aug_max,aug_interval)
    bbox_df = pd.read_csv(csv_path)
    #bbox_df['img_name'] = [i.split('/')[-1] for i in bbox_df['path']]
    bbox_df['label_width']=bbox_df['xmax']-bbox_df['xmin']
    bbox_df['label_height']=bbox_df['ymax']-bbox_df['ymin']
    img_list= bbox_df['path'].unique()
    PIL_image=Image.open(input_img_dir+img_list[0])
    image_mode =PIL_image.mode
    all_df = pd.DataFrame()
    print((augment_list))
    for augment_name in augment_list:
        print(str(augment_name)+' in progress!!!')
        for i in tqdm(img_list):  
            img_name,img1,bboxes1,label,width,height,orig_label_height,orig_label_width,no_of_labels = read_data(input_img_dir+i,bbox_df)
            img_name = img_name[:img_name.rfind(".")]
            if(augment_name=='scale'):
                for j in range(len(scale_list)):
                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomScale(scale_list[j],diff=False)])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_scale_false_'+str(scale_list[j])+'.'+img_format
                    all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomScale(scale_list[j],diff=True)])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_scale_true_'+str(scale_list[j])+'.'+img_format
                    all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

            elif(augment_name=='translate'): 
                 for j in range(len(scale_list)):
                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomTranslate(scale_list[j],diff=False)])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_trans_false_'+str(scale_list[j])+'.'+img_format
                    all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomTranslate(scale_list[j],diff=True)])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_trans_true_'+str(scale_list[j])+'.'+img_format
                    all_df= write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

            elif(augment_name=='rotate'):
                 for j in range(len(scale_list)):
                    img2=img1.copy()
                    bboxes2=bboxes1.copy()
                    transforms = Sequence([RandomRotate(scale_list[j])])
                    img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                    mod_img_name = img_name +'_rotate_'+str(scale_list[j])+'.'+img_format
                    all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

            elif(augment_name=='trans_scale'):
                for j in range(len(scale_list)):
                    for k in range(len(scale_list)):
                        img2=img1.copy()
                        bboxes2=bboxes1.copy()
                        transforms = Sequence([RandomTranslate(scale_list[k],diff=True),RandomScale(scale_list[j],diff=True)])
                        img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                        mod_img_name = img_name +'_trans_true_'+str(scale_list[k])+'_scale_true_'+str(scale_list[j])+'.'+img_format
                        all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

                        img2=img1.copy()
                        bboxes2=bboxes1.copy()
                        transforms = Sequence([RandomTranslate(scale_list[k],diff=True),RandomScale(scale_list[j],diff=False)])
                        img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                        mod_img_name = img_name +'_trans_true_'+str(scale_list[k])+'_scale_false_'+str(scale_list[j])+'.'+img_format
                        all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

                        img2=img1.copy()
                        bboxes2=bboxes1.copy()
                        transforms = Sequence([RandomTranslate(scale_list[k],diff=False),RandomScale(scale_list[j],diff=False)])
                        img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                        mod_img_name = img_name +'_trans_false_'+str(scale_list[k])+'_scale_false_'+str(scale_list[j])+'.'+img_format
                        all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

                        img2=img1.copy()
                        bboxes2=bboxes1.copy()
                        transforms = Sequence([RandomTranslate(scale_list[k],diff=False),RandomScale(scale_list[j],diff=True)])
                        img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                        mod_img_name = img_name +'_trans_false_'+str(scale_list[k])+'_scale_true_'+str(scale_list[j])+'.'+img_format
                        all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

            elif(augment_name=='trans_rotate'):
                for j in range(len(scale_list)):
                    for k in range(len(scale_list)):
                         img2=img1.copy()
                         bboxes2=bboxes1.copy()
                         img_name,img,bboxes,label = read_data(i,input_img_dir)
                         transforms = Sequence([RandomTranslate(scale_list[k],diff=False),RandomRotate(scale_list[j])])
                         img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                         mod_img_name = img_name +'_tr_false_'+str(scale_list[k])+'_rot_'+str(scale_list[j])+'.'+img_format
                         all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)

                         img2=img1.copy()
                         bboxes2=bboxes1.copy()
                         transforms = Sequence([RandomTranslate(scale_list[k],diff=True),RandomRotate(scale_list[j])])
                         img_new,bboxes_new = transform_data(transforms,img2,bboxes2)
                         mod_img_name = img_name +'_tr_true_'+str(scale_list[k])+'_rot_'+str(scale_list[j])+'.'+img_format
                         all_df=write_data(all_df,img_new,bboxes_new,mod_img_name,label,width,height,orig_label_height,orig_label_width,no_of_labels,out_dir)
                
            all_df=all_df[['path','xmin','ymin','xmax','ymax','label','width','height','label_width',
                   'label_height','original_label_width','original_label_height',
                   'ratio_width','ratio_height']]
            #final_csv_name='augment_all.csv'
            all_df.to_csv(os.path.join(out_dir,final_csvname),index=False)
            if(convert_image_depth=='yes'):
                convert_bit_depth(out_dir,img_format,image_mode)
    print("Augmentation is done successfully!!!")    
    

            
    
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
               


