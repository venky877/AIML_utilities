# -*- coding: utf-8 -*-
"""
Created on Wen Dec 12 2018
@author: sivaselvi

This script is a utility for test data analysis
Usage: Could be used by calling in jupyter notbooks.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import numpy as np


liby= 'D:/PROJECTS_ROOT/DataServices/Lexis Nexis/ecrash/Modeling/codes/analysis codes/'

files= os.listdir(liby)

gt= pd.read_csv(liby+'test.csv')
pred= pd.read_csv(liby+'test_predictions (1).csv')

class analysis_obj():
    def __init__(self,gt,pred,var):
        self.gt= gt.copy()
        self.pred=pred.copy()
        self.var=var
    @staticmethod 
    def create_coordinates(df):
        df['x']= (df['xmin']+df['xmax'])/2
        df['y']= (df['ymin']+df['ymax'])/2
        return(df)
        
    @staticmethod         
    def return_sub(pred, gt, path, var):
        pred_sub=pred[pred[var]==path]
        gt_sub=gt[gt[var]==path]   
        pred_sub=pred_sub.reset_index(drop= True)
        gt_sub=gt_sub.reset_index(drop= True)    
        pred_sub['serial_no']=range(len(pred_sub))
        gt_sub['serial_no']=range(len(gt_sub))
        #pred_sub=pred_sub[['serial_no','x','y']]
        #gt_sub=gt_sub[['serial_no','x','y']]
        return(pred_sub,gt_sub)
    def analysis(self):      
        self.pred= create_coordinates(self.pred)
        self.gt= create_coordinates(self.gt)
        gtpaths= self.gt[self.var].unique()
        pred_merged= pd.DataFrame()
        for path in tqdm(gtpaths):
            pred_sub, gt_sub=return_sub(self.pred, self.gt, path,self.var)
            if pred_sub.shape[0] > 0:
                nbrs = NearestNeighbors(n_neighbors=min(3,gt_sub.shape[0]), algorithm='brute').fit(gt_sub[['x','y']])
                distances, indices = nbrs.kneighbors(pred_sub[['x','y']])
                pred_tuples=list(pred_sub[['ymin','xmin','ymax','xmax']].itertuples(index=False, name=None))
                gt_tuples=list(gt_sub[['ymin','xmin','ymax','xmax']].itertuples(index=False, name=None))
                len(gt_tuples)
                resultlist=[]
                for j in range(indices.shape[0]):
                    iou=[]
                    for k,l in enumerate(indices[j,:]):
                        iou.append((compute_iou(pred_tuples[j],gt_tuples[l])))
                    if max(iou) > 0.5:
                        resultlist.append(indices[j,np.argmax(iou)])
                    else:
                        resultlist.append(None)
                pred_sub['gt_serialnum']=resultlist
                pred_merged= pred_merged.append(pred_sub)
        return(pred_merged)

pred_merged.info()  
pred.info()
df= df.append(gt)
df.info()
pred_merged.info()
max(arr)
arr = [1, 2, 30, 4, 5]  
arr.append(None)
np.argmax(arr)
_epsilon = 1e-12
def area(boxes):
    """Computes area of boxes.
    Args:
    boxes: Numpy array with shape [N, 4] holding N boxes
    Returns:
    a numpy array with shape [N*1] representing box areas
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.
    Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes
    boxes2: a numpy array with shape [M, 4] holding M boxes
    Returns:
    a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(
      np.zeros(all_pairs_max_ymin.shape),
      all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(
      np.zeros(all_pairs_max_xmin.shape),
      all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths

def intersection_tuples(boxes1, boxes2):
    """Compute pairwise intersection areas between boxes.
    Args:
    boxes1: a numpy array with shape [N, 4] holding N boxes
    boxes2: a numpy array with shape [M, 4] holding M boxes
    Returns:
    a numpy array with shape [N*M] representing pairwise intersection area
    """
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    min_ymax = np.minimum(y_max1, y_max2)
    max_ymin = np.maximum(y_min1, y_min2)
    intersect_heights = np.maximum(   0,min_ymax - max_ymin)
    min_xmax = np.minimum(x_max1, x_max2)
    max_xmin = np.maximum(x_min1, x_min2)
    intersect_widths = np.maximum(0,min_xmax - max_xmin)
    return intersect_heights * intersect_widths


def compute_iou(boxes1, boxes2):
    """Computes intersection-over-union between a pair of boxes. Rounded off to 5 decimal places
    Args:
    boxes1: a tuple with 4 elements in order (y1,x1,y2,x2)
    boxes2: a tuple with 4 elements in order (y1,x1,y2,x2)

    Returns:
    IOU: floating value, rounded to 5 decimal places
    """
    boxes1 = np.expand_dims(np.array(boxes1), axis=0)
    boxes2 = np.expand_dims(np.array(boxes2), axis=0)
    intersect = intersection_tuples(boxes1, boxes2)
    area1 = area(boxes1)
    area2 = area(boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(
      area2, axis=0) - intersect
    iou = intersect / union
    return round(iou[0][0], 5)

def compute_iomin(boxes1, boxes2):
    """Computes intersection-over-union between a pair of boxes. Rounded off to 5 decimal places
    Args:
    boxes1: a tuple with 4 elements in order (y1,x1,y2,x2)
    boxes2: a tuple with 4 elements in order (y1,x1,y2,x2)

    Returns:
    IOU: floating value, rounded to 5 decimal places
    """
    boxes1 = np.expand_dims(np.array(boxes1), axis=0)
    boxes2 = np.expand_dims(np.array(boxes2), axis=0)
    intersect = intersection_tuples(boxes1, boxes2)
    area1 = area(boxes1)
    area2 = area(boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(
      area2, axis=0) - intersect
    iomin = intersect / (min(area1,area2)+0.000001)
    return round(iomin[0][0], 5)


def confusion_matrix_for_object_detection_new(ground_truth_df, prediction_df, iou_threshold=0.5, case_insensitive=True):
    """Generates the confusion matrix given ground truth and prediction df

    Arguments:
        ground_truth_df (DataFrame): containing the following columns: path, xmin, ymin, xmax, ymax, label
        prediction_df (DataFrame): containing the following columns: path, xmin, ymin, xmax, ymax, label
        iou_threshold (float): threshold for IOU. (Default 0.5)
        case_insensitive (bool): Do a case_insensitive comparision of labels. (Default True)
    Returns:
        Two DataFrames:
        1. confusion_matrix_df: contains an additional column called, prediction_type containing, TP, FP, FN
        2. result_df: contains a DataFrame with the following column names: path, precision, recall, f1
    """
    
    
    _prediction_df_comp = prediction_df.copy()
    _ground_truth_df_comp = ground_truth_df.copy()
    _prediction_df_comp["prediction_type"] = "FP"
    _prediction_df_comp["iou"] = 0.0
    _prediction_df_comp["flag"] = 0.0
    if case_insensitive:
        _prediction_df_comp.label = _prediction_df_comp.label.str.upper()
        _ground_truth_df_comp.label = _ground_truth_df_comp.label.str.upper()
    
    pred_grp = _prediction_df_comp.groupby('label')
    gt_grp = _ground_truth_df_comp.groupby('label')
    all_df = pd.DataFrame()
    for k in _ground_truth_df_comp.label.unique():
        print(k)
        _ground_truth_df = gt_grp.get_group(k).reset_index(drop=True)
        try:
            _prediction_df = pred_grp.get_group(k).reset_index(drop=True)
        except Exception as e:
            continue
        miss_df = _prediction_df[~(_prediction_df['path'].isin(_ground_truth_df['path']) )].dropna().reset_index(drop=True)
        group_gt = _ground_truth_df.groupby("path")
        group_pred = _prediction_df.groupby("path")
        cm_list = []
        path_list = []
        for path in tqdm(list(set(_ground_truth_df.path))):
            gt = group_gt.get_group(path).reset_index(drop=True)
            try:
                pred = group_pred.get_group(path).reset_index(drop=True)
                matches = []
                for i in range(len(gt)):
                    for j in range(len(pred)):    
                        iou = compute_iou((gt.ymin[i], gt.xmin[i], gt.ymax[i], gt.xmax[i]), (pred.ymin[j], pred.xmin[j], pred.ymax[j], pred.xmax[j]))
                        matches.append([i, j, iou])
                matches = np.array(matches)
            # Sort list of matches by descending IOU so we can remove duplicate detections
            # while keeping the highest IOU entry.
                matches = matches[matches[:, 2].argsort()[::-1]]
            # Remove duplicate ground truths from the list.
                match_df = pd.DataFrame(matches, columns=['gt','pred','iou'])
                match_df = match_df.sort_values(['gt','iou'], ascending = [True,False])
                match_df_gt = match_df.sort_values('iou').drop_duplicates(subset='gt', keep='last')
                match_df_pred = match_df.sort_values('iou').drop_duplicates(subset='pred', keep='last')
                com_df = pd.concat([match_df_gt,match_df_pred],ignore_index=True)
                com_df = com_df.sort_values('iou', ascending = False).drop_duplicates(subset=['gt','pred'], keep='last')
                matches = com_df.values
                gt_index_list = []
                pred_index_list = []           
                for gt_index, pred_index, IOU in matches:
                    if gt_index not in gt_index_list and pred_index not in pred_index_list:
                        if gt.label[gt_index] == pred.label[pred_index] and IOU >= iou_threshold:
                            pred.at[pred_index, "prediction_type"] = "TP"
                            pred.at[pred_index, "iou"] = IOU
                        elif gt.label[gt_index] != pred.label[pred_index] and IOU >= iou_threshold:
                            pred.at[pred_index, "prediction_type"] = "FP"
                            pred.at[pred_index, "iou"] = IOU
                        elif gt.label[gt_index] == pred.label[pred_index] and IOU < iou_threshold:
                            pred.at[pred_index, "prediction_type"] = "FP"
                            pred.at[pred_index, "iou"] = IOU
                            pred.at[pred_index, "flag"] = 1.0
                        elif gt.label[gt_index] == pred.label[pred_index] and IOU == 0.0:
                            pred.at[pred_index, "prediction_type"] = "FP"
                            pred.at[pred_index, "iou"] = IOU
                        else:
                            pass
                        gt_index_list.append(gt_index)
                        pred_index_list.append(pred_index)
                    elif gt_index not in gt_index_list and pred_index in pred_index_list:
                        pred = pred.append(pd.DataFrame({"path": [gt.path[gt_index]],
                                                             "xmin": [gt.xmin[gt_index]],
                                                             "ymin": [gt.ymin[gt_index]],
                                                             "xmax": [gt.xmax[gt_index]],
                                                             "ymax": [gt.ymax[gt_index]],
                                                             "iou": [IOU],
                                                             "label": [gt.label[gt_index]],
                                                             "prediction_type": ["FN"],
                                                             "flag": ["0.0"]
                                                             }
                                                            ),  ignore_index=True, verify_integrity=True
                                                   )
                        gt_index_list.append(gt_index)
                        pred_index_list.append(pred_index)
                    elif gt_index in gt_index_list and pred_index not in pred_index_list:
                        pred.at[pred_index, "prediction_type"] = "FP"
                        pred.at[pred_index, "iou"] = IOU
                        gt_index_list.append(gt_index)
                        pred_index_list.append(pred_index)                 
                cm_list.append(pred)
        
                
            except Exception as e:
                for x in range(len(gt)):
                    path_list.append((gt.path[x],gt.xmin[x],gt.xmax[x],gt.ymin[x],gt.ymax[x],gt.label[x],"0.0","FN","0.0"))
                continue
            
        FINAL_FN = pd.DataFrame(path_list, columns = ['path','xmin','xmax','ymin','ymax','label','iou','prediction_type','flag'])      
        
        cm_df_old = pd.concat(cm_list, ignore_index=True)
        cm_df = pd.concat([cm_df_old,FINAL_FN, miss_df])
        all_df = pd.concat([all_df,cm_df])
        
    cm_group = all_df.groupby("path")
    PATH = []
    F1 = []
    PRECISION = []
    RECALL = []
    for path in list(set(cm_df.path)):
        PATH.append(path)
        temp_df = cm_group.get_group(path)
        value_counts = temp_df.prediction_type.value_counts()
        precision = value_counts.get("TP", 0)/(value_counts.get("TP", 0) + value_counts.get("FP", 0) + _epsilon)
        recall = value_counts.get("TP", 0)/(value_counts.get("TP", 0) + value_counts.get("FN", 0) + _epsilon)
        PRECISION.append(precision)
        RECALL.append(recall)
        F1.append(2*precision*recall/(precision+recall + _epsilon))

    result_df = pd.DataFrame({"path": PATH,
                              "precision": PRECISION,
                              "recall": RECALL,
                              "f1": F1
                             })

    return (all_df, result_df)

def rollup(df,name):
    df['corpred']=0
    df['pred']=0
    df['gt']=0
    df.loc[(df['prediction_type']=='TP'),'corpred']=1
    df.loc[((df['prediction_type']=='TP') |(df['prediction_type']=='FP')) ,'pred']=1 
    df.loc[((df['prediction_type']=='TP') |(df['prediction_type']=='FN')|((df['prediction_type']=='FP') & (df['flag']==1)))  ,'gt']=1  
    analysis= df.groupby('label')['pred','corpred','gt'].agg(np.sum)
    analysis= analysis.reset_index()
    analysis['precision']=analysis['corpred']/analysis['pred']
    analysis['recall']=analysis['corpred']/analysis['gt']
    analysis['F-Value']=2 * analysis['precision']* analysis['recall']/(analysis['precision'] + analysis['recall'])
    return(analysis)


def path_formation(test_image_path, csv_path):
    data = pd.read_csv(csv_path)
    path_list = data['path'].unique()
    path_list_full = [test_image_path+x for x in path_list]
    return(path_list_full)  
    
def generate_detect(path_list_full,model_path,label_map_path,out_csv_path):
    df= create_detection_df(path_list_full,model_path,label_map_path,score_thresh=0.5)
    df.to_csv(out_csv_path)
    
    
def predict_out(infer_path, csv_path, out_file_label_path,out_label_path):
    testdf_pred= pd.read_csv(infer_path)
    if(testdf_pred.empty == False):
        new_testdf_pred=testdf_pred.rename(columns={'image_path':'path','x1':'xmin','y1':'ymin','x2':'xmax','y2':'ymax','classes':'label'})
        new_testdf_pred['path']=new_testdf_pred['path'].apply(lambda x: x.split("/")[-1])
        testdf_gt = pd.read_csv(csv_path)
        t1,t2 =confusion_matrix_for_object_detection_new(testdf_gt,new_testdf_pred,iou_threshold=0.4)
        t1.to_csv(out_file_label_path)
        test_labellevel1= rollup(t1,"test")
        test_labellevel1.to_csv(out_label_path,index=False)
    
    
def data_analysis(image_path, csv_path, trained_model_path, label_map_path,results_path):
    csv_name = csv_path.split('/')[-1].split('.')[0]
    path_list = path_formation(image_path,csv_path) 
    for (root, dirs, files) in tqdm(os.walk(trained_model_path, topdown=False)):
        for f in files:
            if(f == "frozen_inference_graph.pb"):
                model_name = root.split('/')[-1]
                model_path = os.path.join(root,f)
                det_name = model_name+"_"+csv_name+"_predictions.csv"  
                det_csv_path=os.path.join(results_path,det_name)
                #generate_detect(path_list,model_path,label_map_path,det_csv_path)
                filelabellevel_name = model_name +"_"+ csv_name+"_filelabellevel.csv"
                out_file_label_path = os.path.join(results_path,filelabellevel_name)
                labellevel_name = model_name + "_"+csv_name+"_labellevel.csv"
                out_label_path = os.path.join(results_path,labellevel_name)
                predict_out(det_csv_path,csv_path,out_file_label_path,out_label_path)

                    
                    
def merge_prediction(results_path,analysis_path, name):
    complete_analysis = pd.DataFrame()
    csv_files =  glob.glob(results_path+'*_labellevel.csv')
    for file in csv_files:
        csv_basepath_name = file.split('/')[-1].split('_')[0].split('-')[-1]
        df = pd.read_csv(file)
        df.insert(0,'checkpoint_no', csv_basepath_name)
        complete_analysis = complete_analysis.append(df,ignore_index=True)
    complete_analysis.sort_values(by=['checkpoint_no','F-Value'], ascending=[True,False],inplace=True)
    analysis_filename = name+'_final_analysis.csv'
    complete_analysis.to_csv(os.path.join(analysis_path,analysis_filename),index=False)                      
    
                                                          
                    
image_path = "/home/developer/deep_learning/data/roboreader/rawdata/images/"
trained_model_path = "/home/developer/deep_learning/workspace/roboreader/models/sim14/exported_graphs/"
label_map_path = "/home/developer/deep_learning/data/roboreader/sim01/label_map_all.pbtxt"
train_results_path = "/home/developer/deep_learning/workspace/roboreader/models/sim14/train_results_upd/"
valid_results_path = "/home/developer/deep_learning/workspace/roboreader/models/sim14/valid_results_upd/"
test_results_path = "/home/developer/deep_learning/workspace/roboreader/models/sim14/test_results_upd/"
analysis_path =  "/home/developer/deep_learning/workspace/roboreader/models/sim14/analysis_upd/"
train_csv_path = "/home/developer/deep_learning/data/roboreader/sim01/csv/train_all.csv"
valid_csv_path = "/home/developer/deep_learning/data/roboreader/sim01/csv/valid_all.csv"
test_csv_path = "/home/developer/deep_learning/data/roboreader/sim01/csv/test_all.csv"



if not os.path.exists(train_results_path):
    os.makedirs(train_results_path)    
if not os.path.exists(valid_results_path):
    os.makedirs(valid_results_path)
if not os.path.exists(test_results_path):
    os.makedirs(test_results_path)
if not os.path.exists(analysis_path):
    os.makedirs(analysis_path)

data_analysis(image_path, train_csv_path, trained_model_path, label_map_path,train_results_path)
data_analysis(image_path, valid_csv_path, trained_model_path, label_map_path,valid_results_path)
data_analysis(image_path, test_csv_path, trained_model_path, label_map_path,test_results_path)

merge_prediction(train_results_path,analysis_path,'train')
merge_prediction(valid_results_path,analysis_path,'valid')
merge_prediction(test_results_path,analysis_path,'test')




