# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:45:17 2020

@author: 205557
"""
from PIL import Image as pil_image
from pylab import *
import os
import numpy as np

def predict_from_images(image_path, model, target_dimension,func, color_order):
    test_image = pil_image.open(image_path).convert(color_order)
    test_image=test_image.resize(target_dimension, pil_image.BICUBIC)
    X= array(test_image)
    # prepare the image for the VGG model
    X = np.expand_dims(X, 0)
    X= func(X)
    pred= model.predict(X)
    return pred
