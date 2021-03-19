# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 07:29:51 2021

@author: cand07
"""

import numpy as np
import SimpleITK as sitk 
import os
import pydicom
import cv2
from matplotlib import pyplot as plt



def draw_rectangle(img, pr):
    " computes the non-zero borders, and draws a rectangle around the predicted lung area"
    " the area inside the rectangle can be sent to neural network to process"
    pr = pr.astype(np.float32) 
    vertical_sum = np.sum(pr, axis = 0)
    horizontal_sum = np.sum(pr, axis = 1)
    

    indexes = np.nonzero(vertical_sum)[0]
    border_l = indexes[0]
    border_r = indexes[len(indexes)-1]
    
    indexes = np.nonzero(horizontal_sum)[0]
    border_up = indexes[0]
    border_down = indexes[len(indexes)-1]
    
    start_point = (border_l, border_up)
    end_point = (border_r, border_down)
    color = (1, 1, 0) 
    thickness = 2
    img = cv2.rectangle(img, start_point, end_point, color, thickness) 
    
    return img


def get_np_array_from_itk_image(itk_image):
    '''
    Given a itk image object it returns the numpy array 
    '''
    image_np = np.transpose(sitk.GetArrayFromImage(itk_image), [2,1,0])
    return image_np


def read_dicom_series(data_dir):
    '''
    Read dicom images from a file and return a numpy array
    :param  data_dir
    :return itk image 
    '''

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(data_dir)
    reader.SetFileNames(dicom_names)
    itk_image = reader.Execute()
    
    return itk_image



def show_frames(img, pred, pr):
    
    " show predicted results "
    plt.subplot(131)
    plt.title('input', fontsize = 40)
    plt.axis('off')
    plt.imshow(img, cmap=plt.cm.gray)
        
    plt.subplot(132)
    plt.title('Predicted Lung', fontsize = 40)
    plt.axis('off')
    plt.imshow(pred,cmap='jet')  
        
    " draw rectangle... "
    img = draw_rectangle(img, pr)
    plt.subplot(133)
    plt.title('ProcessArea', fontsize = 40)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    
    plt.savefig('out.png')
    
    plt.show()
    
  
    
    