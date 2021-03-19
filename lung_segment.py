# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:36:48 2021
            
@author: cand07
"""

import os
import numpy as np
import SimpleITK as sitk
from skimage import io, exposure, img_as_float, transform, morphology
from matplotlib import pyplot as plt
from tensorflow.keras.models import model_from_json
from utils import *


PATH = os.getcwd()
img_path = PATH + '/Data/'

" list all frames in the series "
frame_list = os.listdir(img_path)

" load lung segment model and weights... (model U-net) " 
json_file = open('segment_model.json', 'r') 
loaded_model_json = json_file.read() 
json_file.close() 
model = model_from_json(loaded_model_json)
    
" load weights into the model " 
model.load_weights('model.400.hdf5') 
print("Loaded model from disk")

"input shape..." 
seg_w = model.input.shape[1]         
seg_h = model.input.shape[2]    
im_shape = (seg_w,seg_h)    

    
for file_name in frame_list: 
    
    fig = plt.figure(figsize=(30, 10))
    print(file_name)
        
    " check file type " 
    main_fn, ext_fn = os.path.splitext(file_name)
    
    " read image file as numpy array "
    file_name_with_path = img_path + file_name 
    if(ext_fn == '.dcm'): 
        itk_image = sitk.ReadImage(file_name_with_path)
        img = sitk.GetArrayFromImage(itk_image)
        img = np.squeeze(img)
    else: 
        img = img_as_float(io.imread(file_name_with_path))
            
    if(len(img.shape) == 3): 
        img = img[:,:,0]
        
    "pre-process"
    img = transform.resize(img, im_shape)
    orig_img = img 
    img = exposure.equalize_hist(img)
        
    " normalize "
    img -= img.mean()
    img /= img.std()
        
    " Predict the lung region "
    img = np.expand_dims(img, -1)
    img = np.expand_dims(img,axis = 0)
    pred = model.predict(img)[..., 0].reshape(im_shape)
    img = np.squeeze(img)
    
    
    " 0-1 mask conversion "
    pr = pred > 0.5
    pr = morphology.remove_small_objects(pr, int(0.1*im_shape[1]))
    
  
    " show predicted results "  
    show_frames(orig_img, pred, pr)
    