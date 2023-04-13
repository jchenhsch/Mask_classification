import glob
import shutil
import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical,image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from image_load import input_img
import random
import pandas as pd


def test_data_generator(src_dir_mask,src_dir_without_mask,dst_dir,with_mask_limit,without_mask_limit):
    """
    input: src_dir_mask: directory that stores the masked images
           src_dir_without_mask: directory that stores the unmasked images
           with_mask_limit: number of images that take in from the masked images folder
           without_mask_limit: number of images that take in from the unmasked images folder
           dst_dir: the directory that mixed two label images in one folder for testing
           
    output: None ; side effect: form a testing folder in the dst_dir
    """
    for f in os.listdir(dst_dir):
        os.remove(os.path.join(dst_dir,f))# delete the images that previous in the directory as well as DS.store
        
    with_mask_count=0
    for jpgfile in glob.iglob(os.path.join(src_dir_mask, "*.jpg")):
        shutil.copy(jpgfile, dst_dir)
        with_mask_count=with_mask_count+1
        if with_mask_count==with_mask_limit:
            break
    ####
    without_mask_count=0
    for jpgfile in glob.iglob(os.path.join(src_dir_without_mask, "*.jpg")):
        shutil.copy(jpgfile, dst_dir)
        without_mask_count=without_mask_count+1
        if without_mask_count==without_mask_limit:
            break


def img_to_df(dst_dir):

    """
    input: dst_dir: destination directory that stores the testing images (mixed label)
    
    output: img_df, convert the all tesing images files in the directory into a PandaFrame img_df
    """

    jpgfile_lst=[]
    for jpgfile in glob.iglob(os.path.join(dst_dir, "*.jpg")):
        #print(jpgfile)
        if "with_mask" in jpgfile:
            value=(input_img(jpgfile),0)
            jpgfile_lst.append(value)
        else:
            value=(input_img(jpgfile),1)
            jpgfile_lst.append(value)
    column_name=["digit_img_file","class"]
    img_df=pd.DataFrame(jpgfile_lst,columns=column_name)
    return img_df
