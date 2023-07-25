#!/usr/bin/env python
# coding: utf-8
### 
### Last modification (DD/MM/YYY) : 20/07/2023
###
####################################################################################################
###                                                                                              ###
### This python script is an example for Image augmentation saved into a new folder              ###  
###                                                                                              ###
####################################################################################################

####################################################################################################
#                                                                                                  #
# Machine Learning for Materials Informatics                                                       #
#                                                                                                  #
# #### Markus J. Buehler, MIT                                                                      #
#                                                                                                  #
####################################################################################################

### Import libraries for the image augmentation ####################################################

import keras

import cv2

import os

import glob

import tensorflow as tf
#import tensorflow_datasets as tfds

from tensorflow.keras import layers

### Set the path to the directory containing input data ############################################

#in_dir="./dataset/leaf_data_single/"
in_dir="./dataset/leaf_data/"

### Set the path to the directory containing output data ###########################################

out_dir=r'./dataset/leaf_data_augmented/'

### Set how many times we go through the set of original images for augmenting them ################

repeats= 12 

### Create the output directory if it does not exist ###############################################

if not os.path.exists(out_dir):

        os.mkdir(out_dir)

### Set the model for data augmentation ############################################################

data_augmentation = tf.keras.Sequential([
                    layers.RandomFlip("horizontal_and_vertical"),
                    layers.RandomRotation(0.2, 
                    fill_mode='reflect'),
                    layers.RandomZoom(height_factor=(-0.6, -0.2), 
                    fill_mode='nearest'),
#                   layers.RandomTranslation(height_factor=0.2,width_factor=0.2),#
                    ])

### Set the source directory of all images #########################################################

img_dir = in_dir

### Build the full path to all images ##############################################################

data_path = os.path.join(img_dir,'*jpg')

files = glob.glob(data_path)

#data = []

### Apply the augmentation of the initial data set #################################################

i = 0

for jj in range (repeats):

    for f1 in files:

        img = cv2.imread(f1)

        print(f1)

        x = tf.keras.utils.img_to_array(img)

        augmented_image = data_augmentation(x)

        cv2.imwrite(f"{out_dir}{i}_.jpg", augmented_image.numpy())

        i = i + 1
 
### Build the list of files ########################################################################

i = 0

path, dirs, files = next(os.walk(in_dir))

out_path, out_dirs, out_files = next(os.walk(out_dir))

### Count the number of files that were found ######################################################

file_count = len(files) #to find number of files in folder

print('The number of counted input files is {}'.format(file_count))

print(' ')

out_file_count = len(out_files)

print('The number of counted output files is {}'.format(out_file_count))

print(' ')

ratio = out_file_count / file_count

print('ratio : {} ({})'.format(ratio,repeats))

print(' ');

###

print(x[0])

###

#file_count = 5 #how many images you want....

#for batch in datagen.flow (x, batch_size=1, save_to_dir = out_dir,save_prefix="a",save_format='jpg'):
###for batch in augmented_image.flow (x, batch_size=1, save_to_dir = out_dir,save_prefix="a",save_format='jpg'):

#    i+=1

#    if i==file_count:

#      break

