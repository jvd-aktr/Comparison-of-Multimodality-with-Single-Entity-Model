# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 01:52:45 2023

@author: akhta
"""

import os
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split 

path = r"C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project" #replace with the actual path of the folder containing the images
path_posImg = path + '\images_total' + '\pos'  # path to access pos images
path_negImg = path + '\images_total' + '\\neg'  # path to access neg images

path_posTxt = path + '\laion-400m_data' + '\posData.xlsx'  # path to access pos text
path_negTxt = path + '\laion-400m_data' + '\\negData.xlsx'  # path to access neg text

data_path = path + '\ImgTxtMapped_data_total\mnist_likedata_total\withoutMissleadingData_LSTM' # path save data

IF = "jpg"  # Choose image format

#*************For Negative Class Dataset (both textual and image)***************#

#-----list for positive class images-----#
missposImg_list = []
for i in range(1, 8001):  # last number is 1 more than total images in folder
    file_name = "1_" + str(i) + ".jpg" #assuming the image files have the extension .jpg 
    file_path = os.path.join(path_posImg, file_name)
    
    if not os.path.exists(file_path):
        missposImg_list.append(i)
    else:
        try:
            img = Image.open(file_path)
            #do something with the image
        except:
            missposImg_list.append(i)

# for image which are small in size, cann't be resize
misslist_add = [506, 3930, 6042]
for miss in misslist_add:
    missposImg_list.append(miss)
    
# creating a list of image name which are will go to next stage
posImg_name_list = []
for i in range(1, 8001):
    if i not in missposImg_list:
        img_name = f'1_{i}'
        posImg_name_list.append(img_name)
    else:
        continue
    
# print(posImg_name_list)

#----------------------------------------#

#-----storing the image data for LSTM model-----#
posImg_table = []

for k in range(1, 8001):
    if k not in missposImg_list:
        # set the input filenames
        filename_in = f"1_{k}.{IF}"  # >> use 1 for '+' and 0 for '-' class
    else:
        continue
         
    full_filename_in = os.path.join(path_posImg, filename_in)  # access edited image from save_folder
    binary_img = cv2.imread(full_filename_in, 0)  # read edited image
    
    img_resized = cv2.resize(binary_img, (64, 64))  # Resize the image to 128x128
    
    posImg_list_2d = np.array(img_resized)
    posImg_list_1d = np.ravel(posImg_list_2d)
    
    posImg_table.append(posImg_list_1d)


pixel_pos = []
for i in range(1, 65):
    for j in range(1, 65):
        temp = f'{i}x{j}'
        pixel_pos.append(temp)

# dataframe for pixel value information for every image
posImg_df = pd.DataFrame(posImg_table, columns = pixel_pos) 
posImg_df.insert(0, 'label', '1')  # insert class label column with value of '1'
#-------------------------------------------------#

#-----creating posTxtImg dataframe-----#

posData_df_lstm = posImg_df.copy()
posData_df_lstm.insert(0, 'Image_name', posImg_name_list)

# index number for missing images
missposImgIndex_list = []
for img in missposImg_list:
    index = img - 1
    missposImgIndex_list.append(index)

posTxt_df = pd.read_excel(path_posTxt)
posTxt_df = posTxt_df.drop(missposImgIndex_list)
posTxt_df.dropna()

posData_df_lstm.insert(1, 'Image_desc', list(posTxt_df['desc']))
#-------------------------------------#

#-----miss leading data removal from positive class-----#
posData_df_lstm_len = posData_df_lstm.shape[0]
posTxt_desc = posData_df_lstm['Image_desc']
posRemoval_list = []


for i in range(posData_df_lstm_len):
    line = posTxt_desc[i]

    if ((' cat ' in line) or (' dog ' in line) 
        or (' Cat ' in line) or (' Dog ' in line)
        or (' puppy ' in line) or (' Puppy ' in line) 
        or (' kitten ' in line) or (' Kitten ' in line)):
        posRemoval_list.append(i)

posData_df_lstm = posData_df_lstm.drop(posRemoval_list)
posData_df_lstm.dropna()
print(posData_df_lstm)
#------------------------------------------------------#

#-----save the data as csv file in data_path file directory--#
Excel_posfile_name = '\posTxtImg_mapped_missLeadingRemoved'
posData_df_lstm.to_csv(f'{data_path}{Excel_posfile_name}.csv', index=False)
#------------------------------------------------------------#

#*******************************************************************************#

#*************For Negative Class Dataset (both textual and image)***************#

#-----list for negative class images-----#
missnegImg_list = []
for i in range(1, 6001):  # last number is 1 more than total images in folder
    file_name = "0_" + str(i) + ".jpg" #assuming the image files have the extension .jpg 
    file_path = os.path.join(path_negImg, file_name)
    
    if not os.path.exists(file_path):
        missnegImg_list.append(i)
    else:
        try:
            img = Image.open(file_path)
            #do something with the image
        except:
            missnegImg_list.append(i)
            
# for image which are small in size, cann't be resize
misslist_add = [217, 367, 499, 907, 959, 1317, 1433, 2117, 2461, 2534, 2750, 3041, 3543, 4047, 5125, 5715]
for miss in misslist_add:
    missnegImg_list.append(miss)
    
# creating a list of image name which are will go to next stage
negImg_name_list = []
for i in range(1, 6001):
    if i not in missnegImg_list:
        img_name = f'0_{i}'
        negImg_name_list.append(img_name)
    else:
        continue
#----------------------------------------#

#-----storing the image data for LSTM model-----#
negImg_table = []

for k in range(1, 6001):
    if k not in missnegImg_list:
        # set the input filenames
        filename_in = f"0_{k}.{IF}"  # >> use 1 for '+' and 0 for '-' class
    else:
        continue
         
    full_filename_in = os.path.join(path_negImg, filename_in)  # access edited image from save_folder
    binary_img = cv2.imread(full_filename_in, 0)  # read edited image
    
    img_resized = cv2.resize(binary_img, (64, 64))  # Resize the image to 128x128
    
    negImg_list_2d = np.array(img_resized)
    negImg_list_1d = np.ravel(negImg_list_2d)
    
    negImg_table.append(negImg_list_1d)


pixel_neg = []
for i in range(1, 65):
    for j in range(1, 65):
        temp = f'{i}x{j}'
        pixel_neg.append(temp)

# dataframe for pixel value information for every image
negImg_df = pd.DataFrame(negImg_table, columns = pixel_neg) 
negImg_df.insert(0, 'label', '0')  # insert class label column with value of '0'
#-------------------------------------------------#

#-----creating negTxtImg dataframe-----#

negData_df_lstm = negImg_df.copy()
negData_df_lstm.insert(0, 'Image_name', negImg_name_list)

# index number for missing images
missnegImgIndex_list = []
for img in missnegImg_list:
    index = img - 1
    missnegImgIndex_list.append(index)

negTxt_df = pd.read_excel(path_negTxt)
negTxt_df = negTxt_df.drop(missnegImgIndex_list)
negTxt_df.dropna()

negData_df_lstm.insert(1, 'Image_desc', list(negTxt_df['desc']))
#--------------------------------------#

#-----miss leading data removal from negative class-----#
negData_df_lstm_len = negData_df_lstm.shape[0]
negTxt_desc = negData_df_lstm['Image_desc']
negRemoval_list = []


for i in range(negData_df_lstm_len):
    line = negTxt_desc[i]

    if ((' man ' in line) or (' woman ' in line) or (' girl ' in line) or (' boy ' in line)
        or (' men ' in line) or ('human' in line) or ('people' in line) or ('Man ' in line)):
        negRemoval_list.append(i)

negData_df_lstm = negData_df_lstm.drop(negRemoval_list)
negData_df_lstm.dropna()
print(negData_df_lstm)
#--------------------------------------------------------# 

#-----save the data as csv file in data_path file directory--#
Excel_negfile_name = '\\negTxtImg_mapped_missLeadingRemoved'
negData_df_lstm.to_csv(f'{data_path}{Excel_negfile_name}.csv', index=False)
#------------------------------------------------------------#

#*******************************************************************************#

#*****for creating all positive and negative data*****#

# concatenate the dataframes 
posnegData_df_lstm = pd.concat([posData_df_lstm, negData_df_lstm])

# shuffle the instances
posnegData_df_lstm = posnegData_df_lstm.sample(frac=1, random_state=42).reset_index(drop=True)
posnegData_df_lstm = posnegData_df_lstm.dropna()
print(posnegData_df_lstm)

# assume your big dataframe is called 'big_df'
train_df, test_df = train_test_split(posnegData_df_lstm, test_size=0.1, random_state=42)

#--3--Create a Pandas Excel writer using the save_folder as the output directory--3--#
Excel_file_name1 = '\\train_missLeading_Removed'  
train_df.to_csv(f'{data_path}{Excel_file_name1}.csv', index=False)

Excel_file_name2 = '\\test_missLeading_Removed'
test_df.to_csv(f'{data_path}{Excel_file_name2}.csv', index=False)

#*****************************************************#