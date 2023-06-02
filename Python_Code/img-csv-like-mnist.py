# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:59:55 2023

@author: akhta
"""



import os
import numpy as np
import pandas as pd

import cv2
import time

# path of the parent folder where data folder is stored
path = r'C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project\images_total'  

save_folder = path + '\\neg' # path to access images
data_folder = path   # path to save data
path_missing_imglist = path + '\missNeg_total.csv' # change 'pos' and '\neg' for +/- class missing images data

df_missing = pd.read_csv(path_missing_imglist)

IF = "jpg"  # Choose image format

# num_img = 2420 # total number of pos class images
start_img = 1
end_img = 6000 # total number of neg class images

pos_missing_list = list(df_missing['missingImg'])

# # for posImg1 data
# misslist_add = [4, 210, 402, 506, 532, 1187, 1254, 1368, 1378, 1643, 1757, 2031, 2226, 2309, 2341, 2395]

# for negImg1 data
# misslist_add = [217, 226, 367, 426, 436, 499, 601, 681, 751, 790, 907, 959, 1005, 1062, 1123, 1236, 1245, 1305, 1317, 1355, 1433, 1491]

# # for posImg_total data
# misslist_add = [506, 3930, 6042]

# for negImg_total data
misslist_add = [217, 367, 499, 907, 959, 1317, 1433, 2117, 2461, 2534, 2750, 3041, 3543, 4047, 5125, 5715]
for miss in misslist_add:
    pos_missing_list.append(miss)

print(pos_missing_list)


img_table = []

start_time = time.time()
for k in range(start_img, end_img+1):
    if k not in pos_missing_list:
        # set the input filenames
        filename_in = f"0_{k}.{IF}"  # >> use 1 for '+' and 0 for '-' class
    else:
        continue
        
   
    full_filename_in = os.path.join(save_folder, filename_in)  # access edited image from save_folder
    binary_img = cv2.imread(full_filename_in,0)  # read edited image
    
    img_resized = cv2.resize(binary_img, (64, 64))  # Resize the image to 128x128
    
    img_list_2d = np.array(img_resized)
    img_list_1d = np.ravel(img_list_2d)
    
    img_table.append(img_list_1d)


#--2--create dataframes for particle information--2--#
pixel_pos = []
for i in range(1, 65):
    for j in range(1, 65):
        temp = f'{i}x{j}'
        pixel_pos.append(temp)

# dataframe for pixel value information for every image
img_df = pd.DataFrame(img_table, columns = pixel_pos) 
img_df.insert(0, 'label', '0')  # insert class label column >> use 1 for '+' and 0 for '-' class

end_time = time.time()
total_time = end_time - start_time

#--3--Create a Pandas Excel writer using the save_folder as the output directory--3--#
Excel_file_name = '\\negImgPixel_data_total'  # change '\pos' and '\\neg' for +/- class images for name of excel file
img_df.to_csv(f'{data_folder}{Excel_file_name}.csv', index=False)
 
#writer = pd.ExcelWriter(f'{save_folder}\{Excel_file_name}.xlsx') # length of Edited_Images is 14
# Save the workbook and close the Pandas Excel writer
#writer.save()

print(img_df)
print(total_time)


