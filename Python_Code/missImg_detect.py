# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:48:54 2023

@author: akhta
"""

import os
from PIL import Image
import pandas as pd

path = r"C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project\images_total" #replace with the actual path of the folder containing the images
folder_path = path + '\\neg'  # path to access images
data_path = path # path save data

missImg_list = []
for i in range(1, 6001):  # last number is 1 more than total images in folder
    file_name = "0_" + str(i) + ".jpg" #assuming the image files have the extension .jpg >> 1 for pos, 0 for neg
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        # print("Image {} is missing".format(file_name))
        missImg_list.append(i)
    else:
        try:
            img = Image.open(file_path)
            #do something with the image
        except:
            # print("Could not open image {}".format(file_name))
            missImg_list.append(i)

miss_df = pd.DataFrame(missImg_list, columns = ['missingImg'])

excel_name= 'missNeg_total'

miss_df.to_csv(f'{data_path}\{excel_name}.csv', index=False)