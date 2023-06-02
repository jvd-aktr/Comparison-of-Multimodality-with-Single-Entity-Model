# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:51:14 2023

@author: akhta
"""

import pandas as pd


# path of the parent folder where data folder is stored
path = r'C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project'  

path_missing_imglist = path + '\images_total' + '\missNeg_total.csv' # change 'pos' and '\neg' for +/- class missing images data
                                                                     # path to missing img list csv
path_txt_data = path + '\laion-400m_data' + '\\negData.xlsx'  # path to access txt data
data_folder = path + '\ImgTxtMapped_data_total'



df_missing = pd.read_csv(path_missing_imglist)
missing_list = list(df_missing['missingImg'])

# # for posImg_total data
# misslist_add = [506, 3930, 6042]

# for negImg_total data
misslist_add = [217, 367, 499, 907, 959, 1317, 1433, 2117, 2461, 2534, 2750, 3041, 3543, 4047, 5125, 5715]
for miss in misslist_add:
    missing_list.append(miss)
    
missing_instances = []
for value in missing_list:
    missing_instances.append(value-1)
    
# load the data excel file into dataframe
df_txt = pd.read_excel(path_txt_data)

# drop instance corresponding to missing images
df_txt_mod = df_txt.drop(missing_instances)

print(df_txt_mod)

#--3--Create a Pandas Excel writer using the save_folder as the output directory--3--#
Excel_file_name = '\\neg_txt_data_total'  # change '\pos' and '\\neg' for +/- class images for name of excel file
df_txt_mod.to_csv(f'{data_folder}{Excel_file_name}.csv', index=False)

