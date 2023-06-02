# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 01:07:11 2023

@author: akhta
"""

import pandas as pd


# path of the parent folder where data folder is stored
path = r'C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project'  

path_txt_mapped = path + '\ImgTxtMapped_data_total' + '\\neg_txt_data_total.csv' # change 'pos' and '\neg' for +/- class missing images data
                                                                     # path to mapped txt dataset csv
path_imgPxl = path + '\ImgTxtMapped_data_total' + '\\negImgPixel_data_total.csv'  # path to access imgPxl data
data_folder = path + '\ImgTxtMapped_data_total'  # path to save the data in the folder



df_txt = pd.read_csv(path_txt_mapped)
df_txt_desc = df_txt['desc']
df_txt_len = df_txt.shape[0]

df_imgPxl = pd.read_csv(path_imgPxl)

removal_list = []

# # for pos dataset
# for i in range(df_txt_len):
#     line = df_txt_desc[i]

#     if ((' cat ' in line) or (' dog ' in line) or (' Cat ' in line) or (' Dog ' in line)
#         or (' puppy ' in line) or (' Puppy ' in line) or (' kitten ' in line) or (' Kitten ' in line)):
#         removal_list.append(i)

# for neg dataset
for i in range(df_txt_len):
    line = df_txt_desc[i]
    # print(i, line)
    if ((' man ' in line) or (' woman ' in line) or (' girl ' in line) or (' boy ' in line)
        or (' men ' in line) or ('human' in line) or ('people' in line) or ('Man ' in line)):
        removal_list.append(i)
        
df_txt_updated = df_txt.drop(removal_list)
df_imgPxl_updated = df_imgPxl.drop(removal_list)

# print(df_txt_updated.iloc[:10,:])

# print(df_imgPxl_updated.iloc[:10,:])


#--3--Create a Pandas Excel writer using the save_folder as the output directory--3--#
Excel_txtfile_name = '\\neg_txt_data_total_updated'  # change '\pos' and '\\neg' for +/- class images for name of excel file
df_txt_updated.to_csv(f'{data_folder}{Excel_txtfile_name}.csv', index=False)

Excel_imgPxlfile_name = '\\negImgPixel_data_total_updated'  # change '\pos' and '\\neg' for +/- class images for name of excel file
df_imgPxl_updated.to_csv(f'{data_folder}{Excel_imgPxlfile_name}.csv', index=False)

