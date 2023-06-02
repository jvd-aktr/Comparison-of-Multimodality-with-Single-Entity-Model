# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:47:10 2023

@author: akhta
"""
import pandas as pd
from sklearn.model_selection import train_test_split 

# Load the dataframe
path_pos = r"C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project\ImgTxtMapped_data_total\mnist_likedata_total\withoutMissleadingData\posImgPixel_data_total_updated.csv"
path_neg = r"C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project\ImgTxtMapped_data_total\mnist_likedata_total\withoutMissleadingData\negImgPixel_data_total_updated.csv"
data_folder = r'C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project\ImgTxtMapped_data_total\mnist_likedata_total'

df_pos = pd.read_csv(path_pos)
df_neg = pd.read_csv(path_neg)

# concatenate the dataframes 
df = pd.concat([df_pos, df_neg])

# shuffle the instances
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.dropna()
print(df)

# assume your big dataframe is called 'big_df'
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

#--3--Create a Pandas Excel writer using the save_folder as the output directory--3--#
Excel_file_name1 = '\\train_updated'  
train_df.to_csv(f'{data_folder}{Excel_file_name1}.csv', index=False)

Excel_file_name2 = '\\test_updated'  
test_df.to_csv(f'{data_folder}{Excel_file_name2}.csv', index=False)