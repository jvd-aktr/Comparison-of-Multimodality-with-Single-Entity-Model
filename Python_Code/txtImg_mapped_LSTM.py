# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 00:47:17 2023

@author: akhta
"""
import pandas as pd
from sklearn.model_selection import train_test_split 


# Load the dataframe
path = r'C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project\ImgTxtMapped_data_total\mnist_likedata_total'
path_posTxt = path + '\withoutMissleadingData' + '\pos_txt_data_total_updated.csv'
path_negTxt = path + '\withoutMissleadingData' + '\\neg_txt_data_total_updated.csv'
path_posImg = path + '\withoutMissleadingData' + '\posImgPixel_data_total_updated.csv'
path_negImg = path + '\withoutMissleadingData' + '\\negImgPixel_data_total_updated.csv'
data_folder = path + '\withoutMissleadingData_LSTM'

df_posTxt = pd.read_csv(path_posTxt)
df_negTxt = pd.read_csv(path_negTxt)

df_posImg = pd.read_csv(path_posImg)
df_negImg = pd.read_csv(path_negImg)

# create a datframe for LSTM model
df_lstm_pos = df_posImg
df_lstm_pos.insert(1, 'imgDesc', df_posTxt['desc'])

df_lstm_neg = df_negImg
df_lstm_neg.insert(1, 'imgDesc', df_negTxt['desc'])

print(df_lstm_pos)
# # shuffle the instances
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# df = df.dropna()