# -*- coding: utf-8 -*-


# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# load the dataset
path = r'C:\Users\akhta\OneDrive - New Mexico State University\MAE_NMSU\2023\2023 Spring\CS519-Machine Learning\Project\AML_Project\ImgTxtMapped_data_total\mnist_likedata_total\withoutMissleadingData'

path_pos = path + '\pos_txt_data_total_updated.csv'
path_neg = path + '\\' + "neg_txt_data_total_updated.csv"

df_pos = pd.read_csv(path_pos)
#df_true['label'] = 1  # label all real news as 1.
df_neg = pd.read_csv(path_neg)
#df_fake['label'] = 0  # label all fake news as 0. 

# concatenate the dataframes
df = pd.concat([df_pos, df_neg])

# shuffle the instances
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.dropna()
print(df)
# split the dataset into 64% training, 16% validation, and 20% testing
# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['desc'], 
                                                    df['class'], 
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True)

# split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.2,
                                                  random_state=42)


# convert the text data into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
X_val = vectorizer.transform(X_val)

# train the SVM model
svm = LinearSVC()
svm.fit(X_train, y_train)

# make predictions on the testing set
y_pred_test = svm.predict(X_test)
y_pred_train = svm.predict(X_train)
y_pred_val = svm.predict(X_val)

# calculate the accuracy of the model
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Train Accuracy:", accuracy_train)

accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test Accuracy:", accuracy_test)

accuracy_val = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", accuracy_val)
