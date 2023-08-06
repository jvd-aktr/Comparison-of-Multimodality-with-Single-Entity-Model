# Problem Statement:
A comprehensive study was conducted to compare the performance, challenges, and limitations of Multimodal Machine Learning models and single-entity models. The analysis involved training and testing two data entities, namely textual and image data, using Multimodal architectures such as the Long Short-Term Memory (LSTM) model. Additionally, the same data entities were fitted to state-of-the-art single-entity models, including Support Vector Machine (SVM) for textual data and Convolutional Neural Networks (CNN) for image data. The resulting outputs from both approaches were thoroughly examined to gain deeper insights.

# Dataset:
We chose the LAION-400M dataset, which comprises 400 million image-text pairs that have been filtered with CLIP. The textual data provides descriptions of the image, making it an ideal dataset for conducting a comparative analysis of single-entity and multimodality models. Given the massive size of the dataset and the limitations of our computational resources, we focused solely on the first subset of the dataset, (part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.parquet)

# Data Downloader:
The LAION-400M Dataset does not provide the image and text data directly. It only provided Excel files of image links and corresponding text in the columns. Therefore we wrote a downloader to download those files from the links. The DataDownloader.ipynb file was used to perform this task. The strategy we used and the problems we encountered to download the data for this project are discussed in detail in section 3.1 and 3.2 of the "ProjectReport.pdf" file.

# Data Preprocessing:
Data processing is a critical aspect of any modeling project, and it often consumes a significant amount of time. Our project is no different, as we dedicated substantial effort to preprocess the data for each model. You can find detailed descriptions of the data preprocessing steps for Support Vector Machine (single-entity), Convolution Neural Network (single-entity), and Long Short-Term Memory (multimodal) in sections 3.3, 3.4, and 3.5 of the "ProjectReport.pdf" file, respectively.

# Architecture:
In our analysis of single-entity models, we utilized the Support Vector Machine (SVM) for modeling textual data (see section 4 of the "ProjectReport.pdf" file) and the Convolution Neural Network (CNN) for modeling image data (see section 5 of the "ProjectReport.pdf" file).
Moving on to multimodality model analysis, we combined both textual and image data. For this purpose, we employed the Long Short-Term Memory (LSTM) model to implement co-learning. The input data for the LSTM consists of both textual and image entities, which are fed into separate models. The textual data is processed by a basic LSTM, while the image data is processed by a CNN. The output of the two models is concatenated, and the output layer is designed for binary classification. For further details on multimodal co-learning, please refer to section 6 of the "ProjectReport.pdf" file. 


# Steps to Run the Codes:
## Data Processing:
Download the data, using DataDownloader.ipynb

Read the data, using DataReader.ipynb

To identify which images are missing use:\
python3 missImg_detect.py

Remove the unimportant data (e.g. textual data has bad characters)\
python3 missleadingData_removal.py

Now it is important to map the text and image data together. Use\
python3 txtImgMapping.py

Finally, to train the models better, shuffle the positive and negative class data using\
python3 posneg_shuffle.py

## Single Entity Learning:
The textual data is learned by Fake_news_data_SVM.py. Run\
python3 Fake_news_data_SVM.py

All image data processed into a CSV file for CNN model. Run\
python3 img-csv-like-mnist.py

Then the image data is trained with laion-400m_data_CNN2.py file. Run\
python3 laion-400m_data_CNN2.py

#### Multimodal Learning:
Both the image and textual data are being co-learned in txtImg_mapped_LSTM2.py file. Run\
python3 txtImg_mapped_LSTM2.py

Note: Please update the files with the appropriate file path inside the code before running.

