## Problem Statement:
A comprehensive study was conducted to compare the performance, challenges, and limitations of Multimodal Machine Learning models and single-entity models. The analysis involved training and testing two data entities, namely textual and image data, using Multimodal architectures such as the Long Short-Term Memory (LSTM) model. Additionally, the same data entities were fitted to state-of-the-art single-entity models, including Support Vector Machine (SVM) for textual data and Convolutional Neural Networks (CNN) for image data. The resulting outputs from both approaches were thoroughly examined to gain deeper insights.

## Dataset:
We chose the LAION-400M dataset, which comprises 400 million image-text pairs that have been filtered with CLIP. The textual data provides descriptions of the image, making it an ideal dataset for conducting a comparative analysis of single-entity and multimodality models. Given the massive size of the dataset and the limitations of our computational resources, we focused solely on the first subset of the dataset, (part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.parquet)

## Data Downloader:
The LAION-400M Dataset does not provide the image and text data directly. It only provided Excel files of image links and corresponding text in the column.
