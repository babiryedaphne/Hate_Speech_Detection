# Hate Speech and Offensive Language Detection
Exploring Class Imbalance in Hate Speech and Offensive Language Detection: An Examination of Its Effect on Model Accuracy

## Project Description
This project aims to address the issue of hate speech and offensive language on social media platforms, particularly Twitter. The presence of such content can have severe consequences for individual well-being and societal cohesion. Manual moderation and community reporting are insufficient to handle the vast volume of content generated on these platforms, making automated methods necessary.
In this research, we focus on developing a multi-classification machine learning model for detecting hate speech and offensive language on Twitter. We also explore the impact of imbalanced data on model performance and effective strategies for dealing with such imbalances.

## Data Description 

The dataset used in this study consists of 24,783 tweets collected from Twitter. The data is publicly available on Kaggle and contains seven columns (https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset?resource=download ). The relevant columns for the study are the "class" column, which indicates the target class (hate speech: 0, offensive language: 1, neither: 2), and the "tweet" column, which provides the text data for analysis.

The dataset is imbalanced, with the offensive language category having the highest percentage of tweets (77%), followed by the neither category (16.8%), and the hate speech category with the least number of tweets (5.8%).

## Methods 

The research methodology involves several steps:
1. Data Pre-processing: Cleaning the data, converting text to lowercase, removing numbers and punctuation, tokenizing the text, removing stop words, lemmatization, and joining the cleaned tokens into a single string.
2. Splitting the Data: Splitting the dataset into training and testing subsets using stratified sampling to preserve the class distribution.
3. Word Encoding: Using the Term Frequency-Inverse Document Frequency (TF-IDF) encoding to convert the text data into a numerical form that can be processed by machine learning algorithms.
4. Balancing the Dataset: Applying the Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset and improve model performance on the minority class.
5. Model Exploration: Applying the Naive Bayes, SVM, and Random Forest models to the balanced and imbalanced datasets and evaluating their performance using precision, recall, F1-score, and accuracy.
6. Hyperparameter Optimization: Using grid search to find the best hyperparameters for the Random Forest model on both the balanced and imbalanced datasets.

## Shared Files 
1. Project Code in Python
2. Research paper (pdf)
