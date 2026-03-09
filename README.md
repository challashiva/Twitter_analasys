# 📊 Twitter Sentiment Analysis using NLP








## A Natural Language Processing (NLP) project that analyzes Twitter data and classifies tweets as Positive or Negative sentiment using a Naive Bayes classifier.

This project demonstrates the complete NLP pipeline, including:

- Data preprocessing

- Stopword removal

- Feature extraction

- WordCloud visualization

- Sentiment classification

# 📌 Project Overview

Social media platforms like Twitter contain massive amounts of public opinion data.
This project analyzes tweets to determine whether a tweet expresses positive or negative sentiment.

The model is trained using NLTK's Naive Bayes classifier, one of the most widely used algorithms for text classification.

# 🚀 Features

✔ Text preprocessing and cleaning

✔ Stopword removal using NLTK

✔ Feature extraction using Bag-of-Words

✔ WordCloud visualization for positive and negative tweets

✔ Sentiment classification using Naive Bayes

✔ Simple evaluation metrics

#🛠 Technologies Used
| Technology   | Purpose                      |
| ------------ | ---------------------------- |
| Python       | Programming language         |
| Pandas       | Data processing              |
| NumPy        | Numerical computing          |
| NLTK         | Natural Language Processing  |
| Scikit-Learn | Dataset splitting            |
| WordCloud    | Word frequency visualization |
| Matplotlib   | Plotting and visualization   |


#📂 Dataset

The dataset used in this project:

Sentiment.csv

Dataset contains two columns:

| Column    | Description                                     |
| --------- | ----------------------------------------------- |
| text      | Tweet content                                   |
| sentiment | Sentiment label (Positive / Negative / Neutral) |


Example:

| text                   | sentiment |
| ---------------------- | --------- |
| I love this phone      | Positive  |
| Worst customer support | Negative  |
| It is okay             | Neutral   |


⚠ Neutral tweets are removed during training to simplify the model into binary classification.

⚙️ NLP Pipeline

The project follows a standard Natural Language Processing pipeline.

Raw Tweets

     ↓

Text Cleaning

     ↓

Stopword Removal

     ↓

Feature Extraction

     ↓

Model Training

     ↓

Prediction

     ↓
     
Evaluation

#🧹 Text Preprocessing

Tweets contain noisy data such as:

- URLs

- Hashtags

- Mentions

- Retweet symbols

Example tweet:

  RT @user: I love this product! https://link.com #awesome

After preprocessing:

  love product awesome

Steps applied:

- Convert text to lowercase

- Remove URLs

- Remove mentions (@username)

- Remove hashtags (#topic)

- Remove stopwords

- Filter short words

# ☁ WordCloud Visualization

WordCloud is used to visualize the most frequent words in tweets.

Positive Tweet Words

### Examples:

love

great

awesome

happy

best

Negative Tweet Words

### Examples:

bad

worst

hate

problem

issue

Visualization helps understand common sentiment patterns in the dataset.

# 🔎 Feature Extraction

Machine learning models cannot understand raw text.

Therefore tweets are converted into feature dictionaries using a Bag-of-Words approach.

### Example tweet:

I love this phone

### Feature representation:

{

contains(love): True

contains(phone): True

contains(bad): False

}

This allows the model to learn word-sentiment relationships.

#🤖 Model Training

The model used:

### Naive Bayes Classifier

Naive Bayes works well for text classification because it:

✔ Is fast

✔ Handles large vocabularies

✔ Works well with small datasets

Training code:

training_set = nltk.classify.apply_features(extract_features, tweets)

classifier = nltk.NaiveBayesClassifier.train(training_set)

# 📊 Model Evaluation

The trained model predicts sentiment for test tweets.

### Example evaluation output:

[Negative]: 100/82

[Positive]: 120/97

Meaning:

82 out of 100 negative tweets predicted correctly

97 out of 120 positive tweets predicted correctly

# 🖥 Example Code

Import Libraries

import numpy as np

import pandas as pd

import nltk

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

from wordcloud import WordCloud

import matplotlib.pyplot as plt

# ⚡ Installation

Install required libraries:

pip install pandas

pip install numpy

pip install nltk

pip install scikit-learn

pip install matplotlib

pip install wordcloud

### Download NLTK resources:

import nltk

nltk.download('stopwords')

# ▶ How to Run

1️⃣ Clone the repository

git clone https://github.com/challashiva/Twitter_analasys.git

2️⃣ Navigate to project

cd Twitter_analasys

3️⃣ Run the notebook or script

analasys_report.ipynb

# 📈 Future Improvements

This project can be improved by using more advanced NLP techniques:

- TF-IDF Vectorization

- Logistic Regression

- Support Vector Machines

- Deep Learning (LSTM)

- Transformer models (BERT)

- Better evaluation metrics (Precision, Recall, F1-Score)

# 📚 Learning Outcomes

This project demonstrates:

- Natural Language Processing fundamentals

- Text preprocessing techniques

- Feature extraction methods

- Machine learning classification

- Data visualization using WordCloud
