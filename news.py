#!/bin/python3

import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Read in the data
df = pd.read_csv('news.csv')

# Read first 5 colmuns of data
df.head()
df.shape

# Get labels from the file
labels = df.label
labels.head()

# Split training data and test data
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size = 0.2, random_state = 7)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

tfidf_train= tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Use the PassiveAggresiveClassifier algorithm
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)       

y_pred = pac.predict(tfidf_test)

# Calculate score
score = accuracy_score(y_test, y_pred)
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

print(f"Accuracy: {round(score*100, 2)}%")
