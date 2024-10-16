import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os
import re
import nltk
import string
import streamlit as st

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stopword = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

df = pd.read_csv('labeled_dataset.csv')
print(df.head())

df['labels'] = df['class'].map({
    0: "Hate Speech Detected", 
    1: "Offensive Language Detected", 
    2: "No Hate or Offensive Speech"
})

df = df[['tweet', 'labels']]
print(df.head())

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)
df["tweet"] = df["tweet"].apply(clean)
print(df.head())

x = np.array(df["tweet"])
y = np.array(df["labels"])
cv = CountVectorizer()
x = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

test_data = "hello"
df = cv.transform([test_data]).toarray()
print(clf.predict(df)[0])