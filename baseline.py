# src/baseline.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def train_baseline(train_texts, train_labels, val_texts, val_labels):
    vect = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_train = vect.fit_transform(train_texts)
    X_val = vect.transform(val_texts)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_val)
    print("Baseline classification report:")
    print(classification_report(val_labels, preds))
    return vect, clf

def evaluate(clf, vect, test_texts, test_labels):
    X_test = vect.transform(test_texts)
    preds = clf.predict(X_test)
    print(classification_report(test_labels, preds))
    return preds
