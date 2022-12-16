#!/usr/bin/env python
# coding: utf-8

# # Importing and data reading


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


# Read in the train, test data
X_read = pd.read_csv('train_data.csv')
y_read = pd.read_csv('train_results.csv')
X_test_read = pd.read_csv('test_data.csv')



# Narrow the dataset down to the "text" or "target" columns.
X = X_read['text'].copy()
y = y_read['target'].copy()
X_test = X_test_read['text'].copy()


# High-level look at the dataset:
print(f'dataset sizes:')
print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'\ncomposition and distribution of y:')
print(f'y.value counts:\n {y.value_counts(normalize=True)}')


# The target data are almost evenly split between positive and negative, with barely any neutral.

# # Preprocessing

# #### Encode the target values as numbers



le = LabelEncoder()
le.fit(y.unique())
y = le.transform(y) #positive:2, negative:0, neutral:1
y[:5]


# #### Remove punctuation and numbers


def strip_punctuation(text):
    """
    Removes punctuation, digits, single characters and internet-stuff (url, html tags) from a string.
    """
    import re
    import string
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  # remove most punctuation
#     text = text.translate(str.maketrans('','', string.punctuation)) # backup remove most punctuation
    text = text.translate(str.maketrans('', '', string.digits)) # remove digits
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Removes url
    text = re.compile(r'<[^>]+>').sub('', text) #Removes HTML tags:
    text = re.sub(r'\b\w{1}\b', '', text) #re.sub(r"\s+[a-zA-Z]\s+", ' ', text) # Single character removal 
    text = re.sub(r'\s+', ' ', text) # Remove multiple spaces
    return text



# Preprocess both train and test data
X = X.apply(lambda x: strip_punctuation(x))
X_test = X_test.apply(lambda x: strip_punctuation(x))
X


# #### Stem


def stem_text(text, stemmer):
    stemmed_text = ''
    
    if stemmer == 'porter':
        from nltk.stem import PorterStemmer 
        st = PorterStemmer()
    elif stemmer == 'snowball':
        from nltk.stem import SnowballStemmer
        st = SnowballStemmer('english')

    for word in text.split(' '):
        stemmed_text += st.stem(word) + " "
        
    return stemmed_text[:-1]


# Stem data with snowball stemmer
X = X.apply(lambda x: stem_text(x, 'snowball'))
X_test = X_test.apply(lambda x: stem_text(x, 'snowball'))
X


# #### TFIDF Vectorizer


# Vectorize test data
tfvectorizer = TfidfVectorizer(min_df = 1e-6, ngram_range = (1,5), stop_words = None, 
                            lowercase =  True, use_idf= True, sublinear_tf = True)



# Train the vectorizer on the training data and transform the training data and the test data
X_bag = tfvectorizer.fit_transform(X)
print(f'X_bag.shape: {X_bag.shape}')
X_test_bag = tfvectorizer.transform(X_test)



# Train test split the word bag
X_train, X_val, y_train, y_val = train_test_split(X_bag,y, train_size = 0.7, random_state=42)
#See what the resulting size of the train test split is
print(f'X_train shape: {X_train.shape}')



# Make both classifiers
lr = LogisticRegression(max_iter = 2000)
mnb = MultinomialNB()

# Fit the classifiers
lr.fit(X_train,y_train)
mnb.fit(X_train,y_train)

# Score the classifiers
lr_score = lr.score(X_val,y_val)
mnb_score = mnb.score(X_val,y_val)

print(f'lr model validation accuracy: {lr_score}')
print(f'mnb model validation accuracy: {mnb_score}')


# #### Final model training on full train dataset

# Train a final mnb using the full training set
mnb_final = MultinomialNB()
mnb_final.fit(X_bag,y)
# Make predictions using the model trained on the full training set
preds = mnb_final.predict(X_test_bag)
# Turn predictions into the target dataframe
preds = pd.DataFrame({'id': X_test_read['id'], 'target': preds}).set_index('id')
# Save the predictions with an informative name
date = datetime.today().strftime('%Y-%m-%d')
preds.to_csv(f'{date}_naive_bayes_bow_preds_alltrain.csv', )
print(preds)


# Train a final mnb using the full training set
lr_final = LogisticRegression(max_iter = 2000)
lr_final.fit(X_bag,y)
# Make predictions using the model trained on the full training set
preds2 = lr_final.predict(X_test_bag)
# Turn predictions into the target dataframe
preds2 = pd.DataFrame({'id': X_test_read['id'], 'target': preds2}).set_index('id')
# Save the predictions with an informative name
date = datetime.today().strftime('%Y-%m-%d')
preds2.to_csv(f'{date}_logreg_bow_preds_alltrain.csv', )
print(preds2)




