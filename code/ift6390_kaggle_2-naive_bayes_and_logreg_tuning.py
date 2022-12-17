#!/usr/bin/env python
# coding: utf-8

# # Importing and data reading


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from datetime import datetime
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



# Make a second copy to show the effect of different hyper-parameters/treatments
X_ = X.copy()
y_ = y.copy()



# High-level look at the dataset:
print(f'dataset sizes:')
print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'\ncomposition and distribution of y:')
print(f'y.value counts:\n {y.value_counts(normalize=True)}')


# The target data are almost evenly split between positive and negative, with barely any neutral.



# Make a dataframe for storing hyperparameter tuning results
results = pd.DataFrame({'hyperparameter': 0, 'logreg accuracy': 0, 'mnb accuracy': 0}, index=[0])
results


# # Preprocessing

# #### Encode the target values as numbers



le = LabelEncoder()
le.fit(y.unique())
y = le.transform(y) #positive:2, negative:0, neutral:1
y[:5]


# ### Effect of alpha on mnb validation accuracy



for alpha in [1000,1,1/1000]:
    X = X_.copy()
    y = y_.copy()
    # Vectorize test data
    tfvectorizer = TfidfVectorizer(min_df = 1e-6, ngram_range = (1,1), stop_words = None, 
                                   lowercase =  True, use_idf= True, sublinear_tf = True)
    # Train the vectorizer on the training data and transform the training data
    X_bag = tfvectorizer.fit_transform(X)
    print(f'X_bag.shape: {X_bag.shape}')
    # Train test split the word bag
    X_train, X_val, y_train, y_val = train_test_split(X_bag,y, train_size = 0.7, random_state=42)
    #See what the resulting size of the train test split is
    print(f'X_train shape: {X_train.shape}')

    # Make both classifiers
#     lr = LogisticRegression(max_iter = 1000)
    mnb = MultinomialNB(alpha = alpha)

    # Fit the classifiers
#     lr.fit(X_train,y_train)
    mnb.fit(X_train,y_train)

    # Score the classifiers
#     lr_score = lr.score(X_val,y_val)
    mnb_score = mnb.score(X_val,y_val)

#     print(f'lr model validation accuracy: {lr_score}')
    print(f'mnb model validation accuracy: {mnb_score}')

    # Add to the results dataframe
    new = pd.Series({'hyperparameter': f'mnb alpha: {alpha}',
                     'logreg accuracy': f'-', 
                     'mnb accuracy': f'{mnb_score}'})
    results = pd.concat([results, new.to_frame().T], ignore_index = True)



results


# ### Effect of C on lr validation accuracy



for c in [1000,1,1/1000]:
    X = X_.copy()
    y = y_.copy()
    # Vectorize test data
    tfvectorizer = TfidfVectorizer(min_df = 1e-6, ngram_range = (1,1), stop_words = None, 
                                   lowercase =  True, use_idf= True, sublinear_tf = True)
    # Train the vectorizer on the training data and transform the training data
    X_bag = tfvectorizer.fit_transform(X)
    print(f'X_bag.shape: {X_bag.shape}')
    # Train test split the word bag
    X_train, X_val, y_train, y_val = train_test_split(X_bag,y, train_size = 0.7, random_state=42)
    #See what the resulting size of the train test split is
    print(f'X_train shape: {X_train.shape}')

    # Make both classifiers
    lr = LogisticRegression(C=c,max_iter = 1000)
#     mnb = MultinomialNB(alpha = alpha)

    # Fit the classifiers
    lr.fit(X_train,y_train)
#     mnb.fit(X_train,y_train)

    # Score the classifiers
    lr_score = lr.score(X_val,y_val)
#     mnb_score = mnb.score(X_val,y_val)

    print(f'lr model validation accuracy: {lr_score}')
#     print(f'mnb model validation accuracy: {mnb_score}')

    # Add to the results dataframe
    new = pd.Series({'hyperparameter': f'lr C: {c}',
                     'logreg accuracy': f'{lr_score}', 
                     'mnb accuracy': f'-'})
    results = pd.concat([results, new.to_frame().T], ignore_index = True)


results


# #### Punctuation removing function

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


# ### Effect of removing punctuation on validation accuracy


for punc in [True, False]:
    X = X_.copy()
    y = y_.copy()
    if punc:
        # Preprocess both train and test data
        X = X.apply(lambda x: strip_punctuation(x))
        # Vectorize test data
        tfvectorizer = TfidfVectorizer(min_df = 1e-6, ngram_range = (1,1), stop_words = None, 
                                       lowercase =  True, use_idf= True, sublinear_tf = True)
        # Train the vectorizer on the training data and transform the training data
        X_bag = tfvectorizer.fit_transform(X)
        print(f'X_bag.shape: {X_bag.shape}')
        # Train test split the word bag
        X_train, X_val, y_train, y_val = train_test_split(X_bag,y, train_size = 0.7, random_state=42)
        #See what the resulting size of the train test split is
        print(f'X_train shape: {X_train.shape}')

        # Make both classifiers
        lr = LogisticRegression(max_iter = 1000)
        mnb = MultinomialNB()

        # Fit the classifiers
        lr.fit(X_train,y_train)
        mnb.fit(X_train,y_train)

        # Score the classifiers
        lr_score = lr.score(X_val,y_val)
        mnb_score = mnb.score(X_val,y_val)

        print(f'lr model validation accuracy: {lr_score}')
        print(f'mnb model validation accuracy: {mnb_score}')

        # Add to the results dataframe
        new = pd.Series({'hyperparameter': f'punctuation removed: {punc}',
                         'logreg accuracy': f'{lr_score}',
                         'mnb accuracy': f'{mnb_score}'})
        results = pd.concat([results, new.to_frame().T], ignore_index = True)
    else:
        # Vectorize test data
        tfvectorizer = TfidfVectorizer(min_df = 1e-6, ngram_range = (1,1), stop_words = None, 
                                       lowercase =  True, use_idf= True, sublinear_tf = True)
        # Train the vectorizer on the training data and transform the training data
        X_bag = tfvectorizer.fit_transform(X)
        print(f'X_bag.shape: {X_bag.shape}')
        # Train test split the word bag
        X_train, X_val, y_train, y_val = train_test_split(X_bag,y, train_size = 0.7, random_state=42)
        #See what the resulting size of the train test split is
        print(f'X_train shape: {X_train.shape}')

        # Make both classifiers
        lr = LogisticRegression(max_iter = 1000)
        mnb = MultinomialNB()

        # Fit the classifiers
        lr.fit(X_train,y_train)
        mnb.fit(X_train,y_train)

        # Score the classifiers
        lr_score = lr.score(X_val,y_val)
        mnb_score = mnb.score(X_val,y_val)

        print(f'lr model validation accuracy: {lr_score}')
        print(f'mnb model validation accuracy: {mnb_score}')

        # Add to the results dataframe
        new = pd.Series({'hyperparameter': f'punctuation removed: {punc}',
                         'logreg accuracy': f'{lr_score}',
                         'mnb accuracy': f'{mnb_score}'})
        results = pd.concat([results, new.to_frame().T], ignore_index = True)


results


# #### Stemming function


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


# ### Effect of stemming on validation accuracy



for s in [True,False]:
    X = X_.copy()
    y = y_.copy()
    if s:
        for stem_type in ['snowball','porter']:
            # Vectorize test data
            tfvectorizer = TfidfVectorizer(min_df = 1e-6, ngram_range = (1,1), stop_words = None, 
                                         lowercase =  True, use_idf= True, sublinear_tf = True)
            # Apply stemming
            X = X.apply(lambda x: stem_text(x, stem_type))
            # Train the vectorizer on the training data and transform the training data
            X_bag = tfvectorizer.fit_transform(X)
            print(f'X_bag.shape: {X_bag.shape}')
            # Train test split the word bag
            X_train, X_val, y_train, y_val = train_test_split(X_bag,y, train_size = 0.7, random_state=42)
            #See what the resulting size of the train test split is
            print(f'X_train shape: {X_train.shape}')

            # Make both classifiers
            lr = LogisticRegression(max_iter = 1000)
            mnb = MultinomialNB()

            # Fit the classifiers
            lr.fit(X_train,y_train)
            mnb.fit(X_train,y_train)

            # Score the classifiers
            lr_score = lr.score(X_val,y_val)
            mnb_score = mnb.score(X_val,y_val)

            print(f'lr model validation accuracy: {lr_score}')
            print(f'mnb model validation accuracy: {mnb_score}')

            # Add to the results dataframe
            new = pd.Series({'hyperparameter': f'stemming: {stem_type}',
                             'logreg accuracy': f'{lr_score}', 
                             'mnb accuracy': f'{mnb_score}'})
            results = pd.concat([results, new.to_frame().T], ignore_index = True)
    else:
        # Vectorize test data
        tfvectorizer = TfidfVectorizer(min_df = 1e-6, ngram_range = (1,1), stop_words = None, 
                                       lowercase =  True, use_idf= True, sublinear_tf = True)
        # Train the vectorizer on the training data and transform the training data
        X_bag = tfvectorizer.fit_transform(X)
        print(f'X_bag.shape: {X_bag.shape}')
        # Train test split the word bag
        X_train, X_val, y_train, y_val = train_test_split(X_bag,y, train_size = 0.7, random_state=42)
        #See what the resulting size of the train test split is
        print(f'X_train shape: {X_train.shape}')

        # Make both classifiers
        lr = LogisticRegression(max_iter = 1000)
        mnb = MultinomialNB()

        # Fit the classifiers
        lr.fit(X_train,y_train)
        mnb.fit(X_train,y_train)

        # Score the classifiers
        lr_score = lr.score(X_val,y_val)
        mnb_score = mnb.score(X_val,y_val)

        print(f'lr model validation accuracy: {lr_score}')
        print(f'mnb model validation accuracy: {mnb_score}')
        
        # Add to the results dataframe
        new = pd.Series({'hyperparameter': f'stemming: {s}',
                         'logreg accuracy': f'{lr_score}', 
                         'mnb accuracy': f'{mnb_score}'})
        results = pd.concat([results, new.to_frame().T], ignore_index = True)



results


# ### Effect of using idf on validation accuracy


for idf in [True, False]:
    X = X_.copy()
    y = y_.copy()
    # Vectorize test data
    tfvectorizer = TfidfVectorizer(min_df = 1e-6, ngram_range = (1,1), stop_words = None, 
                                   lowercase =  True, use_idf= idf, sublinear_tf = True)
    # Train the vectorizer on the training data and transform the training data
    X_bag = tfvectorizer.fit_transform(X)
    print(f'X_bag.shape: {X_bag.shape}')
    # Train test split the word bag
    X_train, X_val, y_train, y_val = train_test_split(X_bag,y, train_size = 0.7, random_state=42)
    #See what the resulting size of the train test split is
    print(f'X_train shape: {X_train.shape}')

    # Make both classifiers
    lr = LogisticRegression(max_iter = 1000)
    mnb = MultinomialNB()

    # Fit the classifiers
    lr.fit(X_train,y_train)
    mnb.fit(X_train,y_train)

    # Score the classifiers
    lr_score = lr.score(X_val,y_val)
    mnb_score = mnb.score(X_val,y_val)

    print(f'lr model validation accuracy: {lr_score}')
    print(f'mnb model validation accuracy: {mnb_score}')

    # Add to the results dataframe
    new = pd.Series({'hyperparameter': f'use_idf: {idf}',
                     'logreg accuracy': f'{lr_score}',
                     'mnb accuracy': f'{mnb_score}'})
    results = pd.concat([results, new.to_frame().T], ignore_index = True)


results


# ### Effect of lowercase on validation accuracy


for case in [True, False]:
    X = X_.copy()
    y = y_.copy()
    # Vectorize test data
    tfvectorizer = TfidfVectorizer(min_df = 1e-6, ngram_range = (1,1), stop_words = None, 
                                   lowercase =  case, use_idf= True, sublinear_tf = True)
    # Train the vectorizer on the training data and transform the training data
    X_bag = tfvectorizer.fit_transform(X)
    print(f'X_bag.shape: {X_bag.shape}')
    # Train test split the word bag
    X_train, X_val, y_train, y_val = train_test_split(X_bag,y, train_size = 0.7, random_state=42)
    #See what the resulting size of the train test split is
    print(f'X_train shape: {X_train.shape}')

    # Make both classifiers
    lr = LogisticRegression(max_iter = 1000)
    mnb = MultinomialNB()

    # Fit the classifiers
    lr.fit(X_train,y_train)
    mnb.fit(X_train,y_train)

    # Score the classifiers
    lr_score = lr.score(X_val,y_val)
    mnb_score = mnb.score(X_val,y_val)

    print(f'lr model validation accuracy: {lr_score}')
    print(f'mnb model validation accuracy: {mnb_score}')

    # Add to the results dataframe
    new = pd.Series({'hyperparameter': f'lowercase: {case}',
                     'logreg accuracy': f'{lr_score}',
                     'mnb accuracy': f'{mnb_score}'})
    results = pd.concat([results, new.to_frame().T], ignore_index = True)


results


# ### Effect of stop words on validation accuracy


for sw in [None, 'english']:
    X = X_.copy()
    y = y_.copy()
    # Vectorize test data
    tfvectorizer = TfidfVectorizer(min_df = 1e-6, ngram_range = (1,1), stop_words = None, 
                                   lowercase =  True, use_idf= True, sublinear_tf = True)
    # Train the vectorizer on the training data and transform the training data
    X_bag = tfvectorizer.fit_transform(X)
    print(f'X_bag.shape: {X_bag.shape}')
    # Train test split the word bag
    X_train, X_val, y_train, y_val = train_test_split(X_bag,y, train_size = 0.7, random_state=42)
    #See what the resulting size of the train test split is
    print(f'X_train shape: {X_train.shape}')

    # Make both classifiers
    lr = LogisticRegression(max_iter = 1000)
    mnb = MultinomialNB()

    # Fit the classifiers
    lr.fit(X_train,y_train)
    mnb.fit(X_train,y_train)

    # Score the classifiers
    lr_score = lr.score(X_val,y_val)
    mnb_score = mnb.score(X_val,y_val)

    print(f'lr model validation accuracy: {lr_score}')
    print(f'mnb model validation accuracy: {mnb_score}')

    # Add to the results dataframe
    new = pd.Series({'hyperparameter': f'stopwords: {sw}',
                     'logreg accuracy': f'{lr_score}',
                     'mnb accuracy': f'{mnb_score}'})
    results = pd.concat([results, new.to_frame().T], ignore_index = True)


results


# ### Effect of min_df on validation accuracy


for mdf in [1e-3,1e-6]:
    X = X_.copy()
    y = y_.copy()
    # Vectorize test data
    tfvectorizer = TfidfVectorizer(min_df = mdf, ngram_range = (1,1), stop_words = None, 
                                   lowercase =  True, use_idf= True, sublinear_tf = True)
    # Train the vectorizer on the training data and transform the training data
    X_bag = tfvectorizer.fit_transform(X)
    print(f'X_bag.shape: {X_bag.shape}')
    # Train test split the word bag
    X_train, X_val, y_train, y_val = train_test_split(X_bag,y, train_size = 0.7, random_state=42)
    #See what the resulting size of the train test split is
    print(f'X_train shape: {X_train.shape}')

    # Make both classifiers
    lr = LogisticRegression(max_iter = 1000)
    mnb = MultinomialNB()

    # Fit the classifiers
    lr.fit(X_train,y_train)
    mnb.fit(X_train,y_train)

    # Score the classifiers
    lr_score = lr.score(X_val,y_val)
    mnb_score = mnb.score(X_val,y_val)

    print(f'lr model validation accuracy: {lr_score}')
    print(f'mnb model validation accuracy: {mnb_score}')

    # Add to the results dataframe
    new = pd.Series({'hyperparameter': f'min_df: {mdf}',
                     'logreg accuracy': f'{lr_score}', 
                     'mnb accuracy': f'{mnb_score}'})
    results = pd.concat([results, new.to_frame().T], ignore_index = True)


results


# ### Effect of ngram_range on validation accuracy

for ngr in [(1,1),(1,5)]:
    X = X_.copy()
    y = y_.copy()
    # Vectorize test data
    tfvectorizer = TfidfVectorizer(min_df = 1e-6, ngram_range = ngr, stop_words = None, 
                                   lowercase =  True, use_idf= True, sublinear_tf = True)
    # Train the vectorizer on the training data and transform the training data
    X_bag = tfvectorizer.fit_transform(X)
    print(f'X_bag.shape: {X_bag.shape}')
    # Train test split the word bag
    X_train, X_val, y_train, y_val = train_test_split(X_bag,y, train_size = 0.7, random_state=42)
    #See what the resulting size of the train test split is
    print(f'X_train shape: {X_train.shape}')

    # Make both classifiers
    lr = LogisticRegression(max_iter = 1000)
    mnb = MultinomialNB()

    # Fit the classifiers
    lr.fit(X_train,y_train)
    mnb.fit(X_train,y_train)

    # Score the classifiers
    lr_score = lr.score(X_val,y_val)
    mnb_score = mnb.score(X_val,y_val)

    print(f'lr model validation accuracy: {lr_score}')
    print(f'mnb model validation accuracy: {mnb_score}')

    # Add to the results dataframe
    new = pd.Series({'hyperparameter': f'ngram_range: {ngr}',
                     'logreg accuracy': f'{lr_score}', 
                     'mnb accuracy': f'{mnb_score}'})
    results = pd.concat([results, new.to_frame().T], ignore_index = True)


print(results)

