# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 19:22:09 2019

@author: viswanathan.a
"""
# import libraries
import sys
import numpy as np
import pandas as pd
import pickle
import re
import nltk
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier


def load_data(database_filepath):
    '''
    Function to load the input data
    input:
        database_filepath: File path where sql database was saved.
    output:
        X: Training message List.
        Y: Training target.
        category_names: Categorical name for labeling.
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.loc[:, 'related':'direct_report']
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    """
    Tokenization involves 4 main steps:
    Replacing non-numeric and non-alphabets
    Tokenzation of words
    Lemmatization of tokenized words to its root form
    Stripping of white spaceand finally returns an array of stemmed tokens
    """
    # replace all non-alphabets / non-numbers with blank space
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize words
    tokens = word_tokenize(text)
    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # instantiate stemmer
    stemmer = PorterStemmer()
    clean_tokens = []
    for tok in tokens:
        # lemmatize token using noun as part of speech
        clean_tok = lemmatizer.lemmatize(tok)
        # lemmatize token using verb as part of speech
        clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')
        # stem token
        clean_tok = stemmer.stem(clean_tok)
        # strip whitespace and append clean token to array
        clean_tokens.append(clean_tok.strip())
    return clean_tokens


def build_model():
    '''
    input:
        None
    output:
        cv: GridSearch model result.
    '''
    pipeline = Pipeline([('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                     ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))])
    parameters = {
    #'tfidf__max_df': (0.9, 1.0),
    #'tfidf__min_df': (0.01, 1),
    'tfidf__ngram_range': ((1, 1),(1,2))
    #'tfidf__stop_words': (None, 'english'),
    #'clf__estimator__learning_rate': (0.1,1.0),
    #'clf__estimator__n_estimators': (50, 100)
    }

    #cv = GridSearchCV(pipeline, param_grid=parameters, scoring='precision_samples', cv = 5)
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the built model
    Input: the trained model, test part of X, test part of Y, names of column in Y
    Outpur: classification report and accuracy score
    """
    Y_pred = model.predict(X_test)
    pred_df = pd.DataFrame(Y_pred, columns = Y_test.columns)
    for column in Y_test.columns:
        print('------------------------------------------------------\n')
        print('Accuracy: ', accuracy_score(Y_test[column], pred_df[column]))
        print('Feature: {}\n'.format(column))
        print(classification_report(Y_test[column],pred_df[column]))
    
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()
        print('Training model...')
        model.fit(X_train, Y_train)
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()