# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 19:52:03 2019

@author: viswanathan.a
"""

import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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


# load data
database_filepath= "../data/DisasterResponse.db"
engine = create_engine('sqlite:///'+ database_filepath)
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.debug = True
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()