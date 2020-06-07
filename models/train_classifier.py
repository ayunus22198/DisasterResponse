import sys
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords 
import nltk
import re
import string
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle 

wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()
stopwords = set(stopwords.words('english')) 

'''
Descritption: Load data from db.
'''
def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("Disasters",con=engine)
    length = df['message'].shape[0]
    X = df['message'][0:(int) (length)]
    y = df[df.columns[4:]][0:(int) (length)]
    return X, y, y.columns
'''
Description: Set of functions for tokenizing and lemmatizing text
'''
def lemmatization(txt):
    text = [wn.lemmatize(word) for word in txt]
    return text

def clean_text(txt):
    txt = "".join([c for c in txt if c not in string.punctuation])
    tokens = re.split('\W+', txt)
    txt = [word for word in tokens if word not in stopwords]
    return lemmatization(txt)

def tokenize(text):
    return clean_text(text)

'''
Description: Build model with pipeline
'''
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    pass

'''
Description: Saves model
'''
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35)
        
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