import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath[:-3], engine)
    
    # get X, Y values and category names
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns
    
    return X, Y, category_names
    

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Normalization: Remove punctuation characters and lowercase
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    text_ = re.sub(r"[^a-zA-Z0-9]", " ", text).lower().strip()
    
    # Tokenize
    word_list = word_tokenize(text_)
    
    # Remove stop words
    word_list = [w for w in word_list if w not in stopwords.words('english')]
    
    # Lemmatization
    tokens = [WordNetLemmatizer().lemmatize(w) for w in word_list]
    
    return tokens


def build_model():
    dtc = DecisionTreeClassifier(random_state=0)

    pipeline_dtc = Pipeline([
                       ('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', MultiOutputClassifier(dtc)),
                   ])
    
    parameters_dtc = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__splitter': ['best', 'random']
    }

    cv_dtc = GridSearchCV(pipeline_dtc, param_grid=parameters_dtc)
    return cv_dtc


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = cv_dtc.predict(X_test)
    
    accuracy = (y_true == y_pred).mean()

    print('Accuracy: {}'.format(accuracy))
    for i in range(y_pred.shape[1]):
        print('Class Label: {}'.format(names[i]))
        print(classification_report(y_true[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


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