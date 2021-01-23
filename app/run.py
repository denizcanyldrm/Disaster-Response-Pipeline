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
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
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
    
    category_prob = list(df.iloc[:, 4:].mean(axis=0))  
    category_names = df.iloc[:, 4:].columns
    
    top_10_by_direct = df[df['genre'] == 'direct'].iloc[:, 4:].sum().sort_values(ascending=False)[:10]
    top_10_by_direct = top_10_by_direct.reset_index()
    top_10_by_direct.columns = ['direct', 'count']
    
    top_10_by_news = df[df['genre'] == 'news'].iloc[:, 4:].sum().sort_values(ascending=False)[:10]
    top_10_by_news = top_10_by_news.reset_index()
    top_10_by_news.columns = ['news', 'count']
    
    top_10_by_social = df[df['genre'] == 'social'].iloc[:, 4:].sum().sort_values(ascending=False)[:10]
    top_10_by_social = top_10_by_social.reset_index()
    top_10_by_social.columns = ['social', 'count']
    
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
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_prob
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "probability"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x = top_10_by_direct['direct'],
                    y = top_10_by_direct['count']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories Genre=direct',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Top 10 Category Names"
                }
            }
        },
        {
            'data': [
                Bar(
                    x = top_10_by_news['news'],
                    y = top_10_by_news['count']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories Genre=news',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Top 10 Category Names"
                }
            }
        },
        {
            'data': [
                Bar(
                    x = top_10_by_social['social'],
                    y = top_10_by_social['count']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories Genre=social',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Top 10 Category Names"
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()