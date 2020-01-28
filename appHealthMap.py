#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 21:46:00 2019

@author: manishsharma
"""

#Code 2 to Run

# import packages

#get_tweet_data()

import os
import datetime
import re
import nltk
import numpy as np
#import sys
#sys.path.append('../')
import healthmap_add
from collections import deque, Counter

from api import get_tweet_data



import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State

import plotly
import plotly.graph_objs as go

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer



# download nltk dependencies
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# global refresh interval for the application, ms
GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 2000)

# initialize a sentiment analyzer
sid = SentimentIntensityAnalyzer()

#searchTerm=input("Enter the keyword/Hashtag to disease search about :")
#locationTerms=input("Enter the loaction of tweets to analyze :")

#keywords_to_hear = ['#dengue', '#zika','#malaria','#chikungunya','dengue', 'zika','malaria','chikungunya', ]
keywords_to_hear = ['#ARMYStayTuned',]
#keywords_to_hear = ['#HindusRejectHindutva',]
keywords_to_disease =['dengue', 'zika','malaria','chikungunya',]
keywords_to_location =['Delhi','Mumbai','Kolkata','Chennie',]
# stop words for the word-counts
stops = stopwords.words('english')
stops.append('https')
# for keyword in keywords_to_hear:
#     stops.append(keyword)

# initialize the app and server
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

# global color setting
app_color = {
    "graph_bg": "rgb(221, 236, 255)",
    "graph_line": "rgb(8, 70, 151)",
    "graph_font":"rgb(2, 29, 65)"
}

# colors for plots
chart_colors = [
    '#664DFF',
    '#893BFF',
    '#3CC5E8',
    '#2C93E8',
    '#0BEBDD',
    '#0073FF',
    '#00BDFF',
    '#A5E82C',
    '#FFBD42',
    '#FFCA30'
]

# the number of most frequently mentioned tags
num_tags_scatter = 5

# initalize a dictionary to store the number of tweets for each game
scatter_dict = {}

sentiment_dict = {}

# initialize x and y coordinates for scatter plot
# use duque here to store the changing trend of number of tweets
# X is the x-axis with time stamps
X_universal = deque(maxlen=30)
df = get_tweet_data()
# add layout to the app
app.layout = html.Div(
    [
        
     
    # header
        html.Div(
            [   
                html.Div(
                    [
                        html.A(
                            html.H4(
                                "Mosquito-borne disease TWITTER STREAMING",
                            ),
                            href='https://geu.ac.in',
                            target='_blank',
                            className="app__header__title"
                        ),
                        html.P(
                            "This app streams tweets about Mosquito-borne disease in real time and displays live charts for Mosquito-borne disease tracking and sentiment analysis.",
                            className="app__header__title--grey",
                        ),
                    ],
                    className="app__header__desc",
                ),
                # logo
                html.Div(
                    [   
                        html.A(
                            html.Img(
                            src=app.get_asset_url("geu_logo.png"),
                            className="app__menu__img"
                            ),
                            href='https://geu.ac.in',
                            target='_blank'
                        )
                        
                    ],
                    className="app__header__logo",
                ),
            ],
            className="app__header",
        ),
         html.H3('Mosquito-borne disease'),
          dcc.Dropdown(
              id='app-1-dropdown',
              options=[
                  {'label': '{}'.format(i), 'value': i} for i in keywords_to_disease
        ]
        ),
        html.Div(id='app-1-display-value'),
         html.H3('Location'),
          dcc.Dropdown(
              id='app-2-dropdown',
              options=[
                  {'label': '{}'.format(i), 'value': i} for i in keywords_to_location
        ]
        ),
        html.Div(id='app-2-display-value'),
        html.Div([
        dcc.Input(id='username', value='Initial Value', type='text'),
        html.Button(id='submit-button', type='submit', children='Submit'),
        html.Div(id='output_div')
    ]),

   

        
         html.Div([
                dcc.Interval(
                            id="query_update_map",
                            interval=int(GRAPH_INTERVAL),
                            n_intervals=0,
                        ),
                    html.H1('Live tweets location'),
                    html.Div(id='text-content_map'),
                dcc.Graph(id='map', animate=True, figure={
                    'data': [{
                            'lat': df['LAT'],
                            'lon': df['LON'],
                            'text':df['text'],
                    'marker': {
                            'color': df['created_at'],
                            'size': 8,
                            'opacity': 0.6
                            },
                    'customdata': df['LAT'],
                    'type': 'scattermapbox'
                    }],
                    'layout': {
                        'mapbox': {
                            'accesstoken': 'pk.eyJ1IjoiY2hyaWRkeXAiLCJhIjoiY2ozcGI1MTZ3MDBpcTJ3cXR4b3owdDQwaCJ9.8jpMunbKjdq1anXwU5gxIw'
                },
            'hovermode': 'closest',
            'height': 500,
            'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0}
            }
            }),
                dcc.Interval(
                        id='graph-update',
                        interval=1*1000,
                        n_intervals=0,
                ),
            ]),
        html.Div(
            [
                # left hand side, tweets count scatter plot
                html.Div(
                    [   
                        dcc.Interval(
                            id="query_update",
                            interval=int(GRAPH_INTERVAL),
                            n_intervals=0,
                        ),
                        html.Div(
                            [html.H6("WORD-COUNT TREND", className="graph__title")]
                        ),
                        html.Div(
                                    [
                                        html.P(
                                            "Total number of tweets streamed during last 60 seconds: 0",
                                            id="bin-size",
                                            className="auto__p",
                                        ),
                                    ],
                                    className="auto__container",
                                ),
                        dcc.Graph(
                            id="number_of_tweets",
                            animate=False,
                            figure=go.Figure(
                                layout=go.Layout(
                                    plot_bgcolor=app_color["graph_bg"],
                                    paper_bgcolor=app_color["graph_bg"],
                                )
                            ),
                        ),
                    ],
                    className="two-thirds column number_of_tweets",
                ),
                # right hand side, bar plot and pie chart
                html.Div(
                    [
                        # bar chart
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "WORD COUNT",
                                            className="graph__title",
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id="word_counts",
                                    animate=False,
                                    figure=go.Figure(
                                        layout=go.Layout(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                ),
                            ],
                            className="graph__container first",
                        ),
                        # sentiment plot
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "SENTIMENT SCORE", className="graph__title"
                                        )
                                    ]
                                ),
                                dcc.Graph(
                                    id="sentiment_scores",
                                    figure=go.Figure(
                                        layout=go.Layout(
                                            plot_bgcolor=app_color["graph_bg"],
                                            paper_bgcolor=app_color["graph_bg"],
                                        )
                                    ),
                                ),
                            ],
                            className="graph__container second",
                        ),
                    ],
                    className="one-third column bar_pie",
                ),
            ],
            className="app__content",
        ),
        
       
        
       

        # footer
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "Mosquito-borne disease",
                            className="app__comment",
                        ),
                    ]
                    
                )
            ]
        )
    ],
    className="app__container",
)
@app.callback(Output('map', 'figure'),
              [Input('graph-update', 'interval')])
def update_graph_map(interval):

    return {
                    'data': [{
                            'lat': df['LAT'],
                            'lon': df['LON'],
                            'text':df['text'],
                    'marker': {
                            'color': df['created_at'],
                            'size': 8,
                            'opacity': 0.6
                            },
                    'customdata': df['LAT'],
                    'type': 'scattermapbox'
                    }],
                    'layout': {
                        'mapbox': {
                            'accesstoken': 'pk.eyJ1IjoiY2hyaWRkeXAiLCJhIjoiY2ozcGI1MTZ3MDBpcTJ3cXR4b3owdDQwaCJ9.8jpMunbKjdq1anXwU5gxIw'
                },
            'hovermode': 'closest',
            'height': 500,
            'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0}
            }
            }

@app.callback(Output('output_div', 'children'),
                  [Input('submit-button', 'n_clicks')],
                  [State('username', 'value')],
                  )
def update_output(clicks, input_value):
    if clicks is not None:
        return 'You have selected "{}"'.format(input_value)
        #print(clicks, input_value)
       
@app.callback(
     dash.dependencies.Output("text-content_map", "children"),
    [ dash.dependencies.Input("query_update_map", "n_intervals")],
)
def show_num_bins(slider_value):
    """ Display the number of bins. """

    df = get_tweet_data()
    total_tweets = len(df)

    return "Total number of tweets streamed during last 60 seconds: " + str(int(total_tweets))

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
@app.callback(
    Output('app-1-display-value', 'children'),
    [Input('app-1-dropdown', 'value')])
def display_value_disease(value):
    
    return 'You have selected "{}"'.format(value)

@app.callback(
    Output('app-2-display-value', 'children'),
    [Input('app-2-dropdown', 'value')])
def display_value_location(value):
    return 'You have selected "{}"'.format(value)

def hashtag_counter(series):
    """
    count the number of tweets for all the keywords

    Parameters
    ----------
        seriers: pandas Series
            the text column that contains the text of the tweets
    
    Returns
    -------
        cnt: dictionary
            a dictionary with keyword: number of tweets
    """

    cnt = {keyword: 0 for keyword in keywords_to_hear}
    for row in series:
        for keyword in keywords_to_hear:    
            if keyword.lower() in row.lower():
                cnt[keyword] += 1
    return cnt


def bag_of_words(series):
    """
    count the words in all the tweets

    Parameters
    ----------
        seriers: pandas Series
            the text column that contains the text of the tweets

    Returns
    -------
        collections.Counter object
            a dictionary with all the tokens and their number of apperances
    """
    
    # merge the text from all the tweets into one document
    document = ' '.join([row for row in series])

    # lowercasing, tokenization, and keep only alphabetical tokens
    tokens = [word for word in word_tokenize(document.lower()) if word.isalpha()]

    # filtering out tokens that are not all alphabetical
    tokens = [word for word in re.findall(r'[A-Za-z]+', ' '.join(tokens))]

    # remove all stopwords
    no_stop = [word for word in tokens if word not in stops]

    return Counter(no_stop)

def preprocess_nltk(row):
    """
    preprocessing the user description for user tagging

    Parameters
    ----------
        row: string
            a single record of a user's profile description
    
    Returns
    -------
        string
            a clean string
    """

    # lowercasing, tokenization, and keep only alphabetical tokens
    tokens = [word for word in word_tokenize(row.lower()) if word.isalpha()]

    # filtering out tokens that are not all alphabetical
    tokens = [word for word in re.findall(r'[A-Za-z]+', ' '.join(tokens))]
     
    #tokens = [word for word in re.findall('RT @[\w_]+: ', ' '.join(tokens))]
    
    # remove all stopwords
    no_stop = [word for word in tokens if word not in stops]

    return ' '.join(no_stop)


# define callback function for number_of_tweets scatter plot
@app.callback(
    Output('number_of_tweets', 'figure'),
    [Input('query_update', 'n_intervals')])
def update_graph_scatter(n):

    # query tweets from the database
    df = get_tweet_data()

    # get the number of tweets for each keyword
    cnt = bag_of_words(df['text'])

    # get the current time for x-axis
    time = datetime.datetime.now().strftime('%D, %H:%M:%S')
    X_universal.append(time)

    to_pop = []
    for keyword, cnt_queue in scatter_dict.items():
        if cnt_queue:
            while cnt_queue and (cnt_queue[0][1] < X_universal[0]):
                cnt_queue.popleft()
        else:
            to_pop.append(keyword)
        

    for keyword in to_pop:
        scatter_dict.pop(keyword)

    top_N = cnt.most_common(num_tags_scatter)

    for keyword, cnt in top_N:
        if keyword not in scatter_dict:
            scatter_dict[keyword] = deque(maxlen=30)
            scatter_dict[keyword].append([cnt, time])
        else:
            scatter_dict[keyword].append([cnt, time])

    new_colors = chart_colors[:len(scatter_dict)]

    # plot the scatter plot
    data=[go.Scatter(
        x=[time for cnt, time in cnt_queue],
        y=[cnt for cnt, time in cnt_queue],
        name=keyword,
        mode='lines+markers',
        opacity=0.5,
        marker=dict(
            size=10,
            color=color,
        ),
        line=dict(
            width=6,
            # dash='dash',
            color=color,
        )
    ) for color, (keyword, cnt_queue) in list(zip(new_colors, scatter_dict.items()))]

    # specify the layout
    layout = go.Layout(
            xaxis={
                'automargin': False,
                'range': [min(X_universal), max(X_universal)],
                'title': 'Current Time (GMT)',
                'nticks': 6
            },
            yaxis={
                'type': 'log',
                'autorange': True,
                'title': 'Number of Tweets'
            },
            height=700,
            plot_bgcolor=app_color["graph_bg"],
            paper_bgcolor=app_color["graph_bg"],
            font={"color": app_color["graph_font"]},
            autosize=False,
            legend={
                'orientation': 'h',
                'xanchor': 'center',
                'yanchor': 'top',
                'x': 0.5,
                'y': 1.025
            },
            margin=go.layout.Margin(
                l=75,
                r=25,
                b=45,
                t=25,
                pad=4
            ),
        )

    return go.Figure(
        data=data,
        layout=layout,
    )

# define callback function for word-counts
@app.callback(
    Output('word_counts', 'figure'),
    [Input('query_update', 'n_intervals')])
def update_graph_bar(interval):

    # query tweets from the database
    df = get_tweet_data()

    # get the counter for all the tokens
    word_counter = bag_of_words(df.text)

    # get the most common n tokens
    # n is specified by the slider
    top_n = word_counter.most_common(10)[::-1]

    # get the x and y values
    X = [cnt for word, cnt in top_n]
    Y = [word for word, cnt in top_n]

    # plot the bar chart
    bar_chart = go.Bar(
        x=X, y=Y,
        name='Word Counts',
        orientation='h',
        marker=dict(color=chart_colors[::-1])
    )

    # specify the layout
    layout = go.Layout(
            xaxis={
                'type': 'log',
                'autorange': True,
                'title': 'Number of Words'
            },
            height=300,
            plot_bgcolor=app_color["graph_bg"],
            paper_bgcolor=app_color["graph_bg"],
            font={"color": app_color["graph_font"]},
            autosize=True,
            margin=go.layout.Margin(
                l=100,
                r=25,
                b=75,
                t=25,
                pad=4
            ),
        )

    return go.Figure(
        data=[bar_chart], layout=layout
    )

# define callback function for user_group
@app.callback(
    Output('sentiment_scores', 'figure'),
    [Input('query_update', 'n_intervals')])
def update_graph_sentiment(interval):

    # query tweets from the database
    df = get_tweet_data()
    # get the number of tweets for each keyword
    cnt = bag_of_words(df['text'])

    # get top-N words
    top_N = cnt.most_common(num_tags_scatter)
    top_N_words = [keyword for keyword, cnt in top_N]


    # preprocess the text column
    df['text'] = df.text.apply(preprocess_nltk)

    sentiments = {keyword:[] for keyword in top_N_words}
    for row in df['text']:
        # print(row)
        for keyword in top_N_words:
            # print(keyword)
            if keyword.lower() in row.lower():
                # print(sid.polarity_scores(row)['compound'])
                sentiments[keyword].append(sid.polarity_scores(row)['compound'])
    
    # print(sentiments)
    
    avg_sentiments = {}
    for keyword, score_list in sentiments.items():
        avg_sentiments[keyword] = [np.mean(score_list), np.std(score_list)]
    
    # get the current time for x-axis
    time = datetime.datetime.now().strftime('%D, %H:%M:%S')
    X_universal.append(time)

    to_pop = []
    for keyword, score_queue in sentiment_dict.items():
        if score_queue:
            while score_queue and (score_queue[0][1] <= X_universal[0]):
                score_queue.popleft()
        else:
            to_pop.append(keyword)
        

    for keyword in to_pop:
        sentiment_dict.pop(keyword)

    for keyword, score in avg_sentiments.items():
        if keyword not in sentiment_dict:
            sentiment_dict[keyword] = deque(maxlen=30)
            sentiment_dict[keyword].append([score, time])
        else:
            sentiment_dict[keyword].append([score, time])

    new_colors = chart_colors[:len(sentiment_dict)]

    # plot the scatter plot
    data=[go.Scatter(
        x=[time for score, time in score_queue],
        y=[score[0] for score, time in score_queue],
        error_y={
            "type": "data",
            "array": [score[1]/30 for score, time in score_queue],
            "thickness": 1.5,
            "width": 1,
            "color": "#000",
        },
        name=keyword,
        mode='markers',
        opacity=0.7,
        marker=dict(color=color)
    ) for color, (keyword, score_queue) in list(zip(new_colors, sentiment_dict.items()))]

    # specify the layout
    layout = go.Layout(
            xaxis={
                'automargin': False,
                'range': [min(X_universal), max(X_universal)],
                'title': 'Current Time (GMT)',
                'nticks': 2,
            },
            yaxis={
                'autorange': True,
                'title': 'Sentiment Score'
            },
            height=400,
            plot_bgcolor=app_color["graph_bg"],
            paper_bgcolor=app_color["graph_bg"],
            font={"color": app_color["graph_font"]},
            autosize=False,
            legend={
                'orientation': 'v',
                # 'xanchor': 'right',
                # 'yanchor': 'middle',
                # 'x': 0.5,
                # 'y': 1.025
            },
            margin=go.layout.Margin(
                l=75,
                r=25,
                b=70,
                t=25,
                pad=4
            ),
        )

    return go.Figure(
        data=data,
        layout=layout,
    )


# define callback functions for the indicator of the slider
@app.callback(
    Output("bin-size", "children"),
    [Input("query_update", "n_intervals")],
)
def show_num_bins(slider_value):
    """ Display the number of bins. """

    df = get_tweet_data()
    total_tweets = len(df)

    return "Total number of tweets streamed during last 60 seconds: " + str(int(total_tweets))

# run the app
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
if __name__ == '__main__':
    #import streaming
    #app.run_server(debug=False, host='0.0.0.0', port=80)
    app.run_server(debug=True)
    
    
    
    