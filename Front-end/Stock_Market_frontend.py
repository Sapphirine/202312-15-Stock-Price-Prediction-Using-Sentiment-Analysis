import streamlit as st
import pandas as pd 
import numpy as np
import yfinance as yf 
from prophet import Prophet
import plotly.express as px 
import plotly.graph_objects as go
from twscrape import API, gather
from twscrape.logger import set_log_level
import asyncio
from datetime import datetime, timedelta
import csv
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
import matplotlib.pyplot as plt
import nltk

st.title("Stock Market Dashboard")
Stock_name = st.sidebar.selectbox(
    'Select which Stock data you want to explore ',
    ('MSFT', 'GOOGL', 'AMZN'))
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

stock_data_reset = yf.download(Stock_name, start = start_date, end = end_date)
data = stock_data_reset.reset_index()
data


# Rename columns as per fbprophet requirements
data.rename(columns={'Date': 'ds', 'Adj Close': 'y'}, inplace=True)
data['y'] = data['y'].fillna(method='ffill')

df_train = data.head(len(data)-60)
df_test  = data.tail(60)

# Create a fbprophet model
prophet_model = Prophet()
prophet_model.fit(df_train)
forecast = prophet_model.predict(df_test.drop(columns="y"))

plot, twt = st.tabs(['Stock Trends', 'Live Twitter Trends'])

def get_sentiment_score(tweets):
    # Create a DataFrame with the tweets
    df = pd.DataFrame(tweets, columns=['tweet'])

    # Text preprocessing using NLTK
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    def preprocess_text(tweet):
        words = word_tokenize(tweet.lower())  # Tokenize and convert to lowercase
        words = [ps.stem(word) for word in words if word.isalnum()]  # Remove non-alphanumeric characters and apply stemming
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        return ' '.join(words)

    # Apply text preprocessing to the tweets
    df['processed_tweet'] = df['tweet'].apply(preprocess_text)

    # Perform sentiment analysis using VaderSentiment
    def analyze_sentiment_vader(tweet):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(tweet)
        # The compound score represents the overall sentiment
        return sentiment_scores['compound']

    # Apply sentiment analysis to the processed tweets
    df['sentiment_score'] = df['processed_tweet'].apply(analyze_sentiment_vader)
    sentiment_score = df['sentiment_score'].values

    return sentiment_score
async def twitter_data(start, end, comp):
    api = API()
    await api.pool.add_account("jake_ryder007", "T3st_twtr", "testtwtr007@gmail.com", "T3st_twtr")
    await api.pool.login_all()

    q = str(Company+" since:"+start+" until:"+end +" -filter:links lang:en -is:retweet -is:reply")

    data_time =[]
    twt_date = []
    twt_cnt = []

    async for tweet in api.search(q, limit = 20):
        data_time.append(datetime.now())
        twt_date.append(tweet.date)
        twt_cnt.append(tweet.rawContent)
    return twt_cnt

with plot:
    st.header("Graph")
    
    # Create a line plot for the entire list1
    trace1 = go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual Data')

    # Create a scatter plot for the last 20 values of list2
    trace2 = go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted', marker=dict(color='red'))

    # Create the layout
    layout = go.Layout(title='Predicted vs Actual Values ', xaxis=dict(title='Index'), yaxis=dict(title='Values'))

    # Create the figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    st.plotly_chart(fig)

with twt:
    Company =  Stock_name         
    since_date = datetime.today().strftime('%Y-%m-%d')
    until_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    tweets = asyncio.run(twitter_data(since_date, until_date, Company))
    senti_score = get_sentiment_score(tweets)

    positive_count = 0
    negative_count = 0
    zero_count = 0
    st.header("Today's Public Perception")
    # Count positive, negative, and zero values
    for score in senti_score:
        if score > 0:
            positive_count += 1
        elif score < 0:
            negative_count += 1
        else:
            zero_count += 1

    key = ["Positive Sentiment %", "Negative Sentiment %",  "Neutral Sentiment %" ]
    sentiment_counts = [int((positive_count/len(senti_score))*100), int((negative_count/len(senti_score))*100), int((zero_count/len(senti_score))*100)]
    colors = ['#144527', '#BC2023', '#F8B324']        
    # Create a Pie chart
    fig = go.Figure(data=[go.Pie(labels=key, values=sentiment_counts, marker=dict(colors=colors))])

    # Show the plot
    st.plotly_chart(fig)

    st.header("Sample tweets")
    for i in range(0,10):
        st.write(tweets[i])