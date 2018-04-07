import tweepy
import os
import csv
import configparser
import pandas as pd
import quandl
import numpy as np

from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt


# os.getcwd()
# os.chdir(r'D:\CodeRepo\DataSciencePractice\PridictingStockPrices')

# next day-month-year
next_day = np.array([[7,4,2018]])
company_name = 'Google'
quandl_tikker =  'WIKI/GOOGL'

# read credentials from ini file
config = configparser.ConfigParser()
config.read('../credentials.ini')

# Authenticate
consumer_key    = config['Consumer']['API Key']
consumer_secret = config['Consumer']['API Secret']

access_token    = config['Access']['Token']
access_token_secret = config['Access']['Token Secret']

quandl_key = config['Quandl']['API Key']



def get_data_from_quandl(quandl_tikker):

    df = quandl.get(quandl_tikker, api_key=quandl_key)
    #  take only OHLCV
    df = df[['Open','High','Low','Close','Volume']]

    return df


def get_data(df):

    df = df.reset_index()
    dates = df['Date'].tolist()
    prices = df['Close'].tolist()

    return dates, prices



def plot_data(dates, prices):

    plt.plot(dates,prices)
    plt.show()

    return 0


def get_api():

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    return api


def get_company_tweets(api, company_name):

    # searcing company on twitter
    tweets = api.search(company_name)

    return tweets


def get_label(analysis, threshold = 0):

	if analysis.sentiment[0]>threshold:
		return True
	else:
		return False


def get_overall_pos_neg(tweets):
    # return verdict either True means positive or False means Negative
    pos_sent_tweet = 0
    neg_sent_tweet = 0

    for tweet in tweets:
        # tweet = tweets[1]
        # tweet.lang
        analysis = TextBlob(tweet.text)
        if get_label(analysis):
            pos_sent_tweet = pos_sent_tweet+1
        else:
            neg_sent_tweet = neg_sent_tweet+1
    if pos_sent_tweet > neg_sent_tweet:
        verdict = True
        print("Overall Positive")
    else:
        verdict = False
        print("Overall Negative")

    return verdict


def create_datasets(dates, prices):
    train_size=int(0.80*len(dates))
    train_x, train_y = [], []
    test_x, test_y = [], []

    counter = 0
    for date in dates:
        # date = dates[0]
        if counter<train_size:
            # converting timestamp to fretures [day month year]
            train_x.append([date.day, date.month, date.year])
            counter += 1
        else:
            test_x.append(date)

    counter=0
    for price in prices:
        if counter<train_size:
            train_y.append(price)
            counter += 1
        else:
            test_y.append(price)

    return train_x, train_y, test_x, test_y


def train_model(dates, prices):

    train_x, train_y, test_x, test_y = create_datasets(dates, prices)

    train_x = np.reshape(train_x,(len(train_x), 3))
    train_x = np.reshape(train_x,(len(train_x),3))
    train_y = np.reshape(train_y,(len(train_y),1))
    test_x = np.reshape(test_x,(len(test_x),1))
    test_y = np.reshape(test_y,(len(test_y),1))

    # model=Sequential()
    # model.add(Dense(32,input_dim=3,init='uniform',activation='relu'))
    # model.add(Dense(16,init='uniform',activation='relu'))
    # model.add(Dense(1,init='uniform',activation='relu'))
    # model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    # model.fit(train_x,train_y,nb_epoch=10,batch_size=3,verbose=2)

    model=Sequential()
    model.add(Dense(32,input_dim=3,init='uniform',activation='relu'))
    model.add(Dense(32,input_dim=1,init='uniform',activation='relu'))
    model.add(Dense(16,init='uniform',activation='relu'))

    model.add(Dense(1,init='uniform',activation='relu'))
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    model.fit(train_x,train_y,nb_epoch=10,batch_size=3,verbose=1)


    return model


def predict_prices(model, next_day, previous_price):

    # Our prediction for tomorrow
    prediction = model.predict(next_day)

    print('The price will move from {0} to {1} on {2}-{3}-{4}'.format(previous_price, prediction[0][0], next_day[0][0], next_day[0][1], next_day[0][2]))

    return 0


def main():

    df = get_data_from_quandl(quandl_tikker)
    dates, prices = get_data(df)
    plot_data(dates, prices)
    api = get_api()
    tweets = get_company_tweets(api, company_name)
    verdict = get_overall_pos_neg(tweets)

    model = train_model(dates, prices)
    previous_price = df["Close"].iloc[-1]
    predict_prices(model, next_day, previous_price)

    return 0



if __name__ == '__main__':
    main()
