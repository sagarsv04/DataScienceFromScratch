import tweepy
import os
from textblob import TextBlob
import configparser
import numpy as np
import operator

# os.getcwd()
# os.chdir(r'D:\CodeRepo\DataSciencePractice\TwitterSentiment')

# hashtag related to the debate
name_of_debate = 'AI'

# list of candidates on debate
candidates_names = ['Elon Musk']

# date of the debate
since_date = "2018-04-04"
until_date = "2017-04-06"

# read credentials from ini file
config = configparser.ConfigParser()
config.read('../credentials.ini')

# Authenticate
consumer_key    = config['Consumer']['API Key']
consumer_secret = config['Consumer']['API Secret']

access_token    = config['Access']['Token']
access_token_secret = config['Access']['Token Secret']



def get_api():

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    return api


def get_search_results(api, keyword, since_date, until_date, tweet_count=100):

    if type(keyword)==list:
        debate, candidate = keyword
        results = api.search(q=[debate, candidate], count=tweet_count, since = since_date, until=until_date)
    else:
        results = api.search(keyword, count=tweet_count, since = since_date, until=until_date)

    return results


# Function of labelisation of analysis
def get_label(analysis, threshold = 0):
	if analysis.sentiment[0]>threshold:
		return 'Positive'
	else:
		return 'Negative'


def run_twitter_sentiment(api):

    # Retrieve Tweets and Save Them
    all_polarities = dict()
    for candidate in candidates_names:
    	this_candidate_polarities = []
    	# Get the tweets about the debate and the candidate between the dates
    	this_candidate_tweets = get_search_results(api, [name_of_debate, candidate], since_date, until_date, 50)
    	# Save the tweets in csv
    	with open('./%s_tweets.csv' % candidate, 'w') as this_candidate_file:
    		this_candidate_file.write('tweet,sentiment_label\n')
    		for tweet in this_candidate_tweets:
    			analysis = TextBlob(tweet.text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
    			#Get the label corresponding to the sentiment analysis
    			this_candidate_polarities.append(analysis.sentiment[0])
    			this_candidate_file.write('%s,%s\n' % (tweet.text.encode('utf8'), get_label(analysis)))
    	#Save the mean for final results
    	all_polarities[candidate] = np.mean(this_candidate_polarities)

    #Step bonus - Print a Result
    sorted_analysis = sorted(all_polarities.items(), key=operator.itemgetter(1), reverse=True)
    print('Mean Sentiment Polarity in descending order :')
    for candidate, polarity in sorted_analysis:
    	print('%s : %0.3f' % (candidate, polarity))


    return 0


def main():
    api = get_api()
    run_twitter_sentiment(api)

    return 0



if __name__ == '__main__':
    main()
