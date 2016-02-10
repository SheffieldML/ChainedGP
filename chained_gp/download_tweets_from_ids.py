import os
import tweepy
import pandas as pd
import time

import progressbar as pb

# Get private keys from config file
try:
    import ConfigParser
    config = ConfigParser.ConfigParser()
except ImportError:
    import configparser
    config = configparser.ConfigParser()

config.readfp(open("./tweet_config.cfg"))
CONSUMER_KEY = config.get('twitter', 'CONSUMER_KEY')
CONSUMER_SECRET = config.get('twitter', 'CONSUMER_SECRET')

OAUTH_TOKEN = config.get('twitter', 'OAUTH_TOKEN')
OAUTH_TOKEN_SECRET = config.get('twitter', 'OAUTH_TOKEN_SECRET')

# Authenticate
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

# Make tweepy api object, and be carefuly not to abuse the API!
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

requests_per_minute = int(180.0/15.0)


for party in ['labour', 'conservative']:
    print("Downloading {} party data".format(party))
    # Load in the twitter data we want to join
    data = pd.read_csv('./data_download/{}_raw_ids.csv'.format(party))

    #Iterate in blocks
    full_block_size = 100
    num_blocks = 6 #data.shape[0]/full_block_size + 1
    last_block_size = data.shape[0]%full_block_size

    # Progress bar to give some indication of how long we now need to wait!
    pbar = pb.ProgressBar(widgets=[
            ' [', pb.Timer(), '] ',
            pb.Bar(),
            ' (', pb.ETA(), ') ',])

    for block_num in pbar(range(num_blocks)):
        # Get a single block of tweets
        start_ind = block_num*full_block_size
        if block_num == num_blocks - 1:
            end_ind = start_ind + last_block_size 
        else:
            end_ind = start_ind + full_block_size
        tweet_block = data.iloc[start_ind:end_ind]

        # print("Getting block {b}, now {percent:.2%} done".format(b=block_num, percent=1.0*block_num/num_blocks))
        # Gather ther actual data, fill out the missing time
        tweet_block_ids = tweet_block['id_str'].tolist()
        tweet_block_results = api.statuses_lookup(tweet_block_ids, trim_user=True)
        for tweet in tweet_block_results:
            data.ix[data['id_str'] == int(tweet.id_str), 'time'] = tweet.created_at
        
        # Wait so as to stay below the rate limit
        # Stay on the safe side, presume that collection is instantanious
        time.sleep(60.0/requests_per_minute + 0.1)

        if block_num % 5 == 0:
            # print("saving")
            data.to_csv('./data_download/labour_parsed.csv')
        
    #Now convert times to pandas datetimes
    data['time'] = pd.to_datetime(data['time'])
