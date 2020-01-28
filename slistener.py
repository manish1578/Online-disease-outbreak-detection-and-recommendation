#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 21:46:00 2019

@author: manishsharma
"""

# import packages
from tweepy.streaming import StreamListener
import json
import time
import sys
import pandas as pd
from sqlalchemy import create_engine
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
geolocator = Nominatim(user_agent='best_manish')

# inherit from StreamListener class
class SListener(StreamListener):

    # initialize the API and a counter for the number of tweets collected
    def __init__(self, api = None, fprefix = 'streamer'):
        self.api = api or API()
        self.cnt = 0
        # create a engine to the database
        # switch to the following definition if run this code locally
        self.engine = create_engine('sqlite:///tweets.sqlite')

    # for each tweet streamed
    def on_status(self, status):
        
        # increment the counter
        self.cnt += 1

        # parse the status object into JSON
        status_json = json.dumps(status._json)
        # convert the JSON string into dictionary
        status_data = json.loads(status_json)

        # initialize a list of potential full-text
        full_text_list = [status_data['text']]

        # add full-text field from all sources into the list
        if 'extended_tweet' in status_data:
            full_text_list.append(status_data['extended_tweet']['full_text'])
        if 'retweeted_status' in status_data and 'extended_tweet' in status_data['retweeted_status']:
            full_text_list.append(status_data['retweeted_status']['extended_tweet']['full_text'])
        if 'quoted_status' in status_data and 'extended_tweet' in status_data['quoted_status']:
            full_text_list.append(status_data['quoted_status']['extended_tweet']['full_text'])

        # only retain the longest candidate
        full_text = max(full_text_list, key=len)
        
        my_address = format(status_data['user'].get('location'))
        #location = geolocator.geocode(my_address,timeout=10)
        #print(my_address)
    
        try:
            location = geolocator.geocode(my_address, timeout=10)
            #print(location.raw)
            #print(location.latitude, location.longitude)
        except GeocoderTimedOut as e:
            print("Error: geocode failed on input %s with message %s"%(my_address, e.message))
        
        #location = geolocator.geocode(status_data['user'].get('location', {}))

        # extract time and user info
        tweets = {
            'created_at': status_data['created_at'],
            'text':  full_text,
            'location': my_address,
            'LAT': location.latitude,
            'LON': location.longitude,
            'user': status_data['user']['description'],
        }

        # uncomment the following to display tweets in the console
        print("Writing tweets # {} to the database".format(self.cnt))
        print("Tweet Created at: {}".format(tweets['created_at']))
        print("Tweets Content:{}".format(tweets['text']))
        print("Tweets Location:{}".format(status_data['user'].get('location', {})))
        print()

        # convert into dataframe
        df = pd.DataFrame(tweets, index=[0])
        
        # convert string of time into date time obejct
        df['created_at'] = pd.to_datetime(df.created_at)
        #df['user'] = pd.to_string(df.user)
        # push tweet into database
        df.to_sql('tweet', con=self.engine, if_exists='append')

        with self.engine.connect() as con:
            con.execute("""
                        DELETE FROM tweet
                        WHERE created_at in(
                            SELECT created_at
                                FROM(
                                    SELECT created_at, strftime('%s','now') - strftime('%s',created_at) AS time_passed
                                    From tweet
                                    WHERE time_passed >= 6000))""")
        