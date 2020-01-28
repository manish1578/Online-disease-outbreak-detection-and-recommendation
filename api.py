import sqlite3
import pandas as pd
import pathlib

DB_FILE = 'tweets.sqlite'

def get_tweet_data():
    con = sqlite3.connect(DB_FILE)
    statement = 'SELECT * FROM tweet'
    df = pd.read_sql_query(statement, con)
    return df


