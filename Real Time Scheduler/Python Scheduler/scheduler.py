import twint
import csv
import datetime
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nlppreprocess import NLP
import os

df = pd.read_csv('Filtered1.csv')
users = df.drop_duplicates(subset ="username", keep = False) 
users = list(users['username'])
##print(type(users[0]))
since = str(datetime.datetime.now() - datetime.timedelta(minutes=1370))
since = since[0:19]
until = str(datetime.datetime.now())
until = until[0:19]
# since="2020-06-10 00:00:00"
# until="2020-06-11 18:00:00"

for user in users[0:1500]:
    print(user)
    c = twint.Config()
    c.Username = user
    c.Search = "bitcoin"
    c.Lang = "en"
    c.Pandas = True

    c.Since = since
    c.Until = until
    twint.run.Search(c)
    Tweets_df = twint.storage.panda.Tweets_df
    print(Tweets_df)
    Tweets_df.to_csv('raw_scheduler2.csv', mode='a', header=False)

#Sentiment Analysis Runtime

pos = []
neg = []
neu = []
compound = []
sid_obj = SentimentIntensityAnalyzer()
Tweets_df = pd.read_csv('raw_scheduler2.csv')

nlp = NLP()
Tweets_df[Tweets_df.columns[7]] = Tweets_df[Tweets_df.columns[7]].apply(nlp.process)

for ind in Tweets_df.index:
    # date.append(data['date'])
    # username.append(data['username'])
    snt = sid_obj.polarity_scores(Tweets_df[Tweets_df.columns[7]][ind])
    pos.append(snt['pos'])
    neg.append(snt['neg'])
    neu.append(snt['neu'])
    compound.append(snt['compound'])
    date = Tweets_df[Tweets_df.columns[4]][ind]
    Tweets_df[Tweets_df.columns[4]][ind] = date[0:10]

list_of_tuples = list(zip(Tweets_df[Tweets_df.columns[4]], pos, neg, neu, compound))
final_data = pd.DataFrame(list_of_tuples, columns=[
                          'Date', 'Positive', 'Negative', 'Neutral', 'Compound'])
#final_data.to_csv("Prediction_input_without_mean2.csv",mode='a', header=False)
final_data = final_data.groupby(['Date'], as_index=False).mean()

import os
os.remove("raw_scheduler2.csv")

from exchanges.bitfinex import Bitfinex
price = Bitfinex().get_current_price()
final_data['Closing_Price']=price
final_data=final_data.tail(1)
print(final_data)
final_data.to_csv("prediction_input11.csv", mode='a', header=False)

os.system('python prediction.py')

