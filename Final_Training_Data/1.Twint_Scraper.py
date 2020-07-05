import twint
import csv
import datetime
# Configure the scheduler 
c = twint.Config()
c.Search = "bitcoin"
c.Min_likes = 100
c.Lang = "en"
c.Pandas = True
# c.Store_csv = True
# c.Output = "twint4.csv"

since = str(datetime.datetime.now() - datetime.timedelta(minutes=100))
since = since[0:19]
until = str(datetime.datetime.now())
until = until[0:19]
c.Since = since
c.Until = until

# Run
twint.run.Search(c)

#Storing in pandas dataframe
Tweets_df = twint.storage.panda.Tweets_df

#mode a for appending . if you run a Scheduler
Tweets_df.to_csv('my_csv.csv', mode='a', header=False)