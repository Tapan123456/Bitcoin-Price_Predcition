# Bitcoin-Price_Predcition
In this Project we focus on Cryptocurrency named Bitcoin. The predictions of prices are done in real-time based on news and tweets using a LSTM model. Our dataset consists of various features related to bitcoin over 7 years recorded daily.

# Preparing Training Data
The below figure describes the system architecture. TheTraining data for the model consists of past seven year data out of which news is scraped from trusted news sites using
scrapy framework and twitter data using twint library of only those users who have followers more than fifty thousand ,all this data is stored csv in the form of rows and
columns .Next we work on cleaning and pre-processing the data .We use nltk library to remove the stopwords from the data which helps us to get more accurate sentiment score.
In the next step Sentiment Analysis is done using a Vader sentiment library on the processed data which helps in determining the trends by giving us the positive , negative and neutral score. Alongside we have taken the Bitcoin Closing, Opening, High and Low prices of each day from 2013 to 2020. So now we map the sentiment index of a
particular day with above prices of the next day by doing the respective date manipulation. In this way the training data is prepared.

![Preparing Training Data](https://github.com/Tapan123456/Bitcoin-Price_Predcition/blob/master/System%20Architecture/structure.png?raw=true)

# Detection Phase
Now the detection phase begins , in the detection phase real-time tweets are inputted to the model and the model predicts the price for the duration of hours after which we
are running the scheduler .The scheduler runs the twitter scrapper in the crontab on a remote server to get the real time tweets, the twitter scraper is built using the twint library . After getting the tweets ,it is given to the sentiment analysis model and then we take the mean of all the sentiment index of that time period which is fed to the
machine model then finally we get the predicted price for that time period.

![Detection Phase](https://github.com/Tapan123456/Bitcoin-Price_Predcition/blob/master/System%20Architecture/scheduler.png?raw=true)

