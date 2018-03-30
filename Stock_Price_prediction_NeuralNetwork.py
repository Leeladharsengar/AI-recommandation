# Multilayer Perceptron to Predict stock price
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
import os
import sys
import tweepy
import requests
from textblob import TextBlob
import codecs

# First we login into twitter
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
user = tweepy.API(auth)

# Where the csv file will live
FILE_NAME = 'historical.csv'


def stock_sentiment(quote, num_tweets):
    # Checks if the sentiment for our quote is
    # positive or negative, returns True if
    # majority of valid tweets have positive sentiment
    list_of_tweets = user.search(quote, count=num_tweets)
    positive, null = 0, 0

    for tweet in list_of_tweets:
        blob = TextBlob(tweet.text).sentiment
        #print(codecs.getwriter('utf8')(tweet),blob)
       
        #if blob.subjectivity == 0:
         #   null += 1
         #   next
        #if blob.polarity > 0:
         #   positive += 1
        if blob.polarity > 0:
                sent = 'pos'
        elif blob.polarity == 0:
                sent = 'neut'
        elif blob.polarity < 0 :
                sent = 'negt'

        #output_file = open('Tweet_Analysis.txt','a')
        #output_file.write('{} {} {}\n'.format(tweet, blob, sent))
        #output_file.close()

    #if positive > ((num_tweets - null)/2):
        return True
def get_historical(quote):
    # Download our file from google finance
    url = 'https://finance.google.com/finance/historical?output=csv&q='+quote+'&startdate=Jan+1%2C+2018'
    r = requests.get(url, stream=True)
    if r.status_code != 400:
        with open(FILE_NAME, 'wb') as f:
            for chunk in r:
                f.write(chunk)

        return True

# Ask user for a stock quote
stock_quote = input('Enter a stock quote(e.j: AAPL, FB, GOOGL,YHOO): ').upper()

# Check if the stock sentiment is positve
if not stock_sentiment(stock_quote, num_tweets=10):
    print ('This stock has bad sentiment, please re-run the script')
    #sys.exit()

# Check if we got te historical data
if not get_historical(stock_quote):
    print ('Google returned a 404, please re-run the script and')
    print ('enter a valid stock quote from NASDAQ')
    sys.exit()

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('historical.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape dataset
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(12, input_dim=look_back, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=400, batch_size=2, verbose=2)
# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(dataset,label='Historical Data')
plt.plot(trainPredictPlot,label='Train')
plt.plot(testPredictPlot,label='Test')
plt.legend(loc='best')
plt.show()
