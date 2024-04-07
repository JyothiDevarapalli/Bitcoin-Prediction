
import numpy as np
import pandas as pd

pip install mwclient

import mwclient
import time


site = mwclient.Site("en.wikipedia.org")


page = site.pages["Bitcoin"]

revs = list(page.revisions())

revs[0]

#We will use the date field to do this sorting. The dictionary key that contains the date is "timestamp".
revs = sorted(revs, key = lambda rev: rev["timestamp"])

pip install transformers

from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")


def find_sentiment(text):
    #set the maximum text size to 250 characters
    sent = sentiment_pipeline([text[:250]])[0]
    score = sent["score"]
    if sent["label"] == "Negative":
        score *= -1
    return score


edits = {}

#Build a loop for
for rev in revs:
    date = time.strftime("%Y-%m-%d", rev["timestamp"])
    #Append the date count on dict
    if date not in edits:
        edits[date] = dict(sentiments = list(), edit_count=0)

    edits[date]["edit_count"] += 1
    #Append the sentiment value on dict
    edits[date]["sentiments"].append(find_sentiment(rev.get('comment', '')))

from statistics import mean

for key in edits:
    #Number of times a sentiment was found
    if len(edits[key]["sentiments"]) > 0:
        #Find the average for the value of a single feeling in the day
        edits[key]["sentiment"] = mean(edits[key]["sentiments"])
        #Find the percentage of edits where sentiment was negative
        edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s > 0]) / len(edits[key]["sentiments"])
    else:
        #Find the number of times the sentiment was neutral
        edits[key]["sentiment"] = 0
        edits[key]["neg_sentiment"] = 0

    del edits[key]["sentiments"]

#Create a dataframe pandas
edits_df = pd.DataFrame.from_dict(edits, orient='index')

edits_df

#Convert the index to datetime format
edits_df.index = pd.to_datetime(edits_df.index)

from datetime import datetime

dates = pd.date_range(start = '2009-03-08', end = datetime.today())

#On dates when there is no data from Wikipedia, it will fill it with zero
edits_df = edits_df.reindex(dates, fill_value=0)

edits_df

rolling_edits = edits_df.rolling(30).mean()

rolling_edits

rolling_edits = rolling_edits.dropna()

#Saving the dataset in a .csv file to later train the machine learning model
rolling_edits.to_csv("wikipedia_edits.csv")

!pip install yfinance

import yfinance as yf
import os

#Let's create a ticker object to fetch the historical price of Bitcoin
btc_ticker = yf.Ticker('BTC-USD')
#Get all Bitcoin data
btc = btc_ticker.history(period='max')

btc

#Convert index to datetime format
btc.index = pd.to_datetime(btc.index)
btc.index = btc.index.tz_localize(None)

del btc['Dividends']
del btc['Stock Splits']

btc.columns = [i.lower() for i in btc.columns]

btc

wiki = pd.read_csv('wikipedia_edits.csv', index_col = 0, parse_dates = True)

btc = btc.merge(wiki, left_index=True, right_index=True)

btc

btc['tomorrow'] = btc['close'].shift(-1)
btc

btc.to_csv('output.csv',index=False)

df=btc.drop('sentiment',axis=1)

df=btc.drop('neg_sentiment',axis=1)

df=btc.drop('tomorrow',axis=1)

df=btc.drop('edit_count',axis=1)

btc.head()

tail=[]
tail = btc.tail()
tail

import matplotlib.pyplot as plt


# Plotting the column as a line plot
plt.plot(btc['tomorrow'])

# Adding labels and title to the plot
plt.xlabel('date')
plt.ylabel('Price')
plt.title('Bitcoin price')

# Displaying the plot
graph = plt.show()
graph

btc['target'] = (btc['tomorrow'] > btc['close']).astype(int)

btc



btc['target'].value_counts()

from sklearn.ensemble import RandomForestClassifier

#Let's start by training 100 individual decision trees
#Each individual decision tree can only split its node when there are more than 50 samples
model = RandomForestClassifier(n_estimators = 100, min_samples_split=50, random_state=1)

#Splitting the data to create a training model and a test model
train = btc.iloc[:-200]
test = btc[-200:]

#Define the predictors
predictors = ['close', 'volume', 'open', 'high', 'low', 'edit_count', 'sentiment', 'neg_sentiment']
model.fit(train[predictors], train['target'])

from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error

preds = model.predict(test[predictors])
preds = pd.Series(preds, index = test.index)
precision_score(test['target'], preds)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train['target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index, name='predictions')
    #Combine real and predicted values into a single dataset
    combined = pd.concat([test['target'], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=1095, step=150):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    #Concatenate all forecasts
    return pd.concat(all_predictions)

from xgboost import XGBClassifier

model = XGBClassifier(random_state = 1, learning_rate = .1, n_estimators = 200)
predictions = backtest(btc, model, predictors)

#Now, let's look at our prediction score. The closer to one, the better the performance of the model.
precision_score(predictions['target'], predictions['predictions'])

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset

# Extract the features (X) and target variable (y)
X = df[['open', 'high', 'low', 'volume']]
y = df['close']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X = btc[['close', 'volume', 'open', 'high', 'low', 'edit_count', 'sentiment', 'neg_sentiment']]
y = btc['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)


rf_regressor.fit(X_train, y_train)


y_pred = rf_regressor.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

pip install mysql-connector-python

import mysql.connector

# Establish connection
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="MYSQL@2001",
    database="DSS"
)

# Create a cursor object to interact with the database
cursor = connection.cursor()

# Already a table called 'btc_data' has in your MySQL database
# with columns 'date', 'close', and 'tomorrow'

# Iterate over the DataFrame rows and insert the data into the MySQL table
for index, row in btc.iterrows():
    date = row['date']
    close = row['close']
    tomorrow = row['tomorrow']

    query = "INSERT INTO btc_data (date, close, tomorrow) VALUES (%s, %s, %s)"
    values = (date, close, tomorrow)

    cursor.execute(query, values)

# Commit the changes and close the cursor and connection
connection.commit()
cursor.close()
connection.close()

netstat -tuln | grep 3306

