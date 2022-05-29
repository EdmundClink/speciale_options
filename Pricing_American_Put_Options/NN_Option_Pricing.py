import matplotlib
from numpy import loadtxt
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import os
import os.path
from keras.models import save_model, load_model

os.chdir('path_working_directory')

from Build_Model import build, loadData, splitDataXY, plotTrainingMetrics, splitTrainAndTest, build_BS

# Load the dataset
data = loadData('path_file/input_data/spyputs.csv') 
#description = data.describe()

# Add a column with the index for each sample. This is used for selecting the test and training test
data['index'] = range(1, len(data) + 1)

num_of_sample = int(0.75 * len(data))
num_of_epochs = 1200
test_loss_min = 1

## Read in data
data = pd.read_csv("spyputs.csv", sep=',')
data = data.rename(columns={'date': 'Date', 'exp_date':'Maturity'})
data.Date = pd.to_datetime(data.Date, format='%d/%m/%Y')
data.Maturity = pd.to_datetime(data.Maturity, format='%d/%m/%Y')
data['mid_price'] = (data.best_bid + data.best_offer)/2
data['strike_price'] = data['strike_price']/1000
data = data.assign(days_maturity=((data.Maturity - data.Date).dt.days).values)
data = data[['Date', 'Maturity', 'days_maturity', 'strike_price', 'mid_price', 'volume', 'S0']]

data['strike_norm'] = data['strike_price'] / data['strike_price']
data['close_norm'] = data['mid_price'] / data['strike_price']
data['maturity_annual'] = data['days_maturity'] / 252 
data['price_norm'] = data['mid_price'] / data['strike_price']

FED3month = pd.read_csv("FED3month.csv", sep=',')
FED3month = FED3month.rename(columns={'DATE': 'Date', 'DTB3' : 'interest_rate'})
FED3month = FED3month[FED3month.interest_rate !='.']
FED3month.reset_index(drop=True, inplace=True)
FED3month.Date = pd.to_datetime(FED3month.Date, format='%Y-%m-%d')

data = pd.merge(data, FED3month, on='Date', how='left')

volatility_measures = pd.read_csv("volatility_measures.csv", sep=',')
volatility_measures = volatility_measures.rename(columns={'date': 'Date'})
volatility_measures.Date = pd.to_datetime(volatility_measures.Date, format='%m/%d/%Y')
volatility_measures.reset_index(drop=True, inplace=True)
volatility_measures = volatility_measures[['Date','sigma30','sigma60','sigma120','VIX']]

data = pd.merge(data, volatility_measures, on='Date', how='left')

data = data[data.Date !='2017-10-09']
data['interest_rate'] = data.interest_rate.astype(float)
data['interest_rate'] = data['interest_rate'] / 100
data['VIX'] = data['VIX'] / 100

Market_price = data['mid_price']
Strike = data['strike_price']

Market_price = np.array(Market_price)
Strike = np.array(Strike)

# Build the model
X = data[['strike_norm','maturity_annual','close_norm','interest_rate','sigma30','sigma60','sigma120']]
Y = data[['price_norm']]

X = np.array(X)
Y = np.array(Y)

##NN MODEL 1
#model = build_BS(X)

#NN MODEL 2
model = build(X)

# Container for the training data:
val_loss = list()
train_loss = list()
test_loss = list()

X_train, X_test, Y_train, Y_test = splitTrainAndTest(X, Y, test_size=0.25)
Strike_train, Strike_test, Market_price_train, Market_price_test = splitTrainAndTest(Strike, Market_price, test_size=0.25)

#for i in range(num_of_epochs):

# Fit the model
history = model.fit(X_train, Y_train, batch_size=64, validation_split=0.25, epochs=num_of_epochs, verbose=1)

#Saving training metrics 
train_loss.append(history.history['loss'][0])

# Save testing metrics
score = model.evaluate(X_test, Y_test)
test_loss.append(score)

#filepath = 'path_to_file_model_name'
#save_model(model, filepath)

#Predicted values
Y_hat_train = model.predict([X_train])
Y_hat_test = model.predict([X_test])

#diff_NN = Y_test - Y_hat[:,0] 

#mse_NN = np.mean(diff_NN**2) 

#plt.plot(history.history['val_loss'])
#plt.plot(history.history['loss'])
#plt.title('Model Loss')
#plt.ylabel('Loss')
#plt.xlabel('Epochs')
#plt.xlim([-1, num_of_epochs])
#plt.ylim([-0.000001,0.00003])
#plt.xticks(np.arange(0,1400, step=200))
#plt.yticks(np.arange(0,0.000005, step=0.000001))
#plt.legend(['Test Loss','Training Loss'])

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.xlim([900, 1200])
plt.ylim([-0.000001,0.000005])
plt.xticks(np.arange(900,1250, step=50))
plt.yticks(np.arange(0,0.000005, step=0.000001))
plt.legend(['Test Loss','Training Loss'])
