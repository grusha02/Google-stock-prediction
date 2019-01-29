# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:26:25 2018

@author: GRUSHA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

trainingset=pd.read_csv('Google_Stock_Price_Train.csv')
trainingset=trainingset.iloc[:,1:2].values

#We scale the features that is normalise as we use the sigmoid function
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
trainingset=sc.fit_transform(trainingset)

#input and output, input:stock price t to predict t+1
X_train=trainingset[0:1257]
Y_train=trainingset[1:1258]

#converting 2D xtrain to 3D for predict method
#reshape functions takes the new format, here the timestamp is 1 (t+1-t)
X_train=np.reshape(X_train,(1257,1,1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#here we use regressor as regression is being done (continuous nature of output) and not classification
regressor=Sequential()
#input_shape takes timestamp and number of features
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam',loss='mean_squared_error')

regressor.fit(X_train,Y_train,batch_size=32,epochs=200)

testset=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=testset.iloc[:,1:2].values
inputs=real_stock_price
inputs=sc.fit_transform(inputs)

inputs=np.reshape(inputs,(20,1,1))

predict_price=regressor.predict(inputs)

predict_price=sc.inverse_transform(predict_price)

#ploting the results
plt.plot(real_stock_price,color='red',label='Real stock price of Google')
plt.plot(predict_price,color='blue',label='Predicted stock price of Google')
plt.title('Predicting stock price of google')
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()

#evaluating root mean squared error
import math
from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(real_stock_price,predict_price))