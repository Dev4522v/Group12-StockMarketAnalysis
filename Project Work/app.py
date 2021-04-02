from flask import Flask, render_template, url_for, redirect, make_response, jsonify
from datetime import datetime
from flask import request
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from pandas_datareader import data 
from pandas_datareader._utils import RemoteDataError 
from datetime import datetime
from dateutil.relativedelta import relativedelta
import csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


#end_time is current time fetched from the local machine
end_date=datetime.now()
#start_date is the date before 1 years
start_date=end_date-relativedelta(years=1)
#formatting dates in suitable forms
end_date=str(end_date.strftime('%Y-%m-%d'))
start_date=str(start_date.strftime('%Y-%m-%d'))
# print(end_date)
# print(start_date)
company_name='^NSEI'
stock_data=data.DataReader(company_name,'yahoo',start_date,end_date)
json = stock_data.to_json()
print(type(json))

app = Flask(__name__)


@app.route('/', methods = ['POST','GET'])
def index():
        return render_template('index.html')

@app.route('/findCompany', methods = ['POST'])
def data2():
        req = request.get_json()
        print(req)
        end_date=datetime.now()
        #start_date is the date before 1 years
        start_date=end_date-relativedelta(years=1)
        #formatting dates in suitable forms
        end_date=str(end_date.strftime('%Y-%m-%d'))
        start_date=str(start_date.strftime('%Y-%m-%d'))
        # print(end_date)
        # print(start_date)
        company_name=req['name']
        stock_price_dataframe=data.DataReader(company_name,'yahoo',start_date,end_date)
        # close_price_dataframe=stock_price_dataframe['Close']
        # close_price_dataframe=pd.DataFrame(close_price_dataframe,columns=['Close'])

        #Cleaning the data by filling up the average value on holidays and NAN
        stock_price_dataframe_clean=stock_price_dataframe.reindex(pd.date_range(start_date,end_date))
        stock_price_dataframe_clean=stock_price_dataframe_clean.fillna(method='ffill')
        stock_price_dataframe_clean=stock_price_dataframe_clean.fillna(method='bfill')


        #To see the final dataframe uncomment the below lines
        ##print(close_price_dataframe_clean.shape)

        #convert the final dataframe into close_price_csv file
        csved = stock_price_dataframe_clean.to_csv()
        res = make_response(csved, 200)
        return res

#Function name: lstm
    ##Description: Fetch the data from csv file or yahoo API and then split the given data into train and test sets and then apply LSTM.
#Input parameters: company_ticker, fraction_of_train, number_of_features
    ##Description: company_ticker is the ticker of the company for which we want to verify the LSTM model.
    ##Description: fraction_of_train is the fraction of the total data using which we want to train the model, default value=0.6
    ##Description: number_of_features is the number of past data-points used by LSTM to predict the future value, default value=50
    ##Description: number_of_days is denoting for how many days we want to predict the future, default value=30 days
#Output parameters: data, test_predict, train_length
    ##Description: data is 1D numpy aray of 1*366 which contains the Close data of given company for 1 year
    ##Description: test_predict is the predicted values for given fraction_of_train.number_of_features. 
    ##Description: y_future is 1D numpy array of size 1*number_of_days containing future Close price 
    ##Description: train_length is the number of data-points from data array which we are using to train the LSTM model.
def lstm(company_ticker,fraction_of_train=0.6,number_of_features=50,number_of_days=30):
    #clean and neat data of Close price of given company for past 1 year in 1*366 numpy array(loading from csv file)
    data=pd.read_csv('close_price_csv.csv')
    data=data['Close']
    data=data.to_numpy()
    data_length=data.shape[0]

    #scaling the data for better performance, data_scaled is 366*1 numpy array
    my_scaler_1=MinMaxScaler(feature_range=(0,1))
    data_scaled=my_scaler_1.fit_transform(data.reshape(-1,1))
    data_scaled_length=data_scaled.shape[0]

    #Just splitting the data_scaled into to numpy arrays according to the fraction_of_train
    #Note that train and test both are coloumn vectors
    train_length=int(data_scaled_length*fraction_of_train)
    train, test = data_scaled[0:train_length,:], data_scaled[train_length:data_scaled_length,:]
    test_length=test.shape[0]

    #Converting the train and test dataset into suitable format for LSTM model
    X_train=np.zeros([train_length-number_of_features,number_of_features])
    X_test=np.zeros([test_length-number_of_features,number_of_features])
    y_train=np.zeros(train_length-number_of_features)
    y_test=np.zeros(test_length-number_of_features)
    for i in range(train_length-number_of_features):
        X_train[i]=train[i:i+number_of_features].T
        y_train[i]=train[i+number_of_features]
    for i in range(test_length-number_of_features):
        X_test[i]=test[i:i+number_of_features].T
        y_test[i]=test[i+number_of_features]
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    
    #Creating the architecture for the LSTM model
    model=Sequential()
    model.add(LSTM(units=200, activation='relu', return_sequences=True, input_shape=(number_of_features,1)))
    model.add(LSTM(units=50, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    #Training the model
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)

    #Prediction on the test dataset
    test_predict=model.predict(X_test)

    #inverse scaling of the obtained prediction
    test_predict=my_scaler_1.inverse_transform(test_predict)

    #preapring the dataset to predict the future
    X_future=np.zeros([number_of_days,number_of_features])
    y_future=np.zeros([number_of_days])
    X_future[0]=data_scaled[data_scaled_length-number_of_features:data_scaled_length].T

    #predicting the future
    for i in range(number_of_days):
        temp=X_future[i].reshape(-1,1)
        temp=temp.reshape((1,number_of_features,1))
        y_future[i]=model.predict(temp)
        if i<number_of_days-1:
            X_future[i+1,0:number_of_features-1]=X_future[i,1:number_of_features]
            X_future[i+1][-1]=y_future[i]

    #inverse scaling of the obtained future
    y_future=my_scaler_1.inverse_transform(y_future.reshape(-1,1))      
    
    return data, test_predict, y_future, train_length 




if __name__ == "__main__":
    app.run(debug = True)