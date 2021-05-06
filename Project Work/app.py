from flask import Flask, render_template, url_for, redirect, make_response, jsonify
from datetime import datetime
from flask import request
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from pandas_datareader import data as api 
from pandas_datareader._utils import RemoteDataError 
from datetime import datetime
from dateutil.relativedelta import relativedelta
import csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time
import math


# #end_time is current time fetched from the local machine
# end_date=datetime.now()
# #start_date is the date before 1 years
# start_date=end_date-relativedelta(years=1)
# #formatting dates in suitable forms
# end_date=str(end_date.strftime('%Y-%m-%d'))
# start_date=str(start_date.strftime('%Y-%m-%d'))
# # print(end_date)
# # print(start_date)
# company_name='^NSEI'
# stock_data=api.DataReader(company_name,'yahoo',start_date,end_date)
# json = stock_data.to_json()
# print(type(json))

app = Flask(__name__)


@app.route('/', methods = ['POST','GET'])
def index():
        return render_template('index.html')

@app.route('/index.html', methods = ['POST','GET'])
def index2():
        return render_template('index.html')

@app.route('/blog.html', methods = ['POST','GET'])
def blog():
        return render_template('blog.html')

@app.route('/documentation.html', methods = ['POST','GET'])
def documentation():
        return render_template('documentation.html')

@app.route('/findCompany', methods = ['POST'])
def data2():
        req = request.get_json()
        print(req)
        symboldict={
        "BAJAJ FINSERV":"BAJAJFINSV.NS",
        "TCS":"TCS.NS",
        "L&T":"LT.NS",
        "TECH MAHINDRA":"TECHM.NS",
        "BRITANNIA":"BRITANNIA.NS",
        "SHREE CEMENT":"SHREECEM.NS",
        "ONGC":"ONGC.NS",
        "BAJAJ AUTO":"BAJAJ-AUTO.NS",
        "ICICI BANK":"ICICIBANK.NS",
        "COAL INDIA":"COALINDIA.NS",
        "GRASIM":"GRASIM.NS",
        "HERO MOTOCORP":"HEROMOTOCO.NS",
        "ULTRATECH CEMENT":"ULTRACEMCO.NS",
        "TATA CONSUMER":"TATACONSUM.NS",
        "NESTLE":"NESTLEIND.NS",
        "HINDALCO":"HINDALCO.NS",
        "BAJAJ FINANCE":"BAJFINANCE.NS",
        "MARUTI SUZUKI":"MARUTI.NS",
        "INDUSIND BANK":"INDUSINDBK.NS",
        "TATA STEEL":"TATASTEEL.NS",
        "TITAN":"TITAN.NS",
        "CIPLA":"CIPLA.NS",
        "ITC":"ITC.NS",
        "HDFC LIFE INSURANCE":"HDFCLIFE.NS",
        "WIPRO":"WIPRO.NS",
        "KOTAK MAHINDRA BANK":"KOTAKBANK.NS",
        "NTPC":"NTPC.NS",
        "RELIANCE IND.":"RELIANCE.NS",
        "BHARTI AIRTEL":"BHARTIARTL.NS",
        "YES BANK":"YESBANK.NS",
        "IDBI BANK":"IDBI.NS",
        "IDEA":"IDEA.NS",
        "RELIANCE COMMUNICATION":"RCOM.NS",
        "TATA MOTORS":"TATAMOTORS.NS",
        "BANK OF BARODA":"BANKBARODA.NS",
        "STATE BANK OF INDIA":"SBIN.NS",
        "TATA POWER":"TATAPOWER.NS",
        "GAIL LTD.":"GAIL.NS",
        "NANDAN DENIM LTD.":"NDL.BO",
        "SENSEX":"^BSESN",
        "NIFT 50":"^NSEI",
        "DOW":"^DJI",
        "PUNJAB NATIONAL BANK":"PNB.NS",
        "VIVO BIO TECH":"VIVOBIOT.BO",
        "APPLE":"AAPL",
        "AMAZON":"AMZN",
        "FACEBOOK":"FB",
        "GOOGLE":"GOOGL",
        "LOREAL":"OR.PA",
        "UNILEVER":"UL",
        "P&G":"PG",
        "MICROSOFT":"MSFT",
        "RAYMOND":"RAYMOND.NS",
        "PHILIPS":"PHG",
        "TESLA":"TSLA",
        "SONY":"SONY",
        "NOKIA":"NOKIA.HE"
        }
        end_date=datetime.now()
        #start_date is the date before 1 years
        start_date=end_date-relativedelta(years=1)
        #formatting dates in suitable forms
        end_date=str(end_date.strftime('%Y-%m-%d'))
        start_date=str(start_date.strftime('%Y-%m-%d'))
        # print(end_date)
        # print(start_date)
        company_name=req['name']
        stock_price_dataframe=api.DataReader(symboldict[company_name],'yahoo',start_date,end_date)
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


@app.route('/predict', methods = ['GET','POST'])
def dataPredict():
        req = request.get_json()
        print(req)
        res = make_response(str(lstm(req)), 200)
        return res

#Function-name:give_close_price
    ##Description:Function which returns only close price of 1 year for given company
#Input parameters: company_ticker
    ##company_ticker: is a string which has the ticker name of company according to yahoo-finance for which we want to fetch the close                      price data
#Output parameters:data
    ##data: 1*366 numpy array which contains the close price of 1 year for the given company


def give_close_price(company_ticker):
    #end_time is current time fetched from the local machine
    end_date=datetime.now()
    #start_date is the date before 1 year
    start_date=end_date-relativedelta(years=1)
    #formatting dates in suitable form
    #both start_date and end_date are string of dates in (yyyy-mm-dd) format
    end_date=str(end_date.strftime('%Y-%m-%d'))
    start_date=str(start_date.strftime('%Y-%m-%d'))
    #stock_dataframe is in the format of pandas dataframe fetched from yahoo API
    stock_price_dataframe=api.DataReader(company_ticker,'yahoo',start_date,end_date)

    #Forward fill and backward fill on holidays and NAN
    stock_price_dataframe_clean=stock_price_dataframe.reindex(pd.date_range(start_date,end_date))
    stock_price_dataframe_clean=stock_price_dataframe_clean.fillna(method='ffill')
    stock_price_dataframe_clean=stock_price_dataframe_clean.fillna(method='bfill')

    #clean and neat data of Close price of given company for past 1 year in 1*366 numpy array
    data=stock_price_dataframe_clean['Close']
    data=data.to_numpy()
    return data


#Function-name: split_the_data
    ##Description: split the data into train and test set such that it's suitable for LSTM model
#Input parameters: data_scaled,fraction_of_train, number_of_features
    ##data_scaled: 1*366 numpy array which contains the scaled data of close price
    ##fraction_of_train: fraction of the data which should be used as training set, default value is 0.6
    ##number_of_features: is the number of past data-points used by LSTM to predict the future value, default value=50
#Output parameters:X_train, X_test, y_train, y_test
    ##X_train: features for the training dataset
    ##X_test: features for the testing dataset
    ##y_train: targets for the training dataset
    ##y_test: targets for the testing dataset
def split_the_data(data_scaled,fraction_of_train=0.6,number_of_features=50):
    data_scaled_length=data_scaled.shape[0]
    train_length=int(data_scaled_length*fraction_of_train)
    train, test = data_scaled[0:train_length,:], data_scaled
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
    return X_train, X_test, y_train, y_test



#Function-name: train_and_predict
    ##Description: to tain the LSTM model, and to do predictions
#Input parameters:X_train, y_train, X_test, number_of_features
    ##X_train: features for the training dataset
    ##y_train: targets for the training dataset
    ##X_test: features for the testing dataset
    ##number_of_fetures:  is the number of past data-points used by LSTM to predict the future value, default value=50
#Output parameters:y_pred
    ##y_pred: predicted output 

def train_and_predict(X_train, y_train, X_test,number_of_features=50):
    model=Sequential()
    model.add(LSTM(units=200, activation='relu', return_sequences=True, input_shape=(number_of_features,1)))
    model.add(LSTM(units=50, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train,y_train,epochs=25,batch_size=64,verbose=1)

    y_pred=model.predict(X_test)

    return y_pred


#Function name: lstm
    ##Description: to perform LSTM on the given company's close price
#Input parameters: company_name, fraction_of_train, number_of_features
    ##company_name: is the name of the company for which we want to verify the LSTM model.
    ##fraction_of_train: is the fraction of the total data using which we want to train the model, default value=0.6
    ##number_of_features: is the number of past data-points used by LSTM to predict the future value, default value=50
#Output parameters: y_actual, y_pred
    ##final_pred: prediction of close price for tomorrow

def lstm(company_name,fraction_of_train=0.6,number_of_features=50):
    #dictionary to map company name to ticker name
    symboldict2={
    "BAJAJ FINSERV":"BAJAJFINSV.NS",
    "TCS":"TCS.NS",
    "L&T":"LT.NS",
    "TECH MAHINDRA":"TECHM.NS",
    "BRITANNIA":"BRITANNIA.NS",
    "SHREE CEMENT":"SHREECEM.NS",
    "ONGC":"ONGC.NS",
    "BAJAJ AUTO":"BAJAJ-AUTO.NS",
    "ICICI BANK":"ICICIBANK.NS",
    "COAL INDIA":"COALINDIA.NS",
    "GRASIM":"GRASIM.NS",
    "HERO MOTOCORP":"HEROMOTOCO.NS",
    "ULTRATECH CEMENT":"ULTRACEMCO.NS",
    "TATA CONSUMER":"TATACONSUM.NS",
    "NESTLE":"NESTLEIND.NS",
    "HINDALCO":"HINDALCO.NS",
    "BAJAJ FINANCE":"BAJFINANCE.NS",
    "MARUTI SUZUKI":"MARUTI.NS",
    "INDUSIND BANK":"INDUSINDBK.NS",
    "TATA STEEL":"TATASTEEL.NS",
    "TITAN":"TITAN.NS",
    "CIPLA":"CIPLA.NS",
    "ITC":"ITC.NS",
    "HDFC LIFE INSURANCE":"HDFCLIFE.NS",
    "WIPRO":"WIPRO.NS",
    "KOTAK MAHINDRA BANK":"KOTAKBANK.NS",
    "NTPC":"NTPC.NS",
    "RELIANCE IND.":"RELIANCE.NS",
    "BHARTI AIRTEL":"BHARTIARTL.NS",
    "YES BANK":"YESBANK.NS",
    "IDBI BANK":"IDBI.NS",
    "IDEA":"IDEA.NS",
    "RELIANCE COMMUNICATION":"RCOM.NS",
    "TATA MOTORS":"TATAMOTORS.NS",
    "BANK OF BARODA":"BANKBARODA.NS",
    "STATE BANK OF INDIA":"SBIN.NS",
    "TATA POWER":"TATAPOWER.NS",
    "GAIL LTD.":"GAIL.NS",
    "NANDAN DENIM LTD.":"NDL.BO",
    "SENSEX":"^BSESN",
    "NIFT 50":"^NSEI",
    "DOW":"^DJI",
    "PUNJAB NATIONAL BANK":"PNB.NS",
    "VIVO BIO TECH":"VIVOBIOT.BO",
    "APPLE":"AAPL",
    "AMAZON":"AMZN",
    "FACEBOOK":"FB",
    "GOOGLE":"GOOGL",
    "LOREAL":"OR.PA",
    "UNILEVER":"UL",
    "P&G":"PG",
    "MICROSOFT":"MSFT",
    "RAYMOND":"RAYMOND.NS",
    "PHILIPS":"PHG",
    "TESLA":"TSLA",
    "SONY":"SONY",
    "NOKIA":"NOKIA.HE"
    }

    

    #fetching the close price of 1 year for the given company
    nameX=company_name['namePredict']
    data=give_close_price(symboldict2[nameX])
    

    #scaling the data for better performance, data_scaled is 366*1 numpy array
    my_scaler_1=MinMaxScaler(feature_range=(0,1))
    data_scaled=my_scaler_1.fit_transform(data.reshape(-1,1))

    #Splitting the data_scaled into to numpy arrays according to the fraction_of_train
    #Note that train and test both are coloumn vectors
    X_train, X_test, y_train, y_test=split_the_data(data_scaled,fraction_of_train)

    #Prediction for the close price
    y_pred=train_and_predict(X_train, y_train, X_test, number_of_features)
    

    #Inverse scaling of the predicted data
    y_pred=my_scaler_1.inverse_transform(y_pred.reshape(-1,1))

    #calculating the rmse
    y_actual=data[number_of_features:366]
    rmse=math.sqrt(np.mean((y_pred-y_actual)**2))

    #adding extra elements in y_pred
    y_pred=np.concatenate([np.zeros(number_of_features),y_pred[:,0]])
    y_actual=data
    final_pred=y_pred[-1]

    #returning the stock price of tomorrow
    return final_pred



if __name__ == "__main__":
    app.run(debug = True)