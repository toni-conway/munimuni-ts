# General Libraries
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from pmdarima.arima import ADFTest
from statsmodels.tsa.stattools import adfuller
import joblib

#import statsmodels.api as sm
#from statsmodels.tsa.arima_model import ARIMA

# Model deployment
from flask import Flask
import streamlit as st

#preprocess date column
charts_df = pd.read_csv('data/ph_spotify_daily_charts.csv')
charts_df['date'] = pd.to_datetime(charts_df['date'])
charts_df = charts_df.set_index('date')

#model = joblib.load(open('arima.pkl','rb'))
collab_artists=['Magnus Haven','December Avenue','IV Of Spades','Zack Tabudlo',
                'Arthur Nery','Sunkissed Lolas','SB19','Calein','Up Dharma Down',
                'Leanne & Naara','juan karlos']

st.title("Forecasting Munimuni's Possible Collab Artist Performance")
html_temp = """
<div style="background:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;"> View Artist Stream Performance ML App </h2>
</div>
"""

st.markdown(html_temp, unsafe_allow_html = True)

#adding a selectbox
choice = st.selectbox(
    "Select Artist:",
    options = collab_artists)

if st.button("Visualize Forecast"):
    data = charts_df[(charts_df['artist']==choice)][['streams']].resample('D').sum()
    data = data.asfreq('D') # Add to complete dates
    data['streams'] = data['streams'].fillna(method='ffill')
       
    #adf test
    #adf_test = ADFTest(alpha = 0.05)
    #adf_test.should_diff(data)
    
    #testing if stationary
#     def d_counter(data):
#         d=0
#         p_value=adfuller(data)[1]
#         while p_value>0.05:
#            #print(p_value)
#            data=data.diff()[1:]
#            #print(p_value)
#            d+=1
#         return d
        

    #train test split
    train_data = data[:round(len(data)*0.7)]
    test_data = data[round(len(data)*0.7):]

    #arima_model
    arima_model = auto_arima(train_data, start_p=0, d=1, start_q=0,
                         max_p=2, max_d=1,max_q=2, start_P=0,
                         D=1, start_Q=0, max_P=2, max_D=2, max_Q=2, m=3, seasonal=True,
                         error_action='warn', trace=True,
                         supress_warnings=True, stepwise=True,
                         random_state=20, n_fits=50)
    
    #prediction
    #prediction = pd.DataFrame(arima_model.predict(n_periods=700),index=test_data.index)
    #prediction.columns = ['predicted_streams']
    
    #forecast
    forecast = arima_model.fit_predict(data, n_periods=60)
    
    plt.figure(figsize=(8,5))
    plt.plot(train_data, label="Training", color="#808080")
    plt.plot(test_data, label="Test", color="#d3d3d3")
    #plt.plot(prediction, label="Predicted", color="#00008b")
    plt.plot(forecast, label="Forecasted", color="#FFA500")
    
    plt.legend()
        
    st.pyplot(plt)
    
    
    
