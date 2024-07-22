# import libraries  
import streamlit as st  
import yfinance as yf  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns   
import plotly.graph_objects as go   
import plotly.express as px  
import datetime 
from datetime import date, timedelta  
from statsmodels.tsa.seasonal import seasonal_decompose   
import statsmodels.api as sm   
# Title  
st.title("Stock Price Analysis App") 
st.write("This streamlit app helps you explore historical stock data, perform time series analysis, and experiment with various prediction models.")


#sidebar  
st.sidebar.header('Select the parameters from below')  
start_date = st.sidebar.date_input('Start date', date(2003, 1, 1))  
end_date = st.sidebar.date_input('End date', date(2024, 5, 17))  

# add ticker symbol list  
ticker_list =["KOTAKBANK.NS"]
ticker = st.sidebar.selectbox('Select the company', ticker_list)  
# fetch data from user inputs using yfinance library  
data = yf.download(ticker, start=start_date,end=end_date)  
# add Date as a column to the dataframe  
data.insert(0, "Date", data.index, True)  
data.reset_index(drop=True, inplace=True)  
st.write('Data from', start_date, 'to', end_date)  
st.write(data)  
