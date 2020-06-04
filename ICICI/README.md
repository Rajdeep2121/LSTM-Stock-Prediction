# ICICI Bank Analysis of Stock Prices from 2000-2020

An LSTM model is built to predict the opening value of stock prices for each day since the year 2000. 


The data is cleaned first to remove all the 'NaN' values. Now the data has 4900 data points. Then a plot is made to visualize the prices from 2000 to 2020.<br> 
<img src="icici1.PNG" width="900" height="200"><br>
After applying feature scaling, input data is provided to the LSTM in streams of 100 prices at a time.
<hr>

# Architecture of the LSTM model:
1) An LSTM layer with 1000 nodes and input shape of 100<br>
2) Another LSTM layer with 500 nodes<br>
3) Dropout of 0.2<br>
4) A hidden dense layer with 128 nodes<br>
5) An output layer with one node 


The loss function used is 'mean_squared_error' and the optimizer used is 'adam'.


The model is trained on <b><u>3000 data points</u></b> for 200 epochs with batch size of 256.
<hr>

# Predictions 

The prediction is made on the data points from 2000 to 4900.<br>
<img src="icici2.PNG" width="900" height="200">
<hr>

# RMSE Score

The rmse is calculated out to be: 1.17%.
<hr>