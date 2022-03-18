# Darts API Playground

[![Darts and Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/gerardrbentley/darts-playground/main)

:construction: Under Development.

:warning: Not all models compiled in cloud.
Tested list below

A playground web app for the [Darts](https://unit8co.github.io/darts/) API with [Streamlit](https://streamlit.io)!

Featuring:

- Example datasets
- Upload your own dataset
- Model training tuning
- Model forecasting and plotting controls
- Dataset Seasonality, Trend, and other Metrics
- Error Metrics over forecasted periods
- Historical Forecasting
- Backtested Error Metrics
- Flexible forecasting horizon and stride for backtesting
- Downloadable forecasts and data

## Explore A Time Series!

Use your own csv data that has a well formed time series and plot some forecasts!

Or use one of the example Darts [datasets](https://github.com/unit8co/darts/tree/master/datasets)

## Explorable Models

- [X] NaiveDrift
- [X] NaiveMean
- [X] NaiveSeasonal
- [X] ARIMA
- [X] VARIMA (Requires Multivariate dataset)
- [X] ExponentialSmoothing
- [X] LinearRegressionModel (Hand set Lag)
- [X] FFT
- [X] Theta
- [X] FourTheta
- [X] KalmanForecaster
- [X] LightGBMModel
- [X] RandomForest (Hand set Lag)
- [X] RegressionModel

## Not Yet Explorable Models

- Ensembles
  - [ ] NaiveEnsembleModel
  - [ ] EnsembleModel
  - [ ] RegressionEnsembleModel
- Neural Net Based
  - [ ] RNNModel (incl. LSTM and GRU); equivalent to DeepAR in its probabilistic version,True,True,True,True,False,True,DeepAR paper
  - [ ] BlockRNNModel (incl. LSTM and GRU),True,True,True,True,True,False,
  - [ ] NBEATSModel,True,True,True,True,True,False,N-BEATS paper
  - [ ] TCNModel,True,True,True,True,True,False,"TCN paper, DeepTCN paper, blog post"
  - [ ] TransformerModel,True,True,True,True,True,False,
  - [ ] TFTModel (Temporal Fusion Transformer),True,True,True,True,True,True,"TFT paper, PyTorch Forecasting"
  - [ ] Prophet

## More

Cast to np.float32 to slightly speedup the training