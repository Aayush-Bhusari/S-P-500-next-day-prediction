# S&P 500 Next-Day Direction Prediction

A machine learning pipeline to predict the next-day price direction of the S&P 500 index using 25 years of historical data.

## Overview

This project trains a Random Forest classifier on 18 engineered features derived from price, volume, and technical indicators. A rolling backtesting system is used to evaluate real-world performance and prevent look-ahead bias.

## Features Used

- Lag features (1-day, 5-day)
- Moving average ratios (MA10, MA50, MA200)
- RSI (14-day)
- Daily, 5-day, and 21-day returns
- Volatility (10-day, 21-day)
- Momentum (10-day, 21-day)
- Volume ratios (MA10, MA50)
- Candlestick shadows (Upper, Lower)
- Day of week

## Tech Stack

- Python, Pandas, NumPy, Matplotlib
- scikit-learn (RandomForestClassifier, GridSearchCV, TimeSeriesSplit)
- yfinance

## Results

- Backtested accuracy: ~50.5%
- Model performance varies significantly across market regimes, peaking above 65% in trending markets and dropping during high-volatility periods such as the 2020 COVID crash.

## How to Run

1. Open the notebook in Google Colab
2. Run all cells in order
