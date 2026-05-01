# S&P 500 Next-Day Direction Classifier

A machine learning pipeline to predict whether the S&P 500 index will close higher or lower the following trading day.

## Overview

Trained a Random Forest classifier on 25 years of historical S&P 500 data using 18 engineered features derived from price, volume, and technical indicators. A rolling backtesting system evaluates real-world performance and prevents look-ahead bias. Hyperparameters were tuned using GridSearchCV with TimeSeriesSplit cross-validation.

## Features Used

- Lag features (1-day, 5-day)
- Moving average ratios (MA10, MA50, MA200)
- RSI (14-day)
- Returns (1-day, 5-day, 21-day)
- Volatility (10-day, 21-day)
- Momentum (10-day, 21-day)
- Volume ratios (MA10, MA50)
- Candlestick shadows (Upper, Lower)
- Day of week

## Tech Stack

Python, Pandas, NumPy, Matplotlib, scikit-learn, yfinance

## Learning Outcomes

- Built a complete ML pipeline for time-series financial data from scratch
- Understood why shuffling data is invalid for time-series and implemented time-aware splitting
- Learned the difference between overfitting to a test set vs genuine generalization
- Implemented a rolling backtesting system that simulates real-world trading conditions
- Applied GridSearchCV with TimeSeriesSplit for proper hyperparameter tuning on sequential data
- Understood the Efficient Market Hypothesis in practice — price-based features alone cannot consistently predict market direction
