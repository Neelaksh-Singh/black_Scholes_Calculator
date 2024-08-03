# Black-Scholes Option Pricing Calculator

## Overview

This Streamlit-based web application provides an interactive interface for options pricing using the Black-Scholes model. It's designed for traders, financial analysts, and students to quickly calculate option prices and visualize various aspects of options trading.

## Features

1. **Real-time Stock Data**: Fetches live stock prices for accurate calculations.

2. **Option Price Calculation**: Computes call and put option prices based on the Black-Scholes model.

3. **Greeks Calculation**: Displays Delta, Gamma, Theta, Vega, and Rho for both call and put options.

4. **Interactive Heatmaps**: 
   - Visualizes option prices across different stock prices and volatilities.
   - Shows potential profit/loss (PNL) for user-defined option positions.

5. **Greek Visualizations**: Plots selected Greeks against stock price changes.

6. **Custom Position Entry**: Allows users to input and analyze their own option positions.

7. **Flexible Parameters**: Users can adjust all key inputs including stock price, strike price, time to maturity, volatility, and risk-free rate.

## How It Helps

1. **Quick Pricing**: Instantly calculate option prices for any given set of parameters.

2. **Risk Assessment**: Understand the sensitivity of option prices to various factors through Greeks.

3. **Strategy Testing**: Use the PNL heatmap to visualize potential outcomes of different option strategies.

4. **Educational Tool**: Helps in understanding the relationship between different variables in options pricing.

5. **Market Analysis**: Compare theoretical prices with market prices to identify potential mispricings.

6. **Decision Support**: Assists traders in making informed decisions about option trades.

## How to Use

1. Select a stock ticker or input custom stock price.
2. Adjust the Black-Scholes parameters as needed.
3. View the calculated option prices and Greeks.
4. Explore the heatmaps to understand price sensitivities.
5. Add your own option positions to see potential PNL scenarios.
6. Use the Greek visualizations to deep dive into specific risk measures.

## 💻 Installation

`!!! Having Python is a must. !!!` <br>

<b>Port 8501 must be reserverd</b> <br>

Python 3.10+ is required to run code from this repo. 

```console
$ git clone https://github.com/Neelaksh-Singh/black_Scholes_Calculator.git
$ cd black_Scholes_Calculator/
$ python -m venv blackScholesEnv
$ blackScholesEnv\Scripts\activate.bat
$ pip install -r requirements.txt 
$ streamlit run app.py
```
