import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import plotly.graph_objs as go
import yfinance as yf
from datetime import date, datetime, timedelta

class BlackScholesModel:
    def __init__(self, r, S, K, T, sigma):
        self.r = r
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma

    def calculate_d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price_option(self, option_type='call', S=None, K=None, sigma=None):
        
        if S is None:
            S = self.S
        if K is None:
            K = self.K
        if sigma is None:
            sigma = self.sigma

        d1, d2 = self.calculate_d1_d2()
        if option_type.lower() == 'call':
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return round(price, 3)

    def calculate_greeks(self, option_type='call'):
        d1, d2 = self.calculate_d1_d2()
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                     - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            delta = -norm.cdf(-d1)
            theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                     + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)

        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)

        return {
            'delta': round(delta, 3),
            'gamma': round(gamma, 5),
            'theta': round(theta / 365, 4),
            'vega': round(vega * 0.01, 3),
            'rho': round(rho * 0.01, 3)
        }

class Position:
    def __init__(self, option_type, quantity, strike, premium):
        self.option_type = option_type
        self.quantity = quantity
        self.strike = strike
        self.premium = premium

def calculate_pnl(model, positions, spot_price, volatility):
    total_pnl = 0
    for position in positions:
        option_price = model.price_option(position.option_type, spot_price, position.strike, volatility)
        if position.option_type == 'call':
            pnl = (option_price - position.premium) * position.quantity
        else:  # put
            pnl = (option_price - position.premium) * position.quantity
        total_pnl += pnl
    return total_pnl


def create_pnl_heatmap(model, positions, min_s, max_s, min_v, max_v):
    spot_values = np.linspace(min_s, max_s, 10)
    vol_values = np.linspace(min_v, max_v, 10)
    pnl_values = np.array([[calculate_pnl(model, positions, s, v) 
                            for s in spot_values] for v in vol_values])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pnl_values, xticklabels=[f"{x:.0f}" for x in spot_values], 
                yticklabels=[f"{y:.2f}" for y in vol_values], 
                cmap='RdYlGn', annot=True, fmt=".2f", ax=ax, center=0)
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Volatility')
    ax.set_title('PNL Heatmap')
    return fig

@st.cache_data
def fetch_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        return round(stock.history(period="1d")['Close'].iloc[-1], 2)
    except:
        return None

def create_heatmap(model, option_type, min_s, max_s, min_v, max_v):
    spot_values = np.linspace(min_s, max_s, 10)
    vol_values = np.linspace(min_v, max_v, 10)
    prices = np.array([[BlackScholesModel(model.r, s, model.K, model.T, v).price_option(option_type) 
                        for s in spot_values] for v in vol_values])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(prices, xticklabels=[f"{x:.0f}" for x in spot_values], 
                yticklabels=[f"{y:.2f}" for y in vol_values], 
                cmap='viridis', annot=True, fmt=".2f", ax=ax)
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Volatility')
    ax.set_title(f'{option_type.capitalize()} Option Price Heatmap')
    return fig

def plot_greek(model, greek, option_type):
    spot_range = np.linspace(model.S * 0.8, model.S * 1.2, 100)
    greek_values = [BlackScholesModel(model.r, s, model.K, model.T, model.sigma).calculate_greeks(option_type)[greek] 
                    for s in spot_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=greek_values, mode='lines', name=greek.capitalize()))
    fig.update_layout(title=f'{greek.capitalize()} vs Spot Price ({option_type.capitalize()})',
                      xaxis_title='Spot Price', yaxis_title=greek.capitalize())
    return fig

def main():
    st.set_page_config(layout="wide", page_title="Black-Scholes Calculator")
    st.sidebar.title("Black-Scholes Model")

    # Sidebar inputs
    ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL, GOOGL)", value="AAPL")
    use_live_price = st.sidebar.checkbox("Use Live Stock Price", value=True)
    
    if use_live_price:
        S = fetch_stock_price(ticker)
        if S is None:
            st.sidebar.error(f"Could not fetch price for {ticker}. Using default value.")
            S = 100.0
        else:
            st.sidebar.success(f"Fetched price for {ticker}: ${S}")
    else:
        S = st.sidebar.number_input("Current Asset Price", value=100.0, step=0.01)

    K = st.sidebar.number_input("Strike Price", value=S, step=0.01, key="model_strike_price")
    T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, step=0.01, min_value=0.01)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01, format="%.2f")
    r = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, step=0.01, format="%.2f")

    st.sidebar.subheader("Position Entry")
    position_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])
    quantity = st.sidebar.number_input("Quantity", value=1, step=1)
    strike = st.sidebar.number_input("Strike Price", value=K, step=0.01, key="position_strike_price")
    premium = st.sidebar.number_input("Premium Paid", value=0.0, step=0.01)

    if st.sidebar.button("Add Position"):
        if 'positions' not in st.session_state:
            st.session_state.positions = []
        st.session_state.positions.append(Position(position_type.lower(), quantity, strike, premium))

    st.sidebar.subheader("Current Positions")
    if 'positions' in st.session_state:
        for i, pos in enumerate(st.session_state.positions):
            st.sidebar.text(f"{pos.quantity} {pos.option_type} @ {pos.strike}, premium: {pos.premium}")
            if st.sidebar.button(f"Remove Position {i+1}"):
                st.session_state.positions.pop(i)
                st.experimental_rerun()

    st.sidebar.subheader("Heatmap Parameters")
    min_spot = st.sidebar.number_input("Min Spot Price", value=S*0.8, step=1.0)
    max_spot = st.sidebar.number_input("Max Spot Price", value=S*1.2, step=1.0)
    min_vol = st.sidebar.slider("Min Volatility for Heatmap", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    max_vol = st.sidebar.slider("Max Volatility for Heatmap", min_value=0.01, max_value=1.0, value=0.3, step=0.01)

    model = BlackScholesModel(r, S, K, T, sigma)

    # Main content
    st.title("Black-Scholes Option Pricing Model")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Option Prices")
        call_price = model.price_option('call')
        put_price = model.price_option('put')
        st.metric("Call Option Price", f"${call_price:.2f}")
        st.metric("Put Option Price", f"${put_price:.2f}")

        st.subheader("Greeks")
        call_greeks = model.calculate_greeks('call')
        put_greeks = model.calculate_greeks('put')
        greeks_df = pd.DataFrame({'Call': call_greeks, 'Put': put_greeks}).T
        st.dataframe(greeks_df)

    with col2:
        st.subheader("Option Price Heatmaps")
        option_type = st.radio("Select Option Type", ["Call", "Put"])
        st.pyplot(create_heatmap(model, option_type.lower(), min_spot, max_spot, min_vol, max_vol))

    st.subheader("PNL Heatmap")
    if 'positions' in st.session_state and st.session_state.positions:
        st.pyplot(create_pnl_heatmap(model, st.session_state.positions, min_spot, max_spot, min_vol, max_vol))
    else:
        st.write("Add positions to see the PNL heatmap.")


    st.subheader("Greek Visualizations")
    greek = st.selectbox("Select Greek", ["delta", "gamma", "theta", "vega", "rho"])
    st.plotly_chart(plot_greek(model, greek, option_type), use_container_width=True)

if __name__ == "__main__":
    main()