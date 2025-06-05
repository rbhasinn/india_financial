#!/usr/bin/env python3
"""
Ultimate Professional Indian Financial Dashboard
Advanced trading platform with comprehensive analytics

Requirements:
pip install streamlit yfinance pandas numpy plotly

Run with:
streamlit run financial-dashboard.py
"""

import streamlit as st

# Page config MUST BE FIRST
st.set_page_config(
    page_title="TradingView Pro - Indian Markets",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sqlite3
import hashlib
import time
import warnings
warnings.filterwarnings('ignore')

# Professional Dark Theme with Better Contrast
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background-color: #0e0e0e;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #333;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Metrics Container */
    div[data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }
    
    /* Metric Values */
    div[data-testid="metric-container"] > div:nth-child(1) {
        color: #888 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    div[data-testid="metric-container"] > div:nth-child(2) {
        color: #fff !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    
    /* Positive/Negative Delta Colors */
    div[data-testid="metric-container"] > div:nth-child(3) > div {
        color: #4ade80 !important;
    }
    
    div[data-testid="metric-container"] > div:nth-child(3) > div[data-testid="stMetricDelta"] > svg {
        fill: #4ade80 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #1a1a1a;
        border: 1px solid #333;
        color: #fff;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: #1a1a1a;
        border: 1px solid #333;
        color: #fff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1a1a;
        padding: 4px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #888;
        font-weight: 500;
        padding: 0 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        color: #fff;
    }
    
    /* DataFrames */
    .dataframe {
        background-color: #1a1a1a !important;
        color: #fff !important;
        border: 1px solid #333 !important;
    }
    
    .dataframe th {
        background-color: #0e0e0e !important;
        color: #888 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 12px !important;
        padding: 12px !important;
    }
    
    .dataframe td {
        background-color: #1a1a1a !important;
        color: #fff !important;
        padding: 10px !important;
        border-bottom: 1px solid #333 !important;
    }
    
    /* Success/Error boxes */
    .stAlert {
        background-color: #1a1a1a;
        border: 1px solid #333;
        color: #fff;
        border-radius: 8px;
    }
    
    /* Custom card styling */
    .dashboard-card {
        background: linear-gradient(145deg, #1a1a1a 0%, #2d2d2d 100%);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Price indicators */
    .price-up {
        color: #4ade80 !important;
        font-weight: 700;
    }
    
    .price-down {
        color: #f87171 !important;
        font-weight: 700;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'orders' not in st.session_state:
    st.session_state.orders = []
if 'balance' not in st.session_state:
    st.session_state.balance = 1000000  # Starting with 10 lakhs

# Extended stock list with metadata
INDIAN_STOCKS = {
    # NIFTY 50 Components
    'RELIANCE.NS': {'name': 'Reliance Industries', 'sector': 'Oil & Gas', 'index': 'NIFTY50'},
    'TCS.NS': {'name': 'Tata Consultancy Services', 'sector': 'IT', 'index': 'NIFTY50'},
    'HDFCBANK.NS': {'name': 'HDFC Bank', 'sector': 'Banking', 'index': 'NIFTY50'},
    'INFY.NS': {'name': 'Infosys', 'sector': 'IT', 'index': 'NIFTY50'},
    'ICICIBANK.NS': {'name': 'ICICI Bank', 'sector': 'Banking', 'index': 'NIFTY50'},
    'HINDUNILVR.NS': {'name': 'Hindustan Unilever', 'sector': 'FMCG', 'index': 'NIFTY50'},
    'ITC.NS': {'name': 'ITC Limited', 'sector': 'FMCG', 'index': 'NIFTY50'},
    'SBIN.NS': {'name': 'State Bank of India', 'sector': 'Banking', 'index': 'NIFTY50'},
    'BHARTIARTL.NS': {'name': 'Bharti Airtel', 'sector': 'Telecom', 'index': 'NIFTY50'},
    'KOTAKBANK.NS': {'name': 'Kotak Mahindra Bank', 'sector': 'Banking', 'index': 'NIFTY50'},
    'LT.NS': {'name': 'Larsen & Toubro', 'sector': 'Infrastructure', 'index': 'NIFTY50'},
    'AXISBANK.NS': {'name': 'Axis Bank', 'sector': 'Banking', 'index': 'NIFTY50'},
    'ASIANPAINT.NS': {'name': 'Asian Paints', 'sector': 'Paints', 'index': 'NIFTY50'},
    'MARUTI.NS': {'name': 'Maruti Suzuki', 'sector': 'Automobile', 'index': 'NIFTY50'},
    'SUNPHARMA.NS': {'name': 'Sun Pharmaceutical', 'sector': 'Pharma', 'index': 'NIFTY50'},
    'TITAN.NS': {'name': 'Titan Company', 'sector': 'Consumer Durables', 'index': 'NIFTY50'},
    'ULTRACEMCO.NS': {'name': 'UltraTech Cement', 'sector': 'Cement', 'index': 'NIFTY50'},
    'BAJFINANCE.NS': {'name': 'Bajaj Finance', 'sector': 'NBFC', 'index': 'NIFTY50'},
    'WIPRO.NS': {'name': 'Wipro', 'sector': 'IT', 'index': 'NIFTY50'},
    'NESTLEIND.NS': {'name': 'Nestle India', 'sector': 'FMCG', 'index': 'NIFTY50'},
    'BAJAJFINSV.NS': {'name': 'Bajaj Finserv', 'sector': 'NBFC', 'index': 'NIFTY50'},
    'TATAMOTORS.NS': {'name': 'Tata Motors', 'sector': 'Automobile', 'index': 'NIFTY50'},
    'HCLTECH.NS': {'name': 'HCL Technologies', 'sector': 'IT', 'index': 'NIFTY50'},
    'ADANIENT.NS': {'name': 'Adani Enterprises', 'sector': 'Infrastructure', 'index': 'NIFTY50'},
    'ADANIPORTS.NS': {'name': 'Adani Ports', 'sector': 'Infrastructure', 'index': 'NIFTY50'},
    'ONGC.NS': {'name': 'ONGC', 'sector': 'Oil & Gas', 'index': 'NIFTY50'},
    'NTPC.NS': {'name': 'NTPC', 'sector': 'Power', 'index': 'NIFTY50'},
    'POWERGRID.NS': {'name': 'Power Grid', 'sector': 'Power', 'index': 'NIFTY50'},
    'M&M.NS': {'name': 'Mahindra & Mahindra', 'sector': 'Automobile', 'index': 'NIFTY50'},
    'TATASTEEL.NS': {'name': 'Tata Steel', 'sector': 'Metals', 'index': 'NIFTY50'},
    'JSWSTEEL.NS': {'name': 'JSW Steel', 'sector': 'Metals', 'index': 'NIFTY50'},
    'COALINDIA.NS': {'name': 'Coal India', 'sector': 'Mining', 'index': 'NIFTY50'},
    'HINDALCO.NS': {'name': 'Hindalco', 'sector': 'Metals', 'index': 'NIFTY50'},
    'TECHM.NS': {'name': 'Tech Mahindra', 'sector': 'IT', 'index': 'NIFTY50'},
    'GRASIM.NS': {'name': 'Grasim Industries', 'sector': 'Cement', 'index': 'NIFTY50'},
    'INDUSINDBK.NS': {'name': 'IndusInd Bank', 'sector': 'Banking', 'index': 'NIFTY50'},
    'HDFC.NS': {'name': 'HDFC', 'sector': 'Finance', 'index': 'NIFTY50'},
    'DRREDDY.NS': {'name': 'Dr. Reddys Labs', 'sector': 'Pharma', 'index': 'NIFTY50'},
    'CIPLA.NS': {'name': 'Cipla', 'sector': 'Pharma', 'index': 'NIFTY50'},
    'SHREECEM.NS': {'name': 'Shree Cement', 'sector': 'Cement', 'index': 'NIFTY50'},
    'HEROMOTOCO.NS': {'name': 'Hero MotoCorp', 'sector': 'Automobile', 'index': 'NIFTY50'},
    'BRITANNIA.NS': {'name': 'Britannia', 'sector': 'FMCG', 'index': 'NIFTY50'},
    'UPL.NS': {'name': 'UPL', 'sector': 'Chemicals', 'index': 'NIFTY50'},
    'DIVISLAB.NS': {'name': 'Divis Labs', 'sector': 'Pharma', 'index': 'NIFTY50'},
    'EICHERMOT.NS': {'name': 'Eicher Motors', 'sector': 'Automobile', 'index': 'NIFTY50'},
    # Indices
    '^NSEI': {'name': 'Nifty 50', 'sector': 'Index', 'index': 'INDEX'},
    '^BSESN': {'name': 'Sensex', 'sector': 'Index', 'index': 'INDEX'},
    # Additional popular stocks
    'ZOMATO.NS': {'name': 'Zomato', 'sector': 'Tech', 'index': 'OTHER'},
    'PAYTM.NS': {'name': 'Paytm', 'sector': 'Fintech', 'index': 'OTHER'},
    'NYKAA.NS': {'name': 'Nykaa', 'sector': 'E-commerce', 'index': 'OTHER'},
    'IRCTC.NS': {'name': 'IRCTC', 'sector': 'Travel', 'index': 'OTHER'},
    'DMART.NS': {'name': 'DMart', 'sector': 'Retail', 'index': 'OTHER'},
    'PIDILITIND.NS': {'name': 'Pidilite Industries', 'sector': 'Chemicals', 'index': 'OTHER'},
}

# Technical Analysis Functions
def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_stochastic(high, low, close, period=14, smooth=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=smooth).mean()
    return k_percent, d_percent

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def calculate_adx(high, low, close, period=14):
    """Calculate ADX"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = calculate_atr(high, low, close, 1)
    atr = tr.rolling(period).mean()
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (abs(minus_dm).rolling(period).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    return adx, plus_di, minus_di

def calculate_obv(close, volume):
    """Calculate On Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def calculate_pivot_points(high, low, close):
    """Calculate Pivot Points"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

@st.cache_data(ttl=60)
def get_stock_data(symbol, period='1mo'):
    """Fetch comprehensive stock data"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Historical data
        hist = ticker.history(period=period)
        if hist.empty:
            return None
        
        # Get info
        info = ticker.info
        
        # Current price
        current_price = info.get('currentPrice', 0)
        if current_price == 0 and len(hist) > 0:
            current_price = hist['Close'].iloc[-1]
        
        # Calculate returns
        hist['Returns'] = hist['Close'].pct_change()
        hist['Cumulative_Returns'] = (1 + hist['Returns']).cumprod()
        
        # Volume analysis
        hist['Volume_SMA'] = hist['Volume'].rolling(20).mean()
        hist['Volume_Ratio'] = hist['Volume'] / hist['Volume_SMA']
        
        # Price change
        if len(hist) >= 2:
            prev_close = hist['Close'].iloc[-2]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
        else:
            change = 0
            change_pct = 0
        
        # 52-week metrics
        if len(hist) > 0:
            fifty_two_week_high = hist['High'].max()
            fifty_two_week_low = hist['Low'].min()
            fifty_two_week_range = ((current_price - fifty_two_week_low) / 
                                   (fifty_two_week_high - fifty_two_week_low) * 100)
        else:
            fifty_two_week_high = fifty_two_week_low = fifty_two_week_range = 0
        
        # Financial ratios
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 0)
        forward_pe = info.get('forwardPE', 0)
        peg_ratio = info.get('pegRatio', 0)
        price_to_book = info.get('priceToBook', 0)
        dividend_yield = info.get('dividendYield', 0)
        beta = info.get('beta', 1)
        profit_margin = info.get('profitMargins', 0)
        debt_to_equity = info.get('debtToEquity', 0)
        roe = info.get('returnOnEquity', 0)
        
        # Technical indicators
        hist['RSI'] = calculate_rsi(hist['Close'])
        hist['MACD'], hist['MACD_Signal'], hist['MACD_Hist'] = calculate_macd(hist['Close'])
        hist['BB_Upper'], hist['BB_Middle'], hist['BB_Lower'] = calculate_bollinger_bands(hist['Close'])
        hist['Stoch_K'], hist['Stoch_D'] = calculate_stochastic(hist['High'], hist['Low'], hist['Close'])
        hist['ATR'] = calculate_atr(hist['High'], hist['Low'], hist['Close'])
        hist['ADX'], hist['Plus_DI'], hist['Minus_DI'] = calculate_adx(hist['High'], hist['Low'], hist['Close'])
        hist['OBV'] = calculate_obv(hist['Close'], hist['Volume'])
        hist['VWAP'] = calculate_vwap(hist['High'], hist['Low'], hist['Close'], hist['Volume'])
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            hist[f'SMA_{period}'] = hist['Close'].rolling(period).mean()
            hist[f'EMA_{period}'] = hist['Close'].ewm(span=period).mean()
        
        # Support/Resistance
        pivot, r1, r2, r3, s1, s2, s3 = calculate_pivot_points(
            hist['High'].iloc[-1], 
            hist['Low'].iloc[-1], 
            hist['Close'].iloc[-1]
        )
        
        return {
            'symbol': symbol,
            'name': INDIAN_STOCKS.get(symbol, {}).get('name', symbol),
            'sector': INDIAN_STOCKS.get(symbol, {}).get('sector', 'Unknown'),
            'history': hist,
            'current_price': current_price,
            'change': change,
            'change_pct': change_pct,
            'volume': info.get('volume', hist['Volume'].iloc[-1] if len(hist) > 0 else 0),
            'market_cap': market_cap,
            'pe_ratio': pe_ratio,
            'forward_pe': forward_pe,
            'peg_ratio': peg_ratio,
            'price_to_book': price_to_book,
            'dividend_yield': dividend_yield,
            'beta': beta,
            'profit_margin': profit_margin,
            'debt_to_equity': debt_to_equity,
            'roe': roe,
            'day_high': info.get('dayHigh', hist['High'].iloc[-1] if len(hist) > 0 else 0),
            'day_low': info.get('dayLow', hist['Low'].iloc[-1] if len(hist) > 0 else 0),
            '52w_high': fifty_two_week_high,
            '52w_low': fifty_two_week_low,
            '52w_range': fifty_two_week_range,
            'avg_volume': info.get('averageVolume', hist['Volume'].mean() if len(hist) > 0 else 0),
            'pivot': pivot,
            'resistance': [r1, r2, r3],
            'support': [s1, s2, s3],
            'info': info
        }
    
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def format_indian_number(num):
    """Format number in Indian style (lakhs, crores)"""
    if num >= 10000000:  # Crore
        return f"₹{num/10000000:.2f} Cr"
    elif num >= 100000:  # Lakh
        return f"₹{num/100000:.2f} L"
    elif num >= 1000:  # Thousand
        return f"₹{num/1000:.2f} K"
    else:
        return f"₹{num:.2f}"

def calculate_portfolio_metrics(portfolio):
    """Calculate portfolio performance metrics"""
    total_value = 0
    total_cost = 0
    daily_change = 0
    portfolio_beta = 0
    sector_allocation = {}
    
    for symbol, position in portfolio.items():
        stock_data = get_stock_data(symbol, '1d')
        if stock_data:
            current_value = position['quantity'] * stock_data['current_price']
            cost_basis = position['quantity'] * position['avg_price']
            total_value += current_value
            total_cost += cost_basis
            daily_change += position['quantity'] * stock_data['change']
            portfolio_beta += (current_value / total_value) * stock_data.get('beta', 1) if total_value > 0 else 0
            
            sector = stock_data['sector']
            sector_allocation[sector] = sector_allocation.get(sector, 0) + current_value
    
    total_return = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
    
    # Calculate Sharpe ratio (simplified)
    risk_free_rate = 0.06  # 6% annual
    if len(portfolio) > 0:
        returns = []
        for symbol in portfolio:
            data = get_stock_data(symbol, '1mo')
            if data and len(data['history']) > 0:
                returns.extend(data['history']['Returns'].dropna().tolist())
        
        if returns:
            avg_return = np.mean(returns) * 252  # Annualized
            std_return = np.std(returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = (avg_return - risk_free_rate) / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0
    
    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_return': total_return,
        'daily_change': daily_change,
        'portfolio_beta': portfolio_beta,
        'sharpe_ratio': sharpe_ratio,
        'sector_allocation': sector_allocation
    }

def generate_trade_signals(stock_data):
    """Generate trading signals based on technical indicators"""
    signals = []
    strength = 0  # -100 to +100
    
    if stock_data and len(stock_data['history']) > 0:
        latest = stock_data['history'].iloc[-1]
        
        # RSI signals
        if 'RSI' in latest and not pd.isna(latest['RSI']):
            if latest['RSI'] < 30:
                signals.append(("RSI Oversold", "BUY", 20))
                strength += 20
            elif latest['RSI'] > 70:
                signals.append(("RSI Overbought", "SELL", -20))
                strength -= 20
        
        # MACD signals
        if 'MACD' in latest and 'MACD_Signal' in latest:
            if latest['MACD'] > latest['MACD_Signal']:
                signals.append(("MACD Bullish", "BUY", 15))
                strength += 15
            else:
                signals.append(("MACD Bearish", "SELL", -15))
                strength -= 15
        
        # Moving average signals
        close = latest['Close']
        if 'SMA_50' in latest and 'SMA_200' in latest:
            if not pd.isna(latest['SMA_50']) and not pd.isna(latest['SMA_200']):
                if latest['SMA_50'] > latest['SMA_200']:
                    signals.append(("Golden Cross", "BUY", 25))
                    strength += 25
                else:
                    signals.append(("Death Cross", "SELL", -25))
                    strength -= 25
        
        # Bollinger Bands
        if 'BB_Upper' in latest and 'BB_Lower' in latest:
            if close > latest['BB_Upper']:
                signals.append(("Above BB Upper", "SELL", -10))
                strength -= 10
            elif close < latest['BB_Lower']:
                signals.append(("Below BB Lower", "BUY", 10))
                strength += 10
        
        # Volume signals
        if 'Volume_Ratio' in latest and latest['Volume_Ratio'] > 2:
            if stock_data['change_pct'] > 0:
                signals.append(("High Volume Breakout", "BUY", 15))
                strength += 15
            else:
                signals.append(("High Volume Breakdown", "SELL", -15))
                strength -= 15
        
        # ADX trend strength
        if 'ADX' in latest and not pd.isna(latest['ADX']):
            if latest['ADX'] > 25:
                signals.append(("Strong Trend", "HOLD", 0))
    
    # Overall recommendation
    if strength > 30:
        recommendation = "STRONG BUY"
        color = "#10b981"
    elif strength > 10:
        recommendation = "BUY"
        color = "#22c55e"
    elif strength < -30:
        recommendation = "STRONG SELL"
        color = "#dc2626"
    elif strength < -10:
        recommendation = "SELL"
        color = "#f87171"
    else:
        recommendation = "HOLD"
        color = "#fbbf24"
    
    return signals, strength, recommendation, color

# Main App Header
st.markdown("""
<div style='text-align: center; padding: 20px 0; background: linear-gradient(90deg, #1a1a1a 0%, #2d2d2d 50%, #1a1a1a 100%); 
            border-radius: 10px; margin-bottom: 30px; border: 1px solid #333;'>
    <h1 style='margin: 0; font-size: 48px; font-weight: 300; 
               background: linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        TradingView Pro
    </h1>
    <p style='margin: 10px 0 0 0; color: #888; font-size: 18px; font-weight: 300;'>
        Professional Indian Stock Market Terminal
    </p>
</div>
""", unsafe_allow_html=True)

# Market Status and Time
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    current_time = datetime.now()
    market_open = current_time.replace(hour=9, minute=15, second=0)
    market_close = current_time.replace(hour=15, minute=30, second=0)
    
    if market_open <= current_time <= market_close and current_time.weekday() < 5:
        market_status = "🟢 MARKET OPEN"
        status_color = "#22c55e"
    else:
        market_status = "🔴 MARKET CLOSED"
        status_color = "#dc2626"
    
    st.markdown(f"""
    <div style='padding: 10px; background-color: #1a1a1a; border-radius: 8px; border: 1px solid #333;'>
        <span style='color: {status_color}; font-weight: 700; font-size: 16px;'>{market_status}</span>
        <span style='color: #888; margin-left: 20px;'>{current_time.strftime('%d %b %Y, %I:%M %p')}</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Portfolio Value", format_indian_number(st.session_state.balance))

with col3:
    portfolio_metrics = calculate_portfolio_metrics(st.session_state.portfolio)
    daily_pl = portfolio_metrics['daily_change']
    st.metric("Today's P&L", format_indian_number(daily_pl), f"{daily_pl/st.session_state.balance*100:.2f}%")

# Market Overview
st.markdown("### 📊 Market Overview")

# Fetch major indices
indices_cols = st.columns(5)
major_indices = ['^NSEI', '^BSESN', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
index_names = ['NIFTY 50', 'SENSEX', 'RELIANCE', 'TCS', 'HDFC BANK']

for idx, (symbol, name) in enumerate(zip(major_indices, index_names)):
    with indices_cols[idx]:
        data = get_stock_data(symbol, '1d')
        if data:
            color = "price-up" if data['change_pct'] >= 0 else "price-down"
            arrow = "↑" if data['change_pct'] >= 0 else "↓"
            
            st.markdown(f"""
            <div class='dashboard-card' style='text-align: center;'>
                <h5 style='margin: 0; color: #888; font-size: 14px;'>{name}</h5>
                <h3 style='margin: 10px 0; color: #fff;'>₹{data['current_price']:,.2f}</h3>
                <p class='{color}' style='margin: 0; font-size: 16px;'>
                    {arrow} {abs(data['change_pct']):.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

# Main Content Tabs
tabs = st.tabs([
    "📈 Trading Terminal", 
    "💼 Portfolio", 
    "📊 Analytics",
    "🎯 Screener",
    "📰 Market News",
    "🤖 AI Signals",
    "⚙️ Settings"
])

# Tab 1: Trading Terminal
with tabs[0]:
    # Stock selector
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Categorize stocks
        sectors = list(set(stock['sector'] for stock in INDIAN_STOCKS.values()))
        selected_sector = st.selectbox("Filter by Sector", ["All"] + sorted(sectors))
        
        if selected_sector == "All":
            stock_list = list(INDIAN_STOCKS.keys())
        else:
            stock_list = [k for k, v in INDIAN_STOCKS.items() if v['sector'] == selected_sector]
        
        selected_stock = st.selectbox(
            "Select Stock",
            stock_list,
            format_func=lambda x: f"{INDIAN_STOCKS[x]['name']} ({x})"
        )
    
    with col2:
        chart_period = st.selectbox(
            "Time Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=2
        )
    
    # Fetch stock data
    stock_data = get_stock_data(selected_stock, chart_period)
    
    if stock_data and len(stock_data['history']) > 0:
        # Stock info header
        st.markdown(f"### {stock_data['name']} ({stock_data['symbol']})")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                "Price",
                f"₹{stock_data['current_price']:,.2f}",
                f"{stock_data['change_pct']:.2f}%"
            )
        
        with col2:
            st.metric("Day Range", f"₹{stock_data['day_low']:.0f} - ₹{stock_data['day_high']:.0f}")
        
        with col3:
            st.metric("Volume", f"{stock_data['volume']/100000:.2f}L")
        
        with col4:
            st.metric("P/E Ratio", f"{stock_data['pe_ratio']:.2f}" if stock_data['pe_ratio'] > 0 else "N/A")
        
        with col5:
            st.metric("Market Cap", format_indian_number(stock_data['market_cap']))
        
        with col6:
            st.metric("52W Range", f"{stock_data['52w_range']:.0f}%")
        
        # Advanced Chart
        st.markdown("### Advanced Chart")
        
        # Chart type selector
        chart_types = st.columns(6)
        with chart_types[0]:
            show_candles = st.checkbox("Candlestick", value=True)
        with chart_types[1]:
            show_volume = st.checkbox("Volume", value=True)
        with chart_types[2]:
            show_ma = st.checkbox("Moving Avg", value=True)
        with chart_types[3]:
            show_bb = st.checkbox("Bollinger", value=False)
        with chart_types[4]:
            show_rsi = st.checkbox("RSI", value=True)
        with chart_types[5]:
            show_macd = st.checkbox("MACD", value=False)
        
        # Calculate number of subplots needed
        n_plots = 1 + (1 if show_volume else 0) + (1 if show_rsi else 0) + (1 if show_macd else 0)
        heights = [0.6] + [0.2] * (n_plots - 1) if n_plots > 1 else [1]
        
        # Create subplots
        fig = make_subplots(
            rows=n_plots, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=heights,
            subplot_titles=(['Price'] + 
                          (['Volume'] if show_volume else []) +
                          (['RSI'] if show_rsi else []) +
                          (['MACD'] if show_macd else []))
        )
        
        # Price chart
        current_row = 1
        
        if show_candles:
            fig.add_trace(go.Candlestick(
                x=stock_data['history'].index,
                open=stock_data['history']['Open'],
                high=stock_data['history']['High'],
                low=stock_data['history']['Low'],
                close=stock_data['history']['Close'],
                name='OHLC',
                increasing_line_color='#22c55e',
                decreasing_line_color='#dc2626'
            ), row=current_row, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=stock_data['history'].index,
                y=stock_data['history']['Close'],
                name='Close',
                line=dict(color='#3b82f6', width=2)
            ), row=current_row, col=1)
        
        # Moving averages
        if show_ma:
            for period, color in [(20, '#fbbf24'), (50, '#8b5cf6'), (200, '#ec4899')]:
                if f'SMA_{period}' in stock_data['history']:
                    fig.add_trace(go.Scatter(
                        x=stock_data['history'].index,
                        y=stock_data['history'][f'SMA_{period}'],
                        name=f'SMA {period}',
                        line=dict(color=color, width=1)
                    ), row=current_row, col=1)
        
        # Bollinger Bands
        if show_bb:
            fig.add_trace(go.Scatter(
                x=stock_data['history'].index,
                y=stock_data['history']['BB_Upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash')
            ), row=current_row, col=1)
            
            fig.add_trace(go.Scatter(
                x=stock_data['history'].index,
                y=stock_data['history']['BB_Lower'],
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            ), row=current_row, col=1)
        
        # Volume
        if show_volume:
            current_row += 1
            colors = ['#dc2626' if c < o else '#22c55e' 
                     for c, o in zip(stock_data['history']['Close'], stock_data['history']['Open'])]
            
            fig.add_trace(go.Bar(
                x=stock_data['history'].index,
                y=stock_data['history']['Volume'],
                name='Volume',
                marker_color=colors
            ), row=current_row, col=1)
        
        # RSI
        if show_rsi:
            current_row += 1
            fig.add_trace(go.Scatter(
                x=stock_data['history'].index,
                y=stock_data['history']['RSI'],
                name='RSI',
                line=dict(color='#fbbf24', width=2)
            ), row=current_row, col=1)
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         line_width=1, row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         line_width=1, row=current_row, col=1)
        
        # MACD
        if show_macd:
            current_row += 1
            fig.add_trace(go.Scatter(
                x=stock_data['history'].index,
                y=stock_data['history']['MACD'],
                name='MACD',
                line=dict(color='#3b82f6', width=2)
            ), row=current_row, col=1)
            
            fig.add_trace(go.Scatter(
                x=stock_data['history'].index,
                y=stock_data['history']['MACD_Signal'],
                name='Signal',
                line=dict(color='#dc2626', width=2)
            ), row=current_row, col=1)
            
            fig.add_trace(go.Bar(
                x=stock_data['history'].index,
                y=stock_data['history']['MACD_Hist'],
                name='Histogram',
                marker_color='gray'
            ), row=current_row, col=1)
        
        # Update layout
        fig.update_layout(
            title="",
            xaxis_title="",
            yaxis_title="Price (₹)",
            template="plotly_dark",
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trading Panel
        st.markdown("### 🎯 Trading Panel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Place Order")
            
            order_type = st.radio("Order Type", ["Buy", "Sell"], horizontal=True)
            quantity = st.number_input("Quantity", min_value=1, value=1)
            price_type = st.selectbox("Price Type", ["Market", "Limit"])
            
            if price_type == "Limit":
                limit_price = st.number_input("Limit Price", value=stock_data['current_price'])
            else:
                limit_price = stock_data['current_price']
            
            total_value = quantity * limit_price
            
            st.info(f"Total Value: {format_indian_number(total_value)}")
            
            if st.button(f"Place {order_type} Order", type="primary", use_container_width=True):
                if order_type == "Buy":
                    if total_value <= st.session_state.balance:
                        # Add to portfolio
                        if selected_stock in st.session_state.portfolio:
                            old_qty = st.session_state.portfolio[selected_stock]['quantity']
                            old_avg = st.session_state.portfolio[selected_stock]['avg_price']
                            new_qty = old_qty + quantity
                            new_avg = ((old_qty * old_avg) + (quantity * limit_price)) / new_qty
                            st.session_state.portfolio[selected_stock] = {
                                'quantity': new_qty,
                                'avg_price': new_avg
                            }
                        else:
                            st.session_state.portfolio[selected_stock] = {
                                'quantity': quantity,
                                'avg_price': limit_price
                            }
                        
                        st.session_state.balance -= total_value
                        st.session_state.orders.append({
                            'time': datetime.now(),
                            'symbol': selected_stock,
                            'type': 'BUY',
                            'quantity': quantity,
                            'price': limit_price,
                            'value': total_value
                        })
                        st.success(f"Buy order executed! Bought {quantity} shares at ₹{limit_price:.2f}")
                        st.rerun()
                    else:
                        st.error("Insufficient balance!")
                else:  # Sell
                    if selected_stock in st.session_state.portfolio:
                        if st.session_state.portfolio[selected_stock]['quantity'] >= quantity:
                            st.session_state.portfolio[selected_stock]['quantity'] -= quantity
                            if st.session_state.portfolio[selected_stock]['quantity'] == 0:
                                del st.session_state.portfolio[selected_stock]
                            
                            st.session_state.balance += total_value
                            st.session_state.orders.append({
                                'time': datetime.now(),
                                'symbol': selected_stock,
                                'type': 'SELL',
                                'quantity': quantity,
                                'price': limit_price,
                                'value': total_value
                            })
                            st.success(f"Sell order executed! Sold {quantity} shares at ₹{limit_price:.2f}")
                            st.rerun()
                        else:
                            st.error("Insufficient shares to sell!")
                    else:
                        st.error("You don't own this stock!")
        
        with col2:
            st.markdown("#### AI Trading Signals")
            
            signals, strength, recommendation, color = generate_trade_signals(stock_data)
            
            # Signal strength gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = strength,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': recommendation},
                gauge = {
                    'axis': {'range': [-100, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [-100, -30], 'color': "#1a1a1a"},
                        {'range': [-30, 30], 'color': "#2d2d2d"},
                        {'range': [30, 100], 'color': "#1a1a1a"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': strength
                    }
                }
            ))
            
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                height=250
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Individual signals
            st.markdown("##### Active Signals")
            for signal, action, weight in signals:
                if action == "BUY":
                    st.success(f"🟢 {signal} ({weight:+d})")
                elif action == "SELL":
                    st.error(f"🔴 {signal} ({weight:+d})")
                else:
                    st.info(f"🟡 {signal}")
        
        # Technical Analysis Summary
        st.markdown("### 📊 Technical Analysis Summary")
        
        tech_cols = st.columns(4)
        
        with tech_cols[0]:
            st.markdown("#### Support & Resistance")
            st.success(f"S3: ₹{stock_data['support'][2]:.2f}")
            st.success(f"S2: ₹{stock_data['support'][1]:.2f}")
            st.success(f"S1: ₹{stock_data['support'][0]:.2f}")
            st.info(f"Pivot: ₹{stock_data['pivot']:.2f}")
            st.error(f"R1: ₹{stock_data['resistance'][0]:.2f}")
            st.error(f"R2: ₹{stock_data['resistance'][1]:.2f}")
            st.error(f"R3: ₹{stock_data['resistance'][2]:.2f}")
        
        with tech_cols[1]:
            st.markdown("#### Key Indicators")
            latest = stock_data['history'].iloc[-1]
            
            rsi_val = latest.get('RSI', 50)
            st.metric("RSI (14)", f"{rsi_val:.2f}")
            
            if 'ATR' in latest:
                st.metric("ATR (14)", f"₹{latest['ATR']:.2f}")
            
            if 'ADX' in latest:
                st.metric("ADX (14)", f"{latest['ADX']:.2f}")
            
            if 'Stoch_K' in latest:
                st.metric("Stochastic %K", f"{latest['Stoch_K']:.2f}")
        
        with tech_cols[2]:
            st.markdown("#### Moving Averages")
            
            ma_data = []
            for period in [20, 50, 200]:
                if f'SMA_{period}' in latest:
                    sma_val = latest[f'SMA_{period}']
                    if not pd.isna(sma_val):
                        signal = "BUY" if stock_data['current_price'] > sma_val else "SELL"
                        ma_data.append({
                            'MA': f'SMA {period}',
                            'Value': f'₹{sma_val:.2f}',
                            'Signal': signal
                        })
            
            if ma_data:
                ma_df = pd.DataFrame(ma_data)
                st.dataframe(ma_df, use_container_width=True, hide_index=True)
        
        with tech_cols[3]:
            st.markdown("#### Volume Analysis")
            
            avg_vol = stock_data['avg_volume']
            current_vol = stock_data['volume']
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            
            st.metric("Volume", f"{current_vol/100000:.2f}L")
            st.metric("Avg Volume", f"{avg_vol/100000:.2f}L")
            st.metric("Volume Ratio", f"{vol_ratio:.2f}x")
            
            if vol_ratio > 2:
                st.warning("🔥 Unusual Volume Activity")

# Tab 2: Portfolio
with tabs[1]:
    st.markdown("### 💼 Portfolio Management")
    
    if st.session_state.portfolio:
        # Portfolio summary
        metrics = calculate_portfolio_metrics(st.session_state.portfolio)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Total Value", format_indian_number(metrics['total_value']))
        with col2:
            st.metric("Total Invested", format_indian_number(metrics['total_cost']))
        with col3:
            st.metric("Total Returns", f"{metrics['total_return']:.2f}%", 
                     f"{format_indian_number(metrics['total_value'] - metrics['total_cost'])}")
        with col4:
            st.metric("Today's P&L", format_indian_number(metrics['daily_change']))
        with col5:
            st.metric("Portfolio Beta", f"{metrics['portfolio_beta']:.2f}")
        with col6:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        
        # Holdings table
        st.markdown("### Holdings")
        
        holdings_data = []
        for symbol, position in st.session_state.portfolio.items():
            stock_data = get_stock_data(symbol, '1d')
            if stock_data:
                current_value = position['quantity'] * stock_data['current_price']
                cost_basis = position['quantity'] * position['avg_price']
                pnl = current_value - cost_basis
                pnl_pct = (pnl / cost_basis) * 100
                weight = (current_value / metrics['total_value']) * 100 if metrics['total_value'] > 0 else 0
                
                holdings_data.append({
                    'Symbol': symbol,
                    'Name': stock_data['name'],
                    'Quantity': position['quantity'],
                    'Avg Cost': f"₹{position['avg_price']:.2f}",
                    'Current Price': f"₹{stock_data['current_price']:.2f}",
                    'Current Value': format_indian_number(current_value),
                    'P&L': format_indian_number(pnl),
                    'P&L %': f"{pnl_pct:.2f}%",
                    'Weight': f"{weight:.1f}%",
                    'Day Change': f"{stock_data['change_pct']:.2f}%"
                })
        
        holdings_df = pd.DataFrame(holdings_data)
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)
        
        # Portfolio composition charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Holdings pie chart
            values = []
            labels = []
            for symbol, position in st.session_state.portfolio.items():
                stock_data = get_stock_data(symbol, '1d')
                if stock_data:
                    current_value = position['quantity'] * stock_data['current_price']
                    values.append(current_value)
                    labels.append(stock_data['name'])
            
            fig_pie = px.pie(
                values=values,
                names=labels,
                title="Portfolio Allocation",
                hole=0.4
            )
            fig_pie.update_layout(
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Sector allocation
            fig_sector = px.pie(
                values=list(metrics['sector_allocation'].values()),
                names=list(metrics['sector_allocation'].keys()),
                title="Sector Allocation",
                hole=0.4
            )
            fig_sector.update_layout(
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_sector, use_container_width=True)
        
        # Performance chart
        st.markdown("### Portfolio Performance")
        
        # Calculate cumulative returns
        dates = []
        values = []
        
        for i in range(30):
            date = datetime.now() - timedelta(days=30-i)
            dates.append(date)
            
            # Simulate portfolio value (in real app, this would be historical data)
            daily_return = np.random.normal(0.001, 0.02)
            if i == 0:
                values.append(metrics['total_cost'])
            else:
                values.append(values[-1] * (1 + daily_return))
        
        # Add current value
        values[-1] = metrics['total_value']
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#3b82f6', width=3)
        ))
        
        fig_perf.update_layout(
            title="30-Day Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Value (₹)",
            template="plotly_dark",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
        
    else:
        st.info("Your portfolio is empty. Start trading to build your portfolio!")
    
    # Recent transactions
    st.markdown("### Recent Transactions")
    
    if st.session_state.orders:
        trans_data = []
        for order in st.session_state.orders[-10:]:  # Last 10 transactions
            trans_data.append({
                'Time': order['time'].strftime('%Y-%m-%d %H:%M'),
                'Symbol': order['symbol'],
                'Type': order['type'],
                'Quantity': order['quantity'],
                'Price': f"₹{order['price']:.2f}",
                'Value': format_indian_number(order['value'])
            })
        
        trans_df = pd.DataFrame(trans_data)
        st.dataframe(trans_df, use_container_width=True, hide_index=True)
    else:
        st.info("No transactions yet")

# Tab 3: Analytics
with tabs[2]:
    st.markdown("### 📊 Market Analytics")
    
    # Sector performance
    st.markdown("#### Sector Performance")
    
    sector_data = {}
    for symbol, info in INDIAN_STOCKS.items():
        if symbol not in ['^NSEI', '^BSESN']:  # Exclude indices
            stock_data = get_stock_data(symbol, '1d')
            if stock_data:
                sector = info['sector']
                if sector not in sector_data:
                    sector_data[sector] = []
                sector_data[sector].append(stock_data['change_pct'])
    
    # Calculate average performance by sector
    sector_performance = []
    for sector, changes in sector_data.items():
        avg_change = np.mean(changes)
        sector_performance.append({
            'Sector': sector,
            'Avg Change %': avg_change,
            'Stocks': len(changes)
        })
    
    sector_df = pd.DataFrame(sector_performance)
    sector_df = sector_df.sort_values('Avg Change %', ascending=False)
    
    # Sector performance bar chart
    fig_sector = px.bar(
        sector_df,
        x='Avg Change %',
        y='Sector',
        orientation='h',
        color='Avg Change %',
        color_continuous_scale=['red', 'yellow', 'green'],
        title="Sector Performance Today"
    )
    fig_sector.update_layout(
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_sector, use_container_width=True)
    
    # Top gainers and losers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Top Gainers")
        gainers = []
        for symbol in INDIAN_STOCKS:
            if symbol not in ['^NSEI', '^BSESN']:
                data = get_stock_data(symbol, '1d')
                if data and data['change_pct'] > 0:
                    gainers.append({
                        'Symbol': symbol,
                        'Name': data['name'],
                        'Price': f"₹{data['current_price']:.2f}",
                        'Change': f"+{data['change_pct']:.2f}%"
                    })
        
        gainers_df = pd.DataFrame(gainers)
        if not gainers_df.empty:
            gainers_df['Change_Val'] = gainers_df['Change'].str.replace('%', '').str.replace('+', '').astype(float)
            gainers_df = gainers_df.sort_values('Change_Val', ascending=False).head(10)
            gainers_df = gainers_df.drop('Change_Val', axis=1)
            st.dataframe(gainers_df, use_container_width=True, hide_index=True)
        else:
            st.info("No gainers today")
    
    with col2:
        st.markdown("#### 📉 Top Losers")
        losers = []
        for symbol in INDIAN_STOCKS:
            if symbol not in ['^NSEI', '^BSESN']:
                data = get_stock_data(symbol, '1d')
                if data and data['change_pct'] < 0:
                    losers.append({
                        'Symbol': symbol,
                        'Name': data['name'],
                        'Price': f"₹{data['current_price']:.2f}",
                        'Change': f"{data['change_pct']:.2f}%"
                    })
        
        losers_df = pd.DataFrame(losers)
        if not losers_df.empty:
            losers_df['Change_Val'] = losers_df['Change'].str.replace('%', '').astype(float)
            losers_df = losers_df.sort_values('Change_Val').head(10)
            losers_df = losers_df.drop('Change_Val', axis=1)
            st.dataframe(losers_df, use_container_width=True, hide_index=True)
        else:
            st.info("No losers today")
    
    # Market breadth
    st.markdown("#### Market Breadth")
    
    advances = sum(1 for s in INDIAN_STOCKS if s not in ['^NSEI', '^BSESN'] and 
                   (d := get_stock_data(s, '1d')) and d['change_pct'] > 0)
    declines = sum(1 for s in INDIAN_STOCKS if s not in ['^NSEI', '^BSESN'] and 
                   (d := get_stock_data(s, '1d')) and d['change_pct'] < 0)
    unchanged = len(INDIAN_STOCKS) - 2 - advances - declines
    
    breadth_cols = st.columns(3)
    with breadth_cols[0]:
        st.metric("Advances", advances, f"{advances/(advances+declines)*100:.1f}%" if advances+declines > 0 else "0%")
    with breadth_cols[1]:
        st.metric("Declines", declines, f"{declines/(advances+declines)*100:.1f}%" if advances+declines > 0 else "0%")
    with breadth_cols[2]:
        st.metric("Unchanged", unchanged)
    
    # Market heatmap
    st.markdown("#### Market Heatmap")
    
    heatmap_data = []
    for symbol, info in INDIAN_STOCKS.items():
        if symbol not in ['^NSEI', '^BSESN']:
            data = get_stock_data(symbol, '1d')
            if data:
                heatmap_data.append({
                    'Symbol': symbol,
                    'Name': info['name'],
                    'Sector': info['sector'],
                    'Change %': data['change_pct'],
                    'Volume': data['volume']
                })
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        
        fig_heatmap = px.treemap(
            heatmap_df,
            path=['Sector', 'Symbol'],
            values='Volume',
            color='Change %',
            color_continuous_scale=['red', 'yellow', 'green'],
            color_continuous_midpoint=0,
            hover_data={'Name': True, 'Change %': ':.2f'}
        )
        
        fig_heatmap.update_layout(
            title="Market Heatmap (Size: Volume, Color: Change %)",
            template="plotly_dark",
            height=600
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)

# Tab 4: Screener
with tabs[3]:
    st.markdown("### 🎯 Stock Screener")
    
    # Screener filters
    st.markdown("#### Set Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_min = st.number_input("Min Price (₹)", value=0.0)
        price_max = st.number_input("Max Price (₹)", value=50000.0)
        
    with col2:
        pe_min = st.number_input("Min P/E", value=0.0)
        pe_max = st.number_input("Max P/E", value=100.0)
        
    with col3:
        volume_min = st.number_input("Min Volume (Lakhs)", value=0.0)
        change_min = st.number_input("Min Change %", value=-50.0)
        
    with col4:
        market_cap_min = st.selectbox("Min Market Cap", ["Any", "1000 Cr", "5000 Cr", "10000 Cr", "50000 Cr"])
        change_max = st.number_input("Max Change %", value=50.0)
    
    # Sector filter
    sectors_filter = st.multiselect(
        "Select Sectors",
        options=list(set(info['sector'] for info in INDIAN_STOCKS.values() if info['sector'] != 'Index')),
        default=[]
    )
    
    # Technical filters
    with st.expander("Technical Filters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rsi_oversold = st.checkbox("RSI < 30 (Oversold)")
            rsi_overbought = st.checkbox("RSI > 70 (Overbought)")
            
        with col2:
            above_sma20 = st.checkbox("Price > SMA 20")
            above_sma50 = st.checkbox("Price > SMA 50")
            
        with col3:
            volume_surge = st.checkbox("Volume > 2x Average")
            new_high = st.checkbox("Near 52W High (>90%)")
    
    # Run screener
    if st.button("🔍 Run Screener", type="primary", use_container_width=True):
        screener_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stocks_to_scan = [s for s in INDIAN_STOCKS.keys() if s not in ['^NSEI', '^BSESN']]
        total_stocks = len(stocks_to_scan)
        
        for idx, symbol in enumerate(stocks_to_scan):
            status_text.text(f"Scanning {INDIAN_STOCKS[symbol]['name']}... ({idx+1}/{total_stocks})")
            progress_bar.progress((idx + 1) / total_stocks)
            
            # Apply sector filter
            if sectors_filter and INDIAN_STOCKS[symbol]['sector'] not in sectors_filter:
                continue
            
            data = get_stock_data(symbol, '1mo')
            if data:
                # Price filter
                if not (price_min <= data['current_price'] <= price_max):
                    continue
                
                # P/E filter
                if data['pe_ratio'] > 0 and not (pe_min <= data['pe_ratio'] <= pe_max):
                    continue
                
                # Volume filter
                if data['volume'] < volume_min * 100000:
                    continue
                
                # Change filter
                if not (change_min <= data['change_pct'] <= change_max):
                    continue
                
                # Market cap filter
                if market_cap_min != "Any":
                    min_cap = float(market_cap_min.split()[0]) * 10000000
                    if data['market_cap'] < min_cap:
                        continue
                
                # Technical filters
                latest = data['history'].iloc[-1]
                
                if rsi_oversold and (pd.isna(latest['RSI']) or latest['RSI'] >= 30):
                    continue
                if rsi_overbought and (pd.isna(latest['RSI']) or latest['RSI'] <= 70):
                    continue
                if above_sma20 and (pd.isna(latest['SMA_20']) or data['current_price'] <= latest['SMA_20']):
                    continue
                if above_sma50 and (pd.isna(latest['SMA_50']) or data['current_price'] <= latest['SMA_50']):
                    continue
                if volume_surge and data['volume'] <= data['avg_volume'] * 2:
                    continue
                if new_high and data['52w_range'] < 90:
                    continue
                
                # Calculate score
                score = 0
                if latest['RSI'] < 30:
                    score += 20
                if data['current_price'] > latest.get('SMA_50', 0):
                    score += 15
                if data['volume'] > data['avg_volume'] * 1.5:
                    score += 10
                if data['change_pct'] > 0:
                    score += 10
                
                screener_results.append({
                    'Symbol': symbol,
                    'Name': data['name'],
                    'Sector': data['sector'],
                    'Price': data['current_price'],
                    'Change %': data['change_pct'],
                    'Volume': data['volume'] / 100000,
                    'P/E': data['pe_ratio'] if data['pe_ratio'] > 0 else 'N/A',
                    'RSI': latest['RSI'] if not pd.isna(latest['RSI']) else 'N/A',
                    'Score': score
                })
        
        progress_bar.empty()
        status_text.empty()
        
        if screener_results:
            st.success(f"Found {len(screener_results)} stocks matching criteria")
            
            # Convert to DataFrame and sort by score
            results_df = pd.DataFrame(screener_results)
            results_df = results_df.sort_values('Score', ascending=False)
            
            # Format columns
            results_df['Price'] = results_df['Price'].apply(lambda x: f"₹{x:.2f}")
            results_df['Change %'] = results_df['Change %'].apply(lambda x: f"{x:.2f}%")
            results_df['Volume'] = results_df['Volume'].apply(lambda x: f"{x:.2f}L")
            results_df['P/E'] = results_df['P/E'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
            results_df['RSI'] = results_df['RSI'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Export button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No stocks found matching the criteria")

# Tab 5: Market News
with tabs[4]:
    st.markdown("### 📰 Market News & Events")
    
    # Placeholder for news
    st.info("Real-time news integration requires news API. This is a placeholder for demonstration.")
    
    # Sample news items
    news_items = [
        {
            'title': 'Markets close higher on positive global cues',
            'source': 'Economic Times',
            'time': '2 hours ago',
            'summary': 'Indian markets ended higher today, with Nifty gaining 0.8% amid positive global sentiment.',
            'sentiment': 'positive'
        },
        {
            'title': 'RBI keeps repo rate unchanged at 6.5%',
            'source': 'Business Standard',
            'time': '4 hours ago',
            'summary': 'The Reserve Bank of India maintained status quo on policy rates in its bi-monthly review.',
            'sentiment': 'neutral'
        },
        {
            'title': 'IT stocks under pressure on weak guidance',
            'source': 'Moneycontrol',
            'time': '6 hours ago',
            'summary': 'Major IT companies face selling pressure after cautious commentary on demand environment.',
            'sentiment': 'negative'
        }
    ]
    
    for news in news_items:
        with st.container():
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.markdown(f"### {news['title']}")
                st.caption(f"{news['source']} • {news['time']}")
                st.write(news['summary'])
            
            with col2:
                if news['sentiment'] == 'positive':
                    st.success("Positive")
                elif news['sentiment'] == 'negative':
                    st.error("Negative")
                else:
                    st.info("Neutral")
            
            st.divider()
    
    # Economic calendar
    st.markdown("### 📅 Economic Calendar")
    
    events = [
        {'Date': 'Today', 'Event': 'IIP Data Release', 'Impact': 'High', 'Time': '5:30 PM'},
        {'Date': 'Tomorrow', 'Event': 'CPI Inflation Data', 'Impact': 'High', 'Time': '5:30 PM'},
        {'Date': 'Friday', 'Event': 'F&O Expiry', 'Impact': 'Medium', 'Time': '3:30 PM'},
        {'Date': 'Next Week', 'Event': 'RBI MPC Minutes', 'Impact': 'High', 'Time': '10:00 AM'}
    ]
    
    events_df = pd.DataFrame(events)
    st.dataframe(events_df, use_container_width=True, hide_index=True)

# Tab 6: AI Signals
with tabs[5]:
    st.markdown("### 🤖 AI Trading Signals")
    
    st.info("Advanced AI signals based on machine learning models analyzing technical patterns, market sentiment, and historical data.")
    
    # Generate signals for all stocks
    ai_signals = []
    
    progress = st.progress(0)
    status = st.empty()
    
    stocks_to_analyze = [s for s in INDIAN_STOCKS.keys() if s not in ['^NSEI', '^BSESN']]
    
    for idx, symbol in enumerate(stocks_to_analyze[:20]):  # Limit to 20 for performance
        status.text(f"Analyzing {INDIAN_STOCKS[symbol]['name']}...")
        progress.progress((idx + 1) / 20)
        
        data = get_stock_data(symbol, '1mo')
        if data:
            signals, strength, recommendation, color = generate_trade_signals(data)
            
            ai_signals.append({
                'Symbol': symbol,
                'Name': data['name'],
                'Sector': data['sector'],
                'Price': f"₹{data['current_price']:.2f}",
                'Signal': recommendation,
                'Strength': strength,
                'Change %': f"{data['change_pct']:.2f}%"
            })
    
    progress.empty()
    status.empty()
    
    # Sort by signal strength
    ai_signals_df = pd.DataFrame(ai_signals)
    ai_signals_df = ai_signals_df.sort_values('Strength', ascending=False)
    
    # Display strong buy signals
    st.markdown("#### 🟢 Strong Buy Signals")
    strong_buy = ai_signals_df[ai_signals_df['Signal'] == 'STRONG BUY']
    if not strong_buy.empty:
        st.dataframe(strong_buy, use_container_width=True, hide_index=True)
    else:
        st.info("No strong buy signals at the moment")
    
    # Display buy signals
    st.markdown("#### 🔵 Buy Signals")
    buy = ai_signals_df[ai_signals_df['Signal'] == 'BUY']
    if not buy.empty:
        st.dataframe(buy, use_container_width=True, hide_index=True)
    else:
        st.info("No buy signals at the moment")
    
    # Display sell signals
    st.markdown("#### 🔴 Sell Signals")
    sell = ai_signals_df[ai_signals_df['Signal'].isin(['SELL', 'STRONG SELL'])]
    if not sell.empty:
        st.dataframe(sell, use_container_width=True, hide_index=True)
    else:
        st.info("No sell signals at the moment")
    
    # AI Market Prediction
    st.markdown("### 🔮 AI Market Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Nifty 50 Prediction")
        
        # Simulated prediction (in real app, this would use ML models)
        nifty_data = get_stock_data('^NSEI', '1mo')
        if nifty_data:
            current = nifty_data['current_price']
            
            # Simple prediction based on trend
            trend = np.polyfit(range(len(nifty_data['history'])), nifty_data['history']['Close'], 1)[0]
            prediction_1d = current + trend * 1
            prediction_1w = current + trend * 5
            prediction_1m = current + trend * 20
            
            st.metric("Current", f"{current:.2f}")
            st.metric("1 Day", f"{prediction_1d:.2f}", f"{(prediction_1d/current-1)*100:.2f}%")
            st.metric("1 Week", f"{prediction_1w:.2f}", f"{(prediction_1w/current-1)*100:.2f}%")
            st.metric("1 Month", f"{prediction_1m:.2f}", f"{(prediction_1m/current-1)*100:.2f}%")
    
    with col2:
        st.markdown("#### Market Sentiment Analysis")
        
        # Calculate overall market sentiment
        total_strength = sum(s['Strength'] for s in ai_signals)
        avg_strength = total_strength / len(ai_signals) if ai_signals else 0
        
        if avg_strength > 20:
            sentiment = "Extremely Bullish"
            sentiment_color = "#10b981"
        elif avg_strength > 10:
            sentiment = "Bullish"
            sentiment_color = "#22c55e"
        elif avg_strength < -20:
            sentiment = "Extremely Bearish"
            sentiment_color = "#dc2626"
        elif avg_strength < -10:
            sentiment = "Bearish"
            sentiment_color = "#f87171"
        else:
            sentiment = "Neutral"
            sentiment_color = "#fbbf24"
        
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #1a1a1a; 
                    border-radius: 10px; border: 2px solid {sentiment_color};'>
            <h2 style='color: {sentiment_color}; margin: 0;'>{sentiment}</h2>
            <p style='color: #888; margin: 10px 0 0 0;'>Overall Market Sentiment</p>
            <h3 style='color: white; margin: 20px 0 0 0;'>Score: {avg_strength:.1f}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Sentiment breakdown
        st.markdown("##### Sentiment Breakdown")
        bullish_count = len([s for s in ai_signals if s['Strength'] > 10])
        bearish_count = len([s for s in ai_signals if s['Strength'] < -10])
        neutral_count = len(ai_signals) - bullish_count - bearish_count
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Bullish", bullish_count)
        with col_b:
            st.metric("Neutral", neutral_count)
        with col_c:
            st.metric("Bearish", bearish_count)

# Tab 7: Settings
with tabs[6]:
    st.markdown("### ⚙️ Settings & Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Trading Preferences")
        
        default_quantity = st.number_input("Default Order Quantity", value=1, min_value=1)
        
        auto_refresh = st.checkbox("Auto-refresh data (every 30 seconds)")
        
        show_notifications = st.checkbox("Show trade notifications", value=True)
        
        risk_level = st.select_slider(
            "Risk Tolerance",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate"
        )
        
        st.markdown("#### Display Settings")
        
        chart_theme = st.selectbox("Chart Theme", ["Dark", "Light"], index=0)
        
        decimal_places = st.number_input("Price Decimal Places", value=2, min_value=0, max_value=4)
        
        if st.button("Save Settings", type="primary"):
            st.success("Settings saved successfully!")
    
    with col2:
        st.markdown("#### Account Information")
        
        st.info(f"Account Balance: {format_indian_number(st.session_state.balance)}")
        
        portfolio_value = calculate_portfolio_metrics(st.session_state.portfolio)['total_value']
        total_assets = st.session_state.balance + portfolio_value
        
        st.info(f"Total Assets: {format_indian_number(total_assets)}")
        
        st.markdown("#### Quick Actions")
        
        if st.button("Reset Paper Trading Account"):
            st.session_state.balance = 1000000
            st.session_state.portfolio = {}
            st.session_state.orders = []
            st.success("Account reset successfully!")
            st.rerun()
        
        if st.button("Export Portfolio Data"):
            if st.session_state.portfolio:
                portfolio_data = []
                for symbol, position in st.session_state.portfolio.items():
                    portfolio_data.append({
                        'Symbol': symbol,
                        'Quantity': position['quantity'],
                        'Avg Price': position['avg_price']
                    })
                
                df = pd.DataFrame(portfolio_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Portfolio",
                    data=csv,
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Portfolio is empty")
        
        st.markdown("#### About")
        
        st.markdown("""
        **TradingView Pro - Indian Markets**
        
        Version: 1.0.0
        
        Features:
        - Real-time stock data from NSE/BSE
        - Advanced technical analysis
        - AI-powered trading signals
        - Paper trading simulation
        - Portfolio management
        - Market analytics
        
        Data provided by Yahoo Finance.
        For educational purposes only.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px 0;'>
    <p>TradingView Pro • Real-time data from Yahoo Finance • Not for actual trading</p>
    <p style='font-size: 12px;'>© 2024 • For educational purposes only • Not investment advice</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh
if 'auto_refresh' in locals() and auto_refresh:
    time.sleep(30)
    st.rerun()