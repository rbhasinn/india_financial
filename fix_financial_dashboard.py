#!/usr/bin/env python3
"""
Professional Indian Financial Dashboard
Real-time market data, advanced analytics, and portfolio management
"""

import streamlit as st

# Page config MUST BE FIRST
st.set_page_config(
    page_title="StockIQ Pro - Indian Markets",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import sqlite3
import hashlib
import time
from plotly.subplots import make_subplots

# Professional Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 0;
        background-color: #0e1117;
    }
    
    /* Professional Metrics */
    .stMetric {
        background: linear-gradient(135deg, #1e1e2e 0%, #252535 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    [data-testid="metric-container"] {
        color: #ffffff !important;
    }
    
    [data-testid="metric-container"] > div {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    [data-testid="metric-container"] label {
        color: #9ca3af !important;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Professional Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #1e1e2e;
        padding: 4px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 24px;
        background-color: transparent;
        color: #9ca3af;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(59, 130, 246, 0.1);
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.875rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: #1e1e2e;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* DataFrames */
    .dataframe {
        background-color: #1e1e2e !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .dataframe th {
        background-color: #252535 !important;
        color: #9ca3af !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        padding: 1rem !important;
    }
    
    .dataframe td {
        background-color: #1e1e2e !important;
        color: #ffffff !important;
        padding: 0.875rem 1rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1a2e;
        padding: 2rem 1rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e1e2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3b82f6;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user' not in st.session_state:
    st.session_state.user = None
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Database setup
def init_db():
    """Initialize SQLite database for user data"""
    conn = sqlite3.connect('financial_dashboard.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, email TEXT UNIQUE, password TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio
                 (id INTEGER PRIMARY KEY, user_id INTEGER, symbol TEXT, 
                  quantity INTEGER, avg_price REAL, transaction_date DATE,
                  transaction_type TEXT, notes TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist
                 (id INTEGER PRIMARY KEY, user_id INTEGER, symbol TEXT,
                  target_price REAL, stop_loss REAL, notes TEXT,
                  added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY, user_id INTEGER, symbol TEXT,
                  alert_type TEXT, target_price REAL, created_date DATE,
                  triggered BOOLEAN DEFAULT FALSE)''')
    
    conn.commit()
    conn.close()

init_db()

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(email, password):
    conn = sqlite3.connect('financial_dashboard.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=? AND password=?", 
              (email, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user

def create_user(email, password):
    conn = sqlite3.connect('financial_dashboard.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)",
                  (email, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except:
        conn.close()
        return False

# Enhanced Indian stock symbols with sectors
INDIAN_STOCKS = {
    # Nifty 50 Leaders
    'RELIANCE.NS': {'name': 'Reliance Industries', 'sector': 'Oil & Gas'},
    'TCS.NS': {'name': 'Tata Consultancy Services', 'sector': 'IT'},
    'HDFCBANK.NS': {'name': 'HDFC Bank', 'sector': 'Banking'},
    'INFY.NS': {'name': 'Infosys', 'sector': 'IT'},
    'ICICIBANK.NS': {'name': 'ICICI Bank', 'sector': 'Banking'},
    'HINDUNILVR.NS': {'name': 'Hindustan Unilever', 'sector': 'FMCG'},
    'ITC.NS': {'name': 'ITC Limited', 'sector': 'FMCG'},
    'SBIN.NS': {'name': 'State Bank of India', 'sector': 'Banking'},
    'BHARTIARTL.NS': {'name': 'Bharti Airtel', 'sector': 'Telecom'},
    'KOTAKBANK.NS': {'name': 'Kotak Mahindra Bank', 'sector': 'Banking'},
    'LT.NS': {'name': 'Larsen & Toubro', 'sector': 'Infrastructure'},
    'AXISBANK.NS': {'name': 'Axis Bank', 'sector': 'Banking'},
    'ASIANPAINT.NS': {'name': 'Asian Paints', 'sector': 'Paints'},
    'MARUTI.NS': {'name': 'Maruti Suzuki', 'sector': 'Automobile'},
    'SUNPHARMA.NS': {'name': 'Sun Pharmaceutical', 'sector': 'Pharma'},
    'TITAN.NS': {'name': 'Titan Company', 'sector': 'Consumer Goods'},
    'ULTRACEMCO.NS': {'name': 'UltraTech Cement', 'sector': 'Cement'},
    'BAJFINANCE.NS': {'name': 'Bajaj Finance', 'sector': 'Finance'},
    'WIPRO.NS': {'name': 'Wipro', 'sector': 'IT'},
    'NESTLEIND.NS': {'name': 'Nestle India', 'sector': 'FMCG'},
    'BAJAJFINSV.NS': {'name': 'Bajaj Finserv', 'sector': 'Finance'},
    'TATAMOTORS.NS': {'name': 'Tata Motors', 'sector': 'Automobile'},
    'HCLTECH.NS': {'name': 'HCL Technologies', 'sector': 'IT'},
    'ADANIENT.NS': {'name': 'Adani Enterprises', 'sector': 'Infrastructure'},
    'ADANIPORTS.NS': {'name': 'Adani Ports', 'sector': 'Infrastructure'},
    'ONGC.NS': {'name': 'ONGC', 'sector': 'Oil & Gas'},
    'NTPC.NS': {'name': 'NTPC', 'sector': 'Power'},
    'POWERGRID.NS': {'name': 'Power Grid', 'sector': 'Power'},
    'M&M.NS': {'name': 'Mahindra & Mahindra', 'sector': 'Automobile'},
    'TATASTEEL.NS': {'name': 'Tata Steel', 'sector': 'Metals'},
    '^NSEI': {'name': 'Nifty 50', 'sector': 'Index'},
    '^BSESN': {'name': 'Sensex', 'sector': 'Index'}
}

@st.cache_data(ttl=300)
def get_stock_data(symbol, period='1mo'):
    """Fetch real stock data using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        
        # Get historical data
        hist = stock.history(period=period)
        if hist.empty:
            hist = stock.history(period='5d')
        
        # Get current info
        info = stock.info
        
        # Get current price from multiple sources
        current_price = None
        if 'currentPrice' in info:
            current_price = info['currentPrice']
        elif 'regularMarketPrice' in info:
            current_price = info['regularMarketPrice']
        elif len(hist) > 0:
            current_price = hist['Close'].iloc[-1]
        else:
            current_price = 0
        
        # Calculate changes
        if len(hist) >= 2:
            prev_close = hist['Close'].iloc[-2]
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
        else:
            change = 0
            change_percent = 0
        
        # Get all relevant data
        return {
            'symbol': symbol,
            'name': INDIAN_STOCKS.get(symbol, {}).get('name', symbol),
            'sector': INDIAN_STOCKS.get(symbol, {}).get('sector', 'Unknown'),
            'current_price': current_price,
            'change': change,
            'change_percent': change_percent,
            'history': hist,
            'info': info,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 1),
            'volume': info.get('volume', 0),
            'avg_volume': info.get('averageVolume', 0),
            'day_high': info.get('dayHigh', 0),
            'day_low': info.get('dayLow', 0),
            '52w_high': info.get('fiftyTwoWeekHigh', 0),
            '52w_low': info.get('fiftyTwoWeekLow', 0),
            'price_to_book': info.get('priceToBook', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'return_on_equity': info.get('returnOnEquity', 0),
            'profit_margins': info.get('profitMargins', 0),
            'gross_margins': info.get('grossMargins', 0)
        }
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    df = df.copy()
    
    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Support and Resistance Levels
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['Pivot'] - df['Low']
    df['S1'] = 2 * df['Pivot'] - df['High']
    df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
    df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
    
    return df

def get_market_sentiment(stock_data):
    """Calculate market sentiment score"""
    score = 50  # Neutral base
    
    if stock_data and len(stock_data['history']) > 0:
        df = calculate_technical_indicators(stock_data['history'])
        latest = df.iloc[-1]
        
        # RSI Analysis
        if 'RSI' in df.columns and not pd.isna(latest['RSI']):
            if latest['RSI'] < 30:
                score += 20  # Oversold
            elif latest['RSI'] > 70:
                score -= 20  # Overbought
            else:
                score += (50 - abs(latest['RSI'] - 50)) / 5
        
        # MACD Analysis
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            if latest['MACD'] > latest['MACD_Signal']:
                score += 10  # Bullish crossover
            else:
                score -= 10  # Bearish crossover
        
        # Moving Average Analysis
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            if not pd.isna(latest['SMA_50']) and not pd.isna(latest['SMA_200']):
                if latest['Close'] > latest['SMA_50'] > latest['SMA_200']:
                    score += 15  # Strong uptrend
                elif latest['Close'] < latest['SMA_50'] < latest['SMA_200']:
                    score -= 15  # Strong downtrend
        
        # Bollinger Bands
        if 'BB_Position' in df.columns and not pd.isna(latest['BB_Position']):
            if latest['BB_Position'] < 0.2:
                score += 10  # Near lower band
            elif latest['BB_Position'] > 0.8:
                score -= 10  # Near upper band
        
        # Volume Analysis
        if stock_data['volume'] > stock_data['avg_volume'] * 1.5:
            if stock_data['change_percent'] > 0:
                score += 5  # High volume on up day
            else:
                score -= 5  # High volume on down day
    
    return max(0, min(100, score))

# Sidebar Authentication
with st.sidebar:
    st.markdown("## 🏛️ StockIQ Pro")
    st.markdown("---")
    
    if st.session_state.user is None:
        st.markdown("### Sign In")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="user@example.com")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("LOGIN", use_container_width=True)
                
                if submit:
                    user = verify_user(email, password)
                    if user:
                        st.session_state.user = user
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("register_form"):
                new_email = st.text_input("Email", placeholder="user@example.com")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register = st.form_submit_button("REGISTER", use_container_width=True)
                
                if register:
                    if new_password != confirm_password:
                        st.error("Passwords don't match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    elif create_user(new_email, new_password):
                        st.success("Account created! Please login.")
                    else:
                        st.error("Email already exists")
    else:
        st.markdown(f"### Welcome back!")
        st.markdown(f"📧 {st.session_state.user[1]}")
        st.markdown("---")
        
        # Quick Stats
        if st.session_state.user:
            conn = sqlite3.connect('financial_dashboard.db')
            c = conn.cursor()
            
            # Portfolio value
            c.execute("SELECT COUNT(DISTINCT symbol) FROM portfolio WHERE user_id=?", 
                      (st.session_state.user[0],))
            holdings_count = c.fetchone()[0]
            
            # Watchlist count
            c.execute("SELECT COUNT(*) FROM watchlist WHERE user_id=?", 
                      (st.session_state.user[0],))
            watchlist_count = c.fetchone()[0]
            
            conn.close()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Holdings", holdings_count)
            with col2:
                st.metric("Watchlist", watchlist_count)
        
        st.markdown("---")
        
        if st.button("🚪 LOGOUT", use_container_width=True):
            st.session_state.user = None
            st.rerun()

# Main App Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
    font-size: 3rem; font-weight: 800; margin-bottom: 0;'>
    StockIQ Pro
    </h1>
    <p style='text-align: center; color: #9ca3af; font-size: 1.125rem; margin-top: 0;'>
    Professional Indian Stock Market Analytics
    </p>
    """, unsafe_allow_html=True)

# Market Overview Cards
st.markdown("### 📊 Market Overview")
col1, col2, col3, col4, col5 = st.columns(5)

# Fetch market indices
indices = ['^NSEI', '^BSESN', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']
for idx, (col, symbol) in enumerate(zip([col1, col2, col3, col4, col5], indices)):
    with col:
        data = get_stock_data(symbol, '1d')
        if data:
            color = "#22c55e" if data['change_percent'] >= 0 else "#ef4444"
            arrow = "↑" if data['change_percent'] >= 0 else "↓"
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1e1e2e 0%, #252535 100%); 
                        padding: 1.25rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
                        height: 140px;'>
                <p style='color: #9ca3af; font-size: 0.75rem; margin: 0; font-weight: 600; 
                         text-transform: uppercase; letter-spacing: 0.05em;'>
                    {data['name'][:12] + '...' if len(data['name']) > 12 else data['name']}
                </p>
                <p style='color: #ffffff; font-size: 1.5rem; font-weight: 700; margin: 0.25rem 0;'>
                    ₹{data['current_price']:,.2f}
                </p>
                <p style='color: {color}; font-size: 1rem; margin: 0; font-weight: 600;'>
                    {arrow} {abs(data['change_percent']):.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

# Main Content Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Analysis", 
    "💼 Portfolio", 
    "⚡ Technical", 
    "💰 Valuation",
    "📰 News & Sentiment",
    "🎯 Screener",
    "⚙️ Settings"
])

# Tab 1: Stock Analysis
with tab1:
    st.markdown("### Stock Analysis Dashboard")
    
    # Stock selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        # Group stocks by sector
        sector_stocks = {}
        for symbol, info in INDIAN_STOCKS.items():
            sector = info['sector']
            if sector not in sector_stocks:
                sector_stocks[sector] = []
            sector_stocks[sector].append((symbol, info['name']))
        
        # Create options with sector grouping
        stock_options = []
        for sector, stocks in sorted(sector_stocks.items()):
            stock_options.extend([(f"{info} ({symbol})", symbol) for symbol, info in sorted(stocks)])
        
        selected_display, selected_stock = st.selectbox(
            "Select Stock",
            options=stock_options,
            format_func=lambda x: x[0]
        ) if stock_options else (None, None)
    
    with col2:
        period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"])
    
    with col3:
        if st.button("⭐ Add to Watchlist", use_container_width=True) and st.session_state.user and selected_stock:
            conn = sqlite3.connect('financial_dashboard.db')
            c = conn.cursor()
            try:
                c.execute("INSERT INTO watchlist (user_id, symbol) VALUES (?, ?)",
                         (st.session_state.user[0], selected_stock))
                conn.commit()
                st.success("Added to watchlist!")
            except:
                st.info("Already in watchlist")
            conn.close()
    
    if selected_stock:
        stock_data = get_stock_data(selected_stock, period)
        
        if stock_data and stock_data['history'] is not None and len(stock_data['history']) > 0:
            # Key Metrics Row
            st.markdown("#### Key Metrics")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            metrics = [
                ("P/E Ratio", stock_data['pe_ratio'], ""),
                ("Market Cap", stock_data['market_cap']/10000000, " Cr"),
                ("Beta", stock_data['beta'], ""),
                ("Div Yield", stock_data['dividend_yield']*100 if stock_data['dividend_yield'] else 0, "%"),
                ("ROE", stock_data['return_on_equity']*100 if stock_data['return_on_equity'] else 0, "%"),
                ("D/E Ratio", stock_data['debt_to_equity'], "")
            ]
            
            for col, (label, value, suffix) in zip([col1, col2, col3, col4, col5, col6], metrics):
                with col:
                    if value and value != 0:
                        st.metric(label, f"{value:.2f}{suffix}")
                    else:
                        st.metric(label, "N/A")
            
            # Advanced Chart
            st.markdown("#### Price Chart")
            
            # Calculate indicators
            df = calculate_technical_indicators(stock_data['history'])
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=('Price & Indicators', 'Volume', 'RSI')
            )
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='#22c55e',
                decreasing_line_color='#ef4444'
            ), row=1, col=1)
            
            # Add moving averages
            if 'SMA_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['SMA_20'],
                    name='SMA 20',
                    line=dict(color='#3b82f6', width=1)
                ), row=1, col=1)
            
            if 'SMA_50' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['SMA_50'],
                    name='SMA 50',
                    line=dict(color='#8b5cf6', width=1)
                ), row=1, col=1)
            
            # Bollinger Bands
            if 'BB_Upper' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.1)'
                ), row=1, col=1)
            
            # Volume chart
            colors = ['#ef4444' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#22c55e' 
                     for i in range(len(df))]
            
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=colors
            ), row=2, col=1)
            
            # RSI chart
            if 'RSI' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['RSI'],
                    name='RSI',
                    line=dict(color='#f59e0b', width=2)
                ), row=3, col=1)
                
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="Overbought", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", 
                             annotation_text="Oversold", row=3, col=1)
            
            # Update layout
            fig.update_layout(
                title=f"{stock_data['name']} ({stock_data['symbol']}) - {stock_data['sector']}",
                xaxis_title="Date",
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
                )
            )
            
            fig.update_xaxes(rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment Score
            sentiment_score = get_market_sentiment(stock_data)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### Market Sentiment Analysis")
                
                # Sentiment gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = sentiment_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Sentiment Score"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#3b82f6"},
                        'steps': [
                            {'range': [0, 30], 'color': "#ef4444"},
                            {'range': [30, 70], 'color': "#f59e0b"},
                            {'range': [70, 100], 'color': "#22c55e"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                st.markdown("#### Technical Summary")
                
                if sentiment_score >= 70:
                    st.success("🟢 **STRONG BUY**")
                    st.write("Strong bullish signals across multiple indicators")
                elif sentiment_score >= 55:
                    st.info("🔵 **BUY**")
                    st.write("Moderately bullish, good entry opportunity")
                elif sentiment_score >= 45:
                    st.warning("🟡 **HOLD**")
                    st.write("Mixed signals, wait for clearer direction")
                else:
                    st.error("🔴 **SELL**")
                    st.write("Bearish signals, consider reducing position")
                
                # Key technical levels
                if len(df) > 0:
                    latest = df.iloc[-1]
                    st.markdown("**Key Levels:**")
                    st.write(f"• Support: ₹{latest['S1']:.2f}")
                    st.write(f"• Resistance: ₹{latest['R1']:.2f}")
                    if 'ATR' in df.columns:
                        st.write(f"• Volatility (ATR): ₹{latest['ATR']:.2f}")

# Tab 2: Portfolio Management
with tab2:
    st.markdown("### Portfolio Management")
    
    if st.session_state.user:
        # Transaction Entry
        with st.expander("➕ Add Transaction", expanded=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                trans_symbol = st.selectbox(
                    "Stock",
                    options=list(INDIAN_STOCKS.keys()),
                    format_func=lambda x: f"{INDIAN_STOCKS[x]['name']} ({x})"
                )
            
            with col2:
                trans_type = st.selectbox("Type", ["Buy", "Sell"])
            
            with col3:
                trans_quantity = st.number_input("Quantity", min_value=1, value=1)
            
            with col4:
                trans_price = st.number_input("Price/Share", min_value=0.01, value=100.00)
            
            with col5:
                trans_date = st.date_input("Date", value=datetime.now())
            
            trans_notes = st.text_area("Notes (Optional)", height=100)
            
            if st.button("💾 Save Transaction", use_container_width=True):
                conn = sqlite3.connect('financial_dashboard.db')
                c = conn.cursor()
                
                # Check existing position
                c.execute("SELECT quantity, avg_price FROM portfolio WHERE user_id=? AND symbol=?",
                         (st.session_state.user[0], trans_symbol))
                existing = c.fetchone()
                
                if trans_type == "Buy":
                    if existing:
                        new_quantity = existing[0] + trans_quantity
                        new_avg_price = ((existing[0] * existing[1]) + (trans_quantity * trans_price)) / new_quantity
                        
                        c.execute("UPDATE portfolio SET quantity=?, avg_price=? WHERE user_id=? AND symbol=?",
                                 (new_quantity, new_avg_price, st.session_state.user[0], trans_symbol))
                    else:
                        c.execute("INSERT INTO portfolio (user_id, symbol, quantity, avg_price, transaction_date, transaction_type, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                 (st.session_state.user[0], trans_symbol, trans_quantity, trans_price, trans_date, trans_type, trans_notes))
                else:  # Sell
                    if existing and existing[0] >= trans_quantity:
                        new_quantity = existing[0] - trans_quantity
                        if new_quantity > 0:
                            c.execute("UPDATE portfolio SET quantity=? WHERE user_id=? AND symbol=?",
                                     (new_quantity, st.session_state.user[0], trans_symbol))
                        else:
                            c.execute("DELETE FROM portfolio WHERE user_id=? AND symbol=?",
                                     (st.session_state.user[0], trans_symbol))
                    else:
                        st.error("Insufficient quantity to sell")
                        conn.close()
                        st.stop()
                
                conn.commit()
                conn.close()
                st.success("Transaction saved successfully!")
                st.rerun()
        
        # Portfolio Holdings
        st.markdown("#### Current Holdings")
        
        conn = sqlite3.connect('financial_dashboard.db')
        c = conn.cursor()
        c.execute("SELECT symbol, quantity, avg_price FROM portfolio WHERE user_id=?", 
                  (st.session_state.user[0],))
        holdings = c.fetchall()
        conn.close()
        
        if holdings:
            portfolio_data = []
            total_value = 0
            total_cost = 0
            total_day_change = 0
            
            for symbol, quantity, avg_price in holdings:
                stock_data = get_stock_data(symbol, '1d')
                if stock_data:
                    current_price = stock_data['current_price']
                    current_value = quantity * current_price
                    cost_basis = quantity * avg_price
                    profit_loss = current_value - cost_basis
                    profit_loss_pct = (profit_loss / cost_basis) * 100
                    day_change = quantity * stock_data['change']
                    
                    total_value += current_value
                    total_cost += cost_basis
                    total_day_change += day_change
                    
                    portfolio_data.append({
                        'Symbol': symbol,
                        'Name': INDIAN_STOCKS.get(symbol, {}).get('name', symbol),
                        'Quantity': quantity,
                        'Avg Cost': f"₹{avg_price:.2f}",
                        'Current Price': f"₹{current_price:.2f}",
                        'Current Value': f"₹{current_value:,.2f}",
                        'Day Change': f"₹{day_change:,.2f}",
                        'Total P&L': f"₹{profit_loss:,.2f}",
                        'P&L %': f"{profit_loss_pct:.2f}%",
                        'Weight %': 0  # Will calculate after
                    })
            
            # Calculate portfolio weights
            for item in portfolio_data:
                value = float(item['Current Value'].replace('₹', '').replace(',', ''))
                item['Weight %'] = f"{(value / total_value * 100):.1f}%"
            
            # Portfolio Summary Cards
            col1, col2, col3, col4 = st.columns(4)
            
            total_pl = total_value - total_cost
            total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
            
            with col1:
                st.metric("Total Value", f"₹{total_value:,.2f}", 
                         f"{total_day_change:+,.2f} today")
            
            with col2:
                st.metric("Total Invested", f"₹{total_cost:,.2f}")
            
            with col3:
                color = "normal" if total_pl >= 0 else "inverse"
                st.metric("Total P&L", f"₹{total_pl:,.2f}", 
                         f"{total_pl_pct:.2f}%", delta_color=color)
            
            with col4:
                st.metric("Holdings", len(holdings))
            
            # Holdings Table
            df_portfolio = pd.DataFrame(portfolio_data)
            st.dataframe(df_portfolio, use_container_width=True, height=400)
            
            # Portfolio Composition Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                fig_pie = px.pie(
                    values=[float(d['Current Value'].replace('₹', '').replace(',', '')) for d in portfolio_data],
                    names=[d['Name'] for d in portfolio_data],
                    title="Portfolio Allocation",
                    hole=0.4
                )
                fig_pie.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Sector allocation
                sector_allocation = {}
                for symbol, quantity, avg_price in holdings:
                    sector = INDIAN_STOCKS.get(symbol, {}).get('sector', 'Unknown')
                    stock_data = get_stock_data(symbol, '1d')
                    if stock_data:
                        value = quantity * stock_data['current_price']
                        sector_allocation[sector] = sector_allocation.get(sector, 0) + value
                
                fig_sector = px.bar(
                    x=list(sector_allocation.values()),
                    y=list(sector_allocation.keys()),
                    orientation='h',
                    title="Sector Allocation",
                    labels={'x': 'Value (₹)', 'y': 'Sector'}
                )
                fig_sector.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.info("No holdings yet. Add your first transaction above!")
    else:
        st.warning("Please login to manage your portfolio")

# Tab 3: Technical Analysis
with tab3:
    st.markdown("### Advanced Technical Analysis")
    
    if selected_stock:
        stock_data = get_stock_data(selected_stock, period)
        
        if stock_data and len(stock_data['history']) > 0:
            df = calculate_technical_indicators(stock_data['history'])
            
            # Technical Indicators Summary
            st.markdown("#### Technical Indicators")
            
            col1, col2, col3, col4 = st.columns(4)
            
            latest = df.iloc[-1]
            
            with col1:
                rsi_value = latest['RSI'] if 'RSI' in df.columns and not pd.isna(latest['RSI']) else 50
                st.metric("RSI (14)", f"{rsi_value:.2f}")
                if rsi_value > 70:
                    st.error("Overbought")
                elif rsi_value < 30:
                    st.success("Oversold")
                else:
                    st.info("Neutral")
            
            with col2:
                if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                    macd_value = latest['MACD']
                    signal_value = latest['MACD_Signal']
                    st.metric("MACD", f"{macd_value:.2f}")
                    if macd_value > signal_value:
                        st.success("Bullish")
                    else:
                        st.error("Bearish")
            
            with col3:
                if '%K' in df.columns and not pd.isna(latest['%K']):
                    stoch_value = latest['%K']
                    st.metric("Stochastic %K", f"{stoch_value:.2f}")
                    if stoch_value > 80:
                        st.error("Overbought")
                    elif stoch_value < 20:
                        st.success("Oversold")
                    else:
                        st.info("Neutral")
            
            with col4:
                if 'ATR' in df.columns and not pd.isna(latest['ATR']):
                    atr_value = latest['ATR']
                    atr_pct = (atr_value / latest['Close']) * 100
                    st.metric("ATR", f"₹{atr_value:.2f}", f"{atr_pct:.2f}%")
            
            # Support & Resistance Levels
            st.markdown("#### Support & Resistance Levels")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Support Levels**")
                if 'S2' in df.columns:
                    st.write(f"• S2: ₹{latest['S2']:.2f} (Strong)")
                    st.write(f"• S1: ₹{latest['S1']:.2f} (Moderate)")
                    st.write(f"• Pivot: ₹{latest['Pivot']:.2f}")
            
            with col2:
                st.markdown("**Resistance Levels**")
                if 'R2' in df.columns:
                    st.write(f"• Pivot: ₹{latest['Pivot']:.2f}")
                    st.write(f"• R1: ₹{latest['R1']:.2f} (Moderate)")
                    st.write(f"• R2: ₹{latest['R2']:.2f} (Strong)")
            
            # Technical Chart Patterns
            st.markdown("#### Chart Pattern Recognition")
            
            # Simple pattern detection
            patterns = []
            
            # Golden Cross / Death Cross
            if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
                if len(df) > 1:
                    if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1] and df['SMA_50'].iloc[-2] <= df['SMA_200'].iloc[-2]:
                        patterns.append(("Golden Cross", "Bullish", "50-day MA crossed above 200-day MA"))
                    elif df['SMA_50'].iloc[-1] < df['SMA_200'].iloc[-1] and df['SMA_50'].iloc[-2] >= df['SMA_200'].iloc[-2]:
                        patterns.append(("Death Cross", "Bearish", "50-day MA crossed below 200-day MA"))
            
            # RSI Divergence
            if 'RSI' in df.columns and len(df) > 14:
                price_trend = df['Close'].iloc[-7:].pct_change().mean()
                rsi_trend = df['RSI'].iloc[-7:].pct_change().mean()
                
                if price_trend > 0 and rsi_trend < 0:
                    patterns.append(("Bearish Divergence", "Bearish", "Price rising but RSI falling"))
                elif price_trend < 0 and rsi_trend > 0:
                    patterns.append(("Bullish Divergence", "Bullish", "Price falling but RSI rising"))
            
            # Bollinger Band Squeeze
            if 'BB_Width' in df.columns:
                recent_width = df['BB_Width'].iloc[-20:].mean()
                current_width = df['BB_Width'].iloc[-1]
                if current_width < recent_width * 0.7:
                    patterns.append(("Bollinger Squeeze", "Neutral", "Volatility contraction - breakout expected"))
            
            if patterns:
                for pattern, signal, description in patterns:
                    col1, col2, col3 = st.columns([2, 1, 3])
                    with col1:
                        st.write(f"**{pattern}**")
                    with col2:
                        if signal == "Bullish":
                            st.success(signal)
                        elif signal == "Bearish":
                            st.error(signal)
                        else:
                            st.info(signal)
                    with col3:
                        st.write(description)
            else:
                st.info("No significant patterns detected")

# Tab 4: DCF Valuation
with tab4:
    st.markdown("### Discounted Cash Flow (DCF) Valuation")
    
    if selected_stock:
        stock_data = get_stock_data(selected_stock, period)
        
        if stock_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Input Parameters")
                
                # Try to get financial data
                try:
                    ticker = yf.Ticker(selected_stock)
                    financials = ticker.financials
                    cash_flow = ticker.cashflow
                    
                    # Default values
                    fcf = st.number_input("Free Cash Flow (₹ Cr)", value=1000.0, step=100.0)
                    growth_rate = st.slider("Growth Rate (%)", 0.0, 30.0, 15.0, 0.5)
                    terminal_growth = st.slider("Terminal Growth (%)", 0.0, 10.0, 3.0, 0.5)
                    discount_rate = st.slider("Discount Rate (%)", 5.0, 20.0, 12.0, 0.5)
                    years = st.slider("Projection Years", 5, 15, 10)
                    
                except:
                    st.warning("Could not fetch financial data. Using default values.")
                    fcf = st.number_input("Free Cash Flow (₹ Cr)", value=1000.0, step=100.0)
                    growth_rate = st.slider("Growth Rate (%)", 0.0, 30.0, 15.0, 0.5)
                    terminal_growth = st.slider("Terminal Growth (%)", 0.0, 10.0, 3.0, 0.5)
                    discount_rate = st.slider("Discount Rate (%)", 5.0, 20.0, 12.0, 0.5)
                    years = st.slider("Projection Years", 5, 15, 10)
                
                shares_outstanding = st.number_input("Shares Outstanding (Cr)", value=100.0, step=10.0)
                
                if st.button("Calculate DCF Value", use_container_width=True):
                    # DCF Calculation
                    dcf_value = 0
                    cash_flows = []
                    
                    for year in range(1, years + 1):
                        cf = fcf * ((1 + growth_rate/100) ** year)
                        pv = cf / ((1 + discount_rate/100) ** year)
                        dcf_value += pv
                        cash_flows.append({
                            'Year': year,
                            'Cash Flow': cf,
                            'Present Value': pv
                        })
                    
                    # Terminal Value
                    terminal_cf = fcf * ((1 + growth_rate/100) ** years) * (1 + terminal_growth/100)
                    terminal_value = terminal_cf / (discount_rate/100 - terminal_growth/100)
                    terminal_pv = terminal_value / ((1 + discount_rate/100) ** years)
                    
                    enterprise_value = dcf_value + terminal_pv
                    value_per_share = (enterprise_value / shares_outstanding) * 10
                    
                    st.session_state.dcf_result = {
                        'enterprise_value': enterprise_value,
                        'terminal_value': terminal_pv,
                        'value_per_share': value_per_share,
                        'cash_flows': cash_flows
                    }
            
            with col2:
                st.markdown("#### Valuation Results")
                
                if 'dcf_result' in st.session_state:
                    result = st.session_state.dcf_result
                    
                    st.metric("Enterprise Value", f"₹{result['enterprise_value']:,.0f} Cr")
                    st.metric("Fair Value per Share", f"₹{result['value_per_share']:,.2f}")
                    
                    current_price = stock_data['current_price']
                    upside = ((result['value_per_share'] - current_price) / current_price) * 100
                    
                    st.metric("Current Price", f"₹{current_price:.2f}")
                    st.metric("Upside/Downside", f"{upside:+.1f}%")
                    
                    if upside > 20:
                        st.success("🟢 **BUY** - Undervalued")
                    elif upside > -10:
                        st.info("🔵 **HOLD** - Fairly Valued")
                    else:
                        st.error("🔴 **SELL** - Overvalued")
                    
                    # Cash flow projection chart
                    df_cf = pd.DataFrame(result['cash_flows'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_cf['Year'],
                        y=df_cf['Cash Flow'],
                        name='Projected Cash Flow',
                        marker_color='lightblue'
                    ))
                    fig.add_trace(go.Bar(
                        x=df_cf['Year'],
                        y=df_cf['Present Value'],
                        name='Present Value',
                        marker_color='darkblue'
                    ))
                    
                    fig.update_layout(
                        title="Cash Flow Projections",
                        template="plotly_dark",
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Click 'Calculate DCF Value' to see results")

# Tab 5: News & Sentiment
with tab5:
    st.markdown("### News & Market Sentiment")
    
    if selected_stock:
        try:
            import feedparser
            
            company_name = INDIAN_STOCKS.get(selected_stock, {}).get('name', selected_stock)
            
            # Google News RSS
            rss_url = f"https://news.google.com/rss/search?q={company_name.replace(' ', '+')}+stock+NSE&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(rss_url)
            
            if feed.entries:
                for entry in feed.entries[:10]:
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.markdown(f"### [{entry.title}]({entry.link})")
                            if hasattr(entry, 'published'):
                                st.caption(f"📅 {entry.published}")
                        
                        with col2:
                            # Simple sentiment
                            title_lower = entry.title.lower()
                            if any(word in title_lower for word in ['gain', 'rise', 'surge', 'rally', 'profit']):
                                st.success("Positive")
                            elif any(word in title_lower for word in ['fall', 'drop', 'loss', 'decline', 'crash']):
                                st.error("Negative")
                            else:
                                st.info("Neutral")
                        
                        st.divider()
            else:
                st.info("No recent news found. Try another stock.")
                
        except Exception as e:
            st.error("Please install feedparser: pip install feedparser")

# Tab 6: Stock Screener
with tab6:
    st.markdown("### Stock Screener")
    
    # Screener filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_pe = st.number_input("Min P/E", value=0.0)
        max_pe = st.number_input("Max P/E", value=50.0)
    
    with col2:
        min_market_cap = st.number_input("Min Market Cap (Cr)", value=0.0)
        max_market_cap = st.number_input("Max Market Cap (Cr)", value=1000000.0)
    
    with col3:
        min_div_yield = st.number_input("Min Div Yield (%)", value=0.0)
        sectors = st.multiselect("Sectors", 
                                 options=list(set(info['sector'] for info in INDIAN_STOCKS.values())))
    
    with col4:
        min_change = st.number_input("Min Change %", value=-100.0)
        max_change = st.number_input("Max Change %", value=100.0)
    
    if st.button("🔍 Run Screener", use_container_width=True):
        screener_results = []
        
        progress = st.progress(0)
        status = st.empty()
        
        for idx, (symbol, info) in enumerate(INDIAN_STOCKS.items()):
            if symbol in ['^NSEI', '^BSESN']:  # Skip indices
                continue
                
            status.text(f"Scanning {info['name']}...")
            progress.progress((idx + 1) / len(INDIAN_STOCKS))
            
            # Apply sector filter
            if sectors and info['sector'] not in sectors:
                continue
            
            data = get_stock_data(symbol, '1d')
            
            if data:
                # Apply filters
                if min_pe <= data['pe_ratio'] <= max_pe:
                    if min_market_cap <= data['market_cap']/10000000 <= max_market_cap:
                        if data['dividend_yield'] and data['dividend_yield']*100 >= min_div_yield:
                            if min_change <= data['change_percent'] <= max_change:
                                screener_results.append({
                                    'Symbol': symbol,
                                    'Name': info['name'],
                                    'Sector': info['sector'],
                                    'Price': f"₹{data['current_price']:.2f}",
                                    'Change %': f"{data['change_percent']:.2f}%",
                                    'P/E': f"{data['pe_ratio']:.2f}",
                                    'Market Cap': f"₹{data['market_cap']/10000000:.0f} Cr",
                                    'Div Yield': f"{data['dividend_yield']*100:.2f}%" if data['dividend_yield'] else "0%"
                                })
        
        progress.empty()
        status.empty()
        
        if screener_results:
            st.success(f"Found {len(screener_results)} stocks matching criteria")
            df_screener = pd.DataFrame(screener_results)
            st.dataframe(df_screener, use_container_width=True, height=600)
            
            # Download button
            csv = df_screener.to_csv(index=False)
            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No stocks found matching the criteria")

# Tab 7: Settings
with tab7:
    st.markdown("### Settings & Preferences")
    
    if st.session_state.user:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Alert Settings")
            
            alert_symbol = st.selectbox(
                "Stock for Alert",
                options=list(INDIAN_STOCKS.keys()),
                format_func=lambda x: f"{INDIAN_STOCKS[x]['name']} ({x})"
            )
            
            alert_type = st.selectbox("Alert Type", ["Price Above", "Price Below", "% Change"])
            alert_value = st.number_input("Alert Value", value=0.0)
            
            if st.button("🔔 Set Alert"):
                conn = sqlite3.connect('financial_dashboard.db')
                c = conn.cursor()
                c.execute("INSERT INTO alerts (user_id, symbol, alert_type, target_price, created_date) VALUES (?, ?, ?, ?, ?)",
                         (st.session_state.user[0], alert_symbol, alert_type, alert_value, datetime.now()))
                conn.commit()
                conn.close()
                st.success("Alert set successfully!")
        
        with col2:
            st.markdown("#### Export Data")
            
            export_type = st.selectbox("Export Type", ["Portfolio", "Transactions", "Watchlist"])
            
            if st.button("📊 Export to CSV"):
                conn = sqlite3.connect('financial_dashboard.db')
                
                if export_type == "Portfolio":
                    query = "SELECT * FROM portfolio WHERE user_id=?"
                elif export_type == "Transactions":
                    query = "SELECT * FROM portfolio WHERE user_id=?"
                else:
                    query = "SELECT * FROM watchlist WHERE user_id=?"
                
                df_export = pd.read_sql_query(query, conn, params=(st.session_state.user[0],))
                conn.close()
                
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"{export_type.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        # Account Settings
        st.markdown("#### Account Settings")
        
        with st.expander("Change Password"):
            old_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_new = st.text_input("Confirm New Password", type="password")
            
            if st.button("Update Password"):
                if verify_user(st.session_state.user[1], old_password):
                    if new_password == confirm_new:
                        conn = sqlite3.connect('financial_dashboard.db')
                        c = conn.cursor()
                        c.execute("UPDATE users SET password=? WHERE id=?",
                                 (hash_password(new_password), st.session_state.user[0]))
                        conn.commit()
                        conn.close()
                        st.success("Password updated successfully!")
                    else:
                        st.error("New passwords don't match")
                else:
                    st.error("Current password is incorrect")
        
        # Data Management
        st.markdown("#### Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Portfolio", use_container_width=True):
                if st.checkbox("I understand this will delete all portfolio data"):
                    conn = sqlite3.connect('financial_dashboard.db')
                    c = conn.cursor()
                    c.execute("DELETE FROM portfolio WHERE user_id=?", (st.session_state.user[0],))
                    conn.commit()
                    conn.close()
                    st.success("Portfolio cleared!")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Watchlist", use_container_width=True):
                conn = sqlite3.connect('financial_dashboard.db')
                c = conn.cursor()
                c.execute("DELETE FROM watchlist WHERE user_id=?", (st.session_state.user[0],))
                conn.commit()
                conn.close()
                st.success("Watchlist cleared!")
                st.rerun()
    else:
        st.warning("Please login to access settings")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #9ca3af; padding: 2rem 0;'>
    <p style='margin: 0;'>StockIQ Pro - Professional Indian Stock Market Analytics</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.875rem;'>
        Data provided by Yahoo Finance | For educational purposes only | Not investment advice
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.75rem;'>
        Made with ❤️ using Streamlit | © 2024 StockIQ Pro
    </p>
</div>
""", unsafe_allow_html=True)
                                