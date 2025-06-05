#!/usr/bin/env python3
"""
Indian Financial Dashboard - Complete Working System
Uses only FREE APIs - No paid subscriptions needed!
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
from bs4 import BeautifulSoup
import sqlite3
import hashlib
import os
from textblob import TextBlob
import openai  # For AI summaries (optional - can use free alternatives)

# Page config
st.set_page_config(
    page_title="StockIQ India - Professional Financial Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    .st-bw {
        background-color: #ffffff;
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

# Database setup
def init_db():
    """Initialize SQLite database for user data"""
    conn = sqlite3.connect('financial_dashboard.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, email TEXT UNIQUE, password TEXT)''')
    
    # Portfolio table
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio
                 (id INTEGER PRIMARY KEY, user_id INTEGER, symbol TEXT, 
                  quantity INTEGER, avg_price REAL, transaction_date DATE)''')
    
    # Watchlist table
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist
                 (id INTEGER PRIMARY KEY, user_id INTEGER, symbol TEXT)''')
    
    # Alerts table
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY, user_id INTEGER, symbol TEXT,
                  alert_type TEXT, target_price REAL, created_date DATE)''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Helper functions
def hash_password(password):
    """Hash password for security"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(email, password):
    """Verify user credentials"""
    conn = sqlite3.connect('financial_dashboard.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=? AND password=?", 
              (email, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user

def create_user(email, password):
    """Create new user"""
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

# Indian stock symbols mapping
INDIAN_STOCKS = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'HDFCBANK.NS': 'HDFC Bank',
    'INFY.NS': 'Infosys',
    'ICICIBANK.NS': 'ICICI Bank',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'ITC.NS': 'ITC Limited',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank',
    'LT.NS': 'Larsen & Toubro',
    'AXISBANK.NS': 'Axis Bank',
    'ASIANPAINT.NS': 'Asian Paints',
    'MARUTI.NS': 'Maruti Suzuki',
    'SUNPHARMA.NS': 'Sun Pharmaceutical',
    'TITAN.NS': 'Titan Company',
    'ULTRACEMCO.NS': 'UltraTech Cement',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'WIPRO.NS': 'Wipro',
    'NESTLEIND.NS': 'Nestle India',
    '^NSEI': 'Nifty 50',
    '^BSESN': 'Sensex'
}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol, period='1mo'):
    """Fetch stock data using yfinance (FREE)"""
    try:
        stock = yf.Ticker(symbol)
        
        # Get historical data
        hist = stock.history(period=period)
        
        # Get current info
        info = stock.info
        
        # Get current price
        current_price = info.get('currentPrice', hist['Close'].iloc[-1] if len(hist) > 0 else 0)
        
        # Calculate change
        if len(hist) >= 2:
            prev_close = hist['Close'].iloc[-2]
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
        else:
            change = 0
            change_percent = 0
        
        return {
            'symbol': symbol,
            'name': INDIAN_STOCKS.get(symbol, symbol),
            'current_price': current_price,
            'change': change,
            'change_percent': change_percent,
            'history': hist,
            'info': info,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'volume': info.get('volume', 0),
            'day_high': info.get('dayHigh', 0),
            'day_low': info.get('dayLow', 0),
            '52w_high': info.get('fiftyTwoWeekHigh', 0),
            '52w_low': info.get('fiftyTwoWeekLow', 0)
        }
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_news_sentiment(symbol):
    """Get news and sentiment using free news APIs"""
    try:
        # Using NewsAPI free tier (requires free API key)
        # You can get one at https://newsapi.org/register
        # For demo, using mock data
        
        company_name = INDIAN_STOCKS.get(symbol, symbol).split('.')[0]
        
        # In production, use actual NewsAPI:
        # url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey=YOUR_FREE_API_KEY"
        # response = requests.get(url)
        # news_data = response.json()
        
        # Mock news data for demo
        news_items = [
            {
                'title': f'{company_name} reports strong Q3 earnings',
                'description': 'Company beats analyst expectations with 15% revenue growth',
                'sentiment': 0.8,
                'source': 'Economic Times',
                'publishedAt': datetime.now() - timedelta(hours=2)
            },
            {
                'title': f'Analysts upgrade {company_name} stock',
                'description': 'Major brokerages raise target price citing strong fundamentals',
                'sentiment': 0.6,
                'source': 'Moneycontrol',
                'publishedAt': datetime.now() - timedelta(hours=5)
            },
            {
                'title': f'{company_name} announces expansion plans',
                'description': 'Company to invest Rs 5000 Cr in new manufacturing facility',
                'sentiment': 0.7,
                'source': 'Business Standard',
                'publishedAt': datetime.now() - timedelta(days=1)
            }
        ]
        
        # Calculate average sentiment
        avg_sentiment = sum(item['sentiment'] for item in news_items) / len(news_items)
        
        return {
            'news': news_items,
            'average_sentiment': avg_sentiment,
            'sentiment_label': 'Positive' if avg_sentiment > 0.5 else 'Negative' if avg_sentiment < -0.5 else 'Neutral'
        }
        
    except Exception as e:
        return {
            'news': [],
            'average_sentiment': 0,
            'sentiment_label': 'No data'
        }

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    return df

def calculate_dcf(fcf, growth_rate, terminal_growth, discount_rate, years=5):
    """Calculate DCF valuation"""
    dcf_value = 0
    
    # Project cash flows
    for year in range(1, years + 1):
        projected_fcf = fcf * ((1 + growth_rate) ** year)
        discounted_fcf = projected_fcf / ((1 + discount_rate) ** year)
        dcf_value += discounted_fcf
    
    # Terminal value
    terminal_fcf = fcf * ((1 + growth_rate) ** years) * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    discounted_terminal = terminal_value / ((1 + discount_rate) ** years)
    
    total_value = dcf_value + discounted_terminal
    
    return total_value

def get_fundamental_analysis(symbol):
    """Get fundamental analysis using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Financial statements
        income_stmt = stock.quarterly_income_stmt
        balance_sheet = stock.quarterly_balance_sheet
        cash_flow = stock.quarterly_cashflow
        
        # Key metrics
        metrics = {
            'Revenue Growth': calculate_growth(income_stmt.loc['Total Revenue'] if 'Total Revenue' in income_stmt.index else None),
            'Profit Margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
            'ROE': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
            'Debt to Equity': info.get('debtToEquity', 0),
            'Current Ratio': info.get('currentRatio', 0),
            'P/E Ratio': info.get('trailingPE', 0),
            'P/B Ratio': info.get('priceToBook', 0),
            'EPS': info.get('trailingEps', 0)
        }
        
        return metrics
    except:
        return {}

def calculate_growth(series):
    """Calculate growth rate from a series"""
    if series is None or len(series) < 2:
        return 0
    try:
        return ((series.iloc[0] - series.iloc[1]) / abs(series.iloc[1])) * 100
    except:
        return 0

# Sidebar - User Authentication
with st.sidebar:
    st.header("🔐 User Account")
    
    if st.session_state.user is None:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                user = verify_user(email, password)
                if user:
                    st.session_state.user = user
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        with tab2:
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.button("Sign Up"):
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif create_user(new_email, new_password):
                    st.success("Account created! Please login.")
                else:
                    st.error("Email already exists")
    else:
        st.write(f"Welcome, {st.session_state.user[1]}")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()

# Main App
st.title("📈 StockIQ India - Professional Financial Dashboard")
st.markdown("Real-time Indian stock market data, analysis, and portfolio management")

# Top metrics
col1, col2, col3, col4 = st.columns(4)

# Get market indices
nifty = get_stock_data('^NSEI', '1d')
sensex = get_stock_data('^BSESN', '1d')

if nifty:
    col1.metric(
        "NIFTY 50",
        f"{nifty['current_price']:,.2f}",
        f"{nifty['change_percent']:.2f}%"
    )

if sensex:
    col2.metric(
        "SENSEX",
        f"{sensex['current_price']:,.2f}",
        f"{sensex['change_percent']:.2f}%"
    )

# Portfolio value (if logged in)
if st.session_state.user:
    portfolio_value = 0
    portfolio_cost = 0
    
    conn = sqlite3.connect('financial_dashboard.db')
    c = conn.cursor()
    c.execute("SELECT symbol, quantity, avg_price FROM portfolio WHERE user_id=?", 
              (st.session_state.user[0],))
    holdings = c.fetchall()
    conn.close()
    
    for symbol, quantity, avg_price in holdings:
        stock_data = get_stock_data(symbol, '1d')
        if stock_data:
            portfolio_value += quantity * stock_data['current_price']
            portfolio_cost += quantity * avg_price
    
    portfolio_return = ((portfolio_value - portfolio_cost) / portfolio_cost * 100) if portfolio_cost > 0 else 0
    
    col3.metric(
        "Portfolio Value",
        f"₹{portfolio_value:,.2f}",
        f"{portfolio_return:.2f}%"
    )
    
    col4.metric(
        "Today's Gain/Loss",
        f"₹{(portfolio_value - portfolio_cost):,.2f}",
        f"{portfolio_return:.2f}%"
    )

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Market Overview", 
    "🔍 Stock Analysis", 
    "💼 Portfolio", 
    "📈 Technical Analysis",
    "💰 DCF Calculator",
    "🤖 AI Insights"
])

# Tab 1: Market Overview
with tab1:
    st.header("Market Overview")
    
    # Stock search
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_stock = st.selectbox(
            "Search Stocks",
            options=list(INDIAN_STOCKS.keys()),
            format_func=lambda x: f"{INDIAN_STOCKS[x]} ({x})"
        )
    
    with col2:
        if st.button("Add to Watchlist") and st.session_state.user:
            if selected_stock not in st.session_state.watchlist:
                st.session_state.watchlist.append(selected_stock)
                st.success("Added to watchlist!")
    
    # Get stock data
    stock_data = get_stock_data(selected_stock)
    
    if stock_data:
        # Stock info
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Current Price", f"₹{stock_data['current_price']:,.2f}")
        col2.metric("Change", f"₹{stock_data['change']:,.2f}", f"{stock_data['change_percent']:.2f}%")
        col3.metric("Volume", f"{stock_data['volume']:,}")
        col4.metric("Market Cap", f"₹{stock_data['market_cap']/10000000:.2f} Cr")
        
        # Price chart
        st.subheader("Price Chart")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=stock_data['history'].index,
            open=stock_data['history']['Open'],
            high=stock_data['history']['High'],
            low=stock_data['history']['Low'],
            close=stock_data['history']['Close'],
            name='Candles'
        ))
        
        fig.update_layout(
            title=f"{stock_data['name']} Price Chart",
            yaxis_title="Price (₹)",
            xaxis_title="Date",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("P/E Ratio", f"{stock_data['pe_ratio']:.2f}")
        col2.metric("52W High", f"₹{stock_data['52w_high']:,.2f}")
        col3.metric("52W Low", f"₹{stock_data['52w_low']:,.2f}")
        col4.metric("Dividend Yield", f"{stock_data['dividend_yield']*100:.2f}%")

# Tab 2: Stock Analysis
with tab2:
    st.header("Detailed Stock Analysis")
    
    if stock_data:
        # Fundamental Analysis
        st.subheader("Fundamental Analysis")
        
        fundamentals = get_fundamental_analysis(selected_stock)
        
        if fundamentals:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Profitability Metrics**")
                for metric, value in list(fundamentals.items())[:4]:
                    if value != 0:
                        st.write(f"{metric}: {value:.2f}{'%' if 'Growth' in metric or 'Margin' in metric or 'ROE' in metric else ''}")
            
            with col2:
                st.write("**Valuation Metrics**")
                for metric, value in list(fundamentals.items())[4:]:
                    if value != 0:
                        st.write(f"{metric}: {value:.2f}")
        
        # News & Sentiment
        st.subheader("News & Sentiment Analysis")
        
        news_data = get_news_sentiment(selected_stock)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            sentiment_color = "green" if news_data['average_sentiment'] > 0.5 else "red" if news_data['average_sentiment'] < -0.5 else "gray"
            st.markdown(f"### Sentiment: <span style='color: {sentiment_color}'>{news_data['sentiment_label']}</span>", unsafe_allow_html=True)
            st.write(f"Score: {news_data['average_sentiment']:.2f}")
        
        with col2:
            for news in news_data['news'][:3]:
                st.write(f"**{news['title']}**")
                st.write(f"{news['description']}")
                st.write(f"*{news['source']} - {news['publishedAt'].strftime('%Y-%m-%d %H:%M')}*")
                st.write("---")

# Tab 3: Portfolio Management
with tab3:
    st.header("Portfolio Management")
    
    if st.session_state.user:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Add Transaction")
            
            trans_col1, trans_col2, trans_col3, trans_col4 = st.columns(4)
            
            with trans_col1:
                trans_symbol = st.selectbox(
                    "Stock",
                    options=list(INDIAN_STOCKS.keys()),
                    format_func=lambda x: f"{INDIAN_STOCKS[x]}"
                )
            
            with trans_col2:
                trans_type = st.selectbox("Type", ["Buy", "Sell"])
            
            with trans_col3:
                trans_quantity = st.number_input("Quantity", min_value=1, value=1)
            
            with trans_col4:
                trans_price = st.number_input("Price per share", min_value=0.01, value=100.00)
            
            if st.button("Add Transaction"):
                conn = sqlite3.connect('financial_dashboard.db')
                c = conn.cursor()
                
                # Check if stock exists in portfolio
                c.execute("SELECT quantity, avg_price FROM portfolio WHERE user_id=? AND symbol=?",
                         (st.session_state.user[0], trans_symbol))
                existing = c.fetchone()
                
                if trans_type == "Buy":
                    if existing:
                        # Update existing position
                        new_quantity = existing[0] + trans_quantity
                        new_avg_price = ((existing[0] * existing[1]) + (trans_quantity * trans_price)) / new_quantity
                        
                        c.execute("UPDATE portfolio SET quantity=?, avg_price=? WHERE user_id=? AND symbol=?",
                                 (new_quantity, new_avg_price, st.session_state.user[0], trans_symbol))
                    else:
                        # Create new position
                        c.execute("INSERT INTO portfolio (user_id, symbol, quantity, avg_price, transaction_date) VALUES (?, ?, ?, ?, ?)",
                                 (st.session_state.user[0], trans_symbol, trans_quantity, trans_price, datetime.now()))
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
                st.success("Transaction added successfully!")
                st.rerun()
        
        # Portfolio Holdings
        st.subheader("Current Holdings")
        
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
            
            for symbol, quantity, avg_price in holdings:
                stock_data = get_stock_data(symbol, '1d')
                if stock_data:
                    current_price = stock_data['current_price']
                    current_value = quantity * current_price
                    cost_basis = quantity * avg_price
                    profit_loss = current_value - cost_basis
                    profit_loss_pct = (profit_loss / cost_basis) * 100
                    
                    total_value += current_value
                    total_cost += cost_basis
                    
                    portfolio_data.append({
                        'Symbol': symbol,
                        'Name': INDIAN_STOCKS.get(symbol, symbol),
                        'Quantity': quantity,
                        'Avg Price': f"₹{avg_price:.2f}",
                        'Current Price': f"₹{current_price:.2f}",
                        'Current Value': f"₹{current_value:,.2f}",
                        'P&L': f"₹{profit_loss:,.2f}",
                        'P&L %': f"{profit_loss_pct:.2f}%"
                    })
            
            df_portfolio = pd.DataFrame(portfolio_data)
            st.dataframe(df_portfolio, use_container_width=True)
            
            # Portfolio Summary
            st.subheader("Portfolio Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            total_pl = total_value - total_cost
            total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
            
            col1.metric("Total Value", f"₹{total_value:,.2f}")
            col2.metric("Total Cost", f"₹{total_cost:,.2f}")
            col3.metric("Total P&L", f"₹{total_pl:,.2f}", f"{total_pl_pct:.2f}%")
            col4.metric("Holdings", len(holdings))
            
            # Portfolio Pie Chart
            if len(portfolio_data) > 0:
                fig = px.pie(
                    values=[float(d['Current Value'].replace('₹', '').replace(',', '')) for d in portfolio_data],
                    names=[d['Name'] for d in portfolio_data],
                    title="Portfolio Allocation"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No holdings yet. Add your first transaction above!")
    else:
        st.warning("Please login to manage your portfolio")

# Tab 4: Technical Analysis
with tab4:
    st.header("Technical Analysis")
    
    if stock_data:
        # Calculate indicators
        df_ta = stock_data['history'].copy()
        df_ta = calculate_technical_indicators(df_ta)
        
        # Technical chart
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_ta.index,
            open=df_ta['Open'],
            high=df_ta['High'],
            low=df_ta['Low'],
            close=df_ta['Close'],
            name='Price'
        ))
        
        # Moving averages
        fig.add_trace(go.Scatter(
            x=df_ta.index,
            y=df_ta['SMA_20'],
            name='SMA 20',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_ta.index,
            y=df_ta['SMA_50'],
            name='SMA 50',
            line=dict(color='blue', width=1)
        ))
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df_ta.index,
            y=df_ta['BB_upper'],
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df_ta.index,
            y=df_ta['BB_lower'],
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.2)'
        ))
        
        fig.update_layout(
            title="Technical Analysis Chart",
            yaxis_title="Price (₹)",
            height=600,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical Indicators
        col1, col2, col3 = st.columns(3)
        
        latest_rsi = df_ta['RSI'].iloc[-1] if not df_ta['RSI'].isna().all() else 50
        latest_macd = df_ta['MACD'].iloc[-1] if not df_ta['MACD'].isna().all() else 0
        latest_signal = df_ta['Signal'].iloc[-1] if not df_ta['Signal'].isna().all() else 0
        
        with col1:
            st.metric("RSI (14)", f"{latest_rsi:.2f}")
            if latest_rsi > 70:
                st.warning("Overbought")
            elif latest_rsi < 30:
                st.success("Oversold")
            else:
                st.info("Neutral")
        
        with col2:
            st.metric("MACD", f"{latest_macd:.2f}")
            if latest_macd > latest_signal:
                st.success("Bullish Signal")
            else:
                st.warning("Bearish Signal")
        
        with col3:
            current_price = df_ta['Close'].iloc[-1]
            sma_20 = df_ta['SMA_20'].iloc[-1] if not df_ta['SMA_20'].isna().all() else current_price
            
            if current_price > sma_20:
                st.success("Price above SMA 20")
            else:
                st.warning("Price below SMA 20")

# Tab 5: DCF Calculator
with tab5:
    st.header("DCF Valuation Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        fcf = st.number_input("Free Cash Flow (₹ Crores)", value=1000.0, step=100.0)
        growth_rate = st.slider("Growth Rate (%)", 0.0, 30.0, 15.0, 0.5)
        terminal_growth = st.slider("Terminal Growth Rate (%)", 0.0, 10.0, 3.0, 0.5)
        discount_rate = st.slider("Discount Rate (WACC) %", 5.0, 20.0, 12.0, 0.5)
        shares_outstanding = st.number_input("Shares Outstanding (Crores)", value=100.0, step=10.0)
        
        if st.button("Calculate DCF Value"):
            dcf_value = calculate_dcf(
                fcf, 
                growth_rate/100, 
                terminal_growth/100, 
                discount_rate/100
            )
            
            fair_value_per_share = (dcf_value / shares_outstanding) * 10  # Convert to per share
            
            st.session_state.dcf_result = {
                'enterprise_value': dcf_value,
                'fair_value': fair_value_per_share
            }
    
    with col2:
        st.subheader("Valuation Results")
        
        if 'dcf_result' in st.session_state:
            st.metric("Enterprise Value", f"₹{st.session_state.dcf_result['enterprise_value']:,.2f} Cr")
            st.metric("Fair Value per Share", f"₹{st.session_state.dcf_result['fair_value']:,.2f}")
            
            if stock_data:
                current_price = stock_data['current_price']
                upside = ((st.session_state.dcf_result['fair_value'] - current_price) / current_price) * 100
                
                st.metric("Current Price", f"₹{current_price:.2f}")
                st.metric(
                    "Upside/Downside", 
                    f"{upside:.2f}%",
                    delta=f"₹{st.session_state.dcf_result['fair_value'] - current_price:.2f}"
                )
                
                if upside > 20:
                    st.success("Stock appears undervalued")
                elif upside < -20:
                    st.warning("Stock appears overvalued")
                else:
                    st.info("Stock appears fairly valued")

# Tab 6: AI Insights
with tab6:
    st.header("AI-Powered Insights")
    
    if stock_data:
        st.subheader(f"Analysis for {stock_data['name']}")
        
        # Generate insights based on data
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Technical Analysis Summary**")
            
            # RSI Analysis
            if 'dcf_result' not in st.session_state:
                st.info("Run DCF calculator for valuation insights")
            else:
                latest_rsi = 50  # Default
                if stock_data and len(stock_data['history']) > 14:
                    df_temp = calculate_technical_indicators(stock_data['history'])
                    latest_rsi = df_temp['RSI'].iloc[-1] if not df_temp['RSI'].isna().all() else 50
                
                insights = []
                
                if latest_rsi > 70:
                    insights.append("⚠️ RSI indicates overbought conditions")
                elif latest_rsi < 30:
                    insights.append("✅ RSI indicates oversold conditions - potential buying opportunity")
                else:
                    insights.append("📊 RSI in neutral zone")
                
                # Price trend
                if stock_data['change_percent'] > 2:
                    insights.append("📈 Strong upward momentum today")
                elif stock_data['change_percent'] < -2:
                    insights.append("📉 Significant downward pressure")
                
                # Valuation
                if 'dcf_result' in st.session_state:
                    upside = ((st.session_state.dcf_result['fair_value'] - stock_data['current_price']) / stock_data['current_price']) * 100
                    if upside > 20:
                        insights.append("💎 DCF model suggests significant undervaluation")
                    elif upside < -20:
                        insights.append("⚡ DCF model indicates overvaluation")
                
                for insight in insights:
                    st.write(insight)
        
        with col2:
            st.write("**Fundamental Analysis Summary**")
            
            fundamentals = get_fundamental_analysis(selected_stock)
            
            fund_insights = []
            
            if fundamentals.get('P/E Ratio', 0) > 0:
                if fundamentals['P/E Ratio'] < 15:
                    fund_insights.append("✅ Low P/E ratio - potentially undervalued")
                elif fundamentals['P/E Ratio'] > 30:
                    fund_insights.append("⚠️ High P/E ratio - growth priced in")
            
            if fundamentals.get('ROE', 0) > 15:
                fund_insights.append("💪 Strong ROE indicates efficient management")
            
            if fundamentals.get('Debt to Equity', 0) < 0.5:
                fund_insights.append("🛡️ Low debt levels - financially stable")
            elif fundamentals.get('Debt to Equity', 0) > 1.5:
                fund_insights.append("⚠️ High debt levels - monitor closely")
            
            for insight in fund_insights:
                st.write(insight)
        
        # AI Recommendation
        st.subheader("AI Recommendation")
        
        # Calculate overall score
        score = 50  # Base score
        
        # Technical factors
        if latest_rsi < 30:
            score += 10
        elif latest_rsi > 70:
            score -= 10
        
        # Fundamental factors
        if fundamentals.get('P/E Ratio', 25) < 20:
            score += 10
        if fundamentals.get('ROE', 0) > 15:
            score += 10
        
        # Valuation
        if 'dcf_result' in st.session_state:
            upside = ((st.session_state.dcf_result['fair_value'] - stock_data['current_price']) / stock_data['current_price']) * 100
            if upside > 20:
                score += 20
            elif upside < -20:
                score -= 20
        
        # Display recommendation
        if score >= 70:
            st.success(f"🟢 STRONG BUY - Score: {score}/100")
            st.write("The stock shows strong fundamentals, attractive valuation, and positive technical indicators.")
        elif score >= 55:
            st.info(f"🔵 BUY - Score: {score}/100")
            st.write("The stock presents a good investment opportunity with favorable risk-reward ratio.")
        elif score >= 45:
            st.warning(f"🟡 HOLD - Score: {score}/100")
            st.write("Mixed signals suggest waiting for better entry points or holding existing positions.")
        else:
            st.error(f"🔴 SELL/AVOID - Score: {score}/100")
            st.write("Multiple negative factors suggest avoiding or reducing exposure to this stock.")

# Footer
st.markdown("---")
st.markdown("*Data provided by Yahoo Finance (Free API). For educational purposes only. Not investment advice.*")

# Add custom CSS for better styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)