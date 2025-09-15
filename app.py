import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import faiss
import matplotlib.pyplot as plt
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from dotenv import load_dotenv
from datetime import datetime
from yahoo_fin import news as yf_news
from sentence_transformers import SentenceTransformer

# -------------------- ENV SETUP --------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("🚨 Please add GEMINI_API_KEY in .env file")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="VeloraAI", layout="wide")

# Custom CSS for centering
st.markdown(
    """
    <style>
    .centered {
        text-align: center;
    }
    .subtext {
        font-size:18px;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Centered content with tagline
st.markdown('<h1 class="centered">💹✨ VeloraAI ✨💹</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="centered">Real-Time Market Pulse 🌍, AI-Powered Insights 🤖, Sentiment Tracking 💡, and Intelligent Strategy Advisory 🚀 for Smarter Financial Decisions 💹</h3>', unsafe_allow_html=True)
st.markdown("---")

# -------------------- GLOBAL MODELS --------------------
@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

tokenizer, finbert_model = load_sentiment_model()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- FUNCTIONS --------------------
def fetch_stock_data(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_bollinger(series, window=20):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    high = rolling_mean + (2 * rolling_std)
    low = rolling_mean - (2 * rolling_std)
    return high, low

def compute_indicators(df):
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['Signal'] = compute_macd(df['Close'])
    df['Bollinger High'], df['Bollinger Low'] = compute_bollinger(df['Close'])
    return df

def plot_stock_data(df, ticker):
    st.line_chart(df['Close'], height=300, use_container_width=True)
    st.write(f"Latest Close Price: {df['Close'].iloc[-1]:.2f} USD")

def sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = finbert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["Negative", "Neutral", "Positive"]
    return labels[probs.argmax().item()], probs.max().item()

def ask_gemini(query, context=""):
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(f"Context: {context}\n\nQuestion: {query}")
    return response.text

def get_stock_news(ticker="AAPL", limit=5):
    try:
        return yf_news.get_yf_rss(ticker)[:limit]
    except Exception as e:
        return [{"title": f"⚠️ Could not fetch news ({str(e)})"}]

# -------------------- FAISS VECTOR DB --------------------
class NewsVectorDB:
    def __init__(self):
        self.index = None
        self.news_texts = []

    def build_index(self, news_list):
        embeddings = embedder.encode(news_list)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))
        self.news_texts = news_list

    def query(self, text, k=3):
        query_emb = embedder.encode([text])
        D, I = self.index.search(np.array(query_emb), k)
        return [self.news_texts[i] for i in I[0]]

# -------------------- NAVIGATION --------------------
menu = [
    "🏠 Market Pulse",
    "📊 Equity Explorer",
    "📰 Smart News Radar",
    "💹 Market Sentiment Lens",
    "🤖 AI Insights"
]
choice = st.sidebar.radio("**Navigation**", menu)

# -------------------- PAGES --------------------
if choice == "🏠 Market Pulse":
    st.header("🌍 Market Pulse")
    st.write("**Overview of top stocks and latest financial news.**")

    top_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    cols = st.columns(len(top_tickers))
    for i, ticker in enumerate(top_tickers):
        with cols[i]:
            try:
                data = fetch_stock_data(ticker, "1mo")
                st.metric(label=f"{ticker} (Close)", value=f"${data['Close'].iloc[-1]:.2f}")
            except:
                st.error(f"{ticker} data unavailable")

    st.markdown("### 📰 Latest Financial News")
    news_items = get_stock_news("AAPL", limit=5)
    for n in news_items:
        if "link" in n:
            st.markdown(f"- [{n['title']}]({n['link']})")
        else:
            st.write(f"- {n['title']}")

elif choice == "📊 Equity Explorer":
    st.header("📊 Equity Explorer")
    st.write("**Visualize stock prices, trends, and technical indicators with AI insights.**")

    # Hybrid dropdown + text input
    popular_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "JPM", "GS"]
    selected = st.selectbox("**Choose Stock Ticker or Custom**", ["Custom"] + popular_tickers, index=1)

    if selected == "Custom":
        ticker = st.text_input("Enter any valid ticker (e.g., INFY.NS, BTC-USD)", "AAPL").upper()
    else:
        ticker = selected

    try:
        data = fetch_stock_data(ticker)
        data = compute_indicators(data)

        st.subheader(f"{ticker} Price Chart")
        plot_stock_data(data, ticker)

        # Indicators with color-coded cues
        col1, col2, col3 = st.columns(3)

        rsi_latest = data['RSI'].iloc[-1]
        if rsi_latest > 70:
            rsi_status = f"🔴 Overbought ({rsi_latest:.2f})"
        elif rsi_latest < 30:
            rsi_status = f"🟢 Oversold ({rsi_latest:.2f})"
        else:
            rsi_status = f"🟡 Neutral ({rsi_latest:.2f})"

        macd_latest = data['MACD'].iloc[-1]
        signal_latest = data['Signal'].iloc[-1]
        macd_status = "🟢 Bullish" if macd_latest > signal_latest else "🔴 Bearish"

        close_latest = data['Close'].iloc[-1]
        upper_band = data['Bollinger High'].iloc[-1]
        lower_band = data['Bollinger Low'].iloc[-1]
        if close_latest > upper_band:
            bb_status = f"🔴 Above Upper Band ({close_latest:.2f})"
        elif close_latest < lower_band:
            bb_status = f"🟢 Below Lower Band ({close_latest:.2f})"
        else:
            bb_status = f"🟡 Within Bands ({close_latest:.2f})"

        col1.metric("RSI Indicator", rsi_status)
        col2.metric("MACD Signal", macd_status)
        col3.metric("Bollinger Bands", bb_status)

        query = st.text_area("**Ask about this stock**", "What is the growth outlook for this company?")
        if query:
            st.write(ask_gemini(query, context=f"Stock data: {ticker}, latest indicators."))

    except Exception as e:
        st.error(f"⚠️ Could not fetch stock data for {ticker}: {e}")

elif choice == "📰 Smart News Radar":
    st.header("📰 Smart News Radar")
    st.write("**Semantic search of financial news articles with AI context answers.**")

    ticker = st.text_input("**Enter ticker for news (e.g., AAPL, TSLA)**", "AAPL").upper()
    news_list = [n["title"] for n in get_stock_news(ticker, limit=10)]

    if news_list:
        db = NewsVectorDB()
        db.build_index(news_list)

        query = st.text_input("**Search financial topic**", "Apple supply chain risks")
        retrieved = db.query(query)

        st.subheader("**Relevant News**")
        for r in retrieved:
            st.write(f"- {r}")

        query_ai = st.text_area("**Ask about the news**", query)
        if query_ai:
            st.write(ask_gemini(query_ai, context=f"Relevant news: {retrieved}"))
    else:
        st.info("No recent news found for this ticker.")

elif choice == "💹 Market Sentiment Lens":
    st.header("💹 Market Sentiment Lens")
    st.write("**Analyze market sentiment from financial news and user input.**")

    user_text = st.text_area("**Enter financial text for sentiment analysis**")
    if user_text:
        label, confidence = sentiment_analysis(user_text)
        st.success(f"Sentiment: {label} ({confidence:.2f})")

    query_ai = st.text_area("**Ask about market sentiment**", "What is the sentiment on Tesla today?")
    if query_ai:
        st.write(ask_gemini(query_ai, context=f"Sentiment text: {user_text if user_text else 'None'}"))

elif choice == "🤖 AI Insights":
    st.header("🤖 AI Insights - AlphaMind")
    st.write("**Ask complex financial questions and get AI-powered insights using AlphaMind**")

    query = st.text_area("**Enter your financial question**")
    if st.button("Get AI Insight"):
        if query.strip():
            with st.spinner("Analyzing with AlphaMind..."):
                answer = ask_gemini(query, context="Finance, stocks, investment analysis")
            st.success(answer)
        else:
            st.error("Please enter a question.")
