# VeloraAI
AI-Powered Financial Insights
🚀 Velora is an intelligent financial research agent that brings together real-time stock data, AI-powered insights, sentiment analysis, and semantic news search into a single interactive dashboard.

Built with Streamlit, yFinance, FinBERT, FAISS, HuggingFace, and Gemini 2.5 Pro, Velora empowers analysts, investors, and enthusiasts to explore markets, track trends, and make smarter investment decisions.

# ✨ Features
📊 Equity Explorer

Real-time stock prices from Yahoo Finance

Advanced technical indicators: RSI, MACD, Bollinger Bands

AI-generated smart commentary on stock outlooks

📰 Smart News Radar

Live financial news headlines integration

Semantic search using FAISS + embeddings

Ask context-aware AI questions about market news

💹 Market Sentiment Lens

Sentiment analysis with FinBERT

Visual sentiment scoring for financial text

AI-driven insights into overall market tone

🤖 AI Insights (Gemini 2.5 Pro)

Query AI for investment strategies

Context-aware answers combining stock data + news

Helps bridge quantitative + qualitative analysis

🌍 Market Pulse (Home Page)

Curated overview of top trending tickers

Snapshot of key financial metrics

Live financial news feed integrated in real-time

#  📂 Project Structure
Velora_1/
│── app.py               # Main Streamlit app (monolithic version)
│── requirements.txt     # Project dependencies
│── .env                 # API keys (GEMINI_API_KEY)
│── venv/                # Virtual environment (ignored in Git)
└── README.md            # Documentation

#  ⚡ Quick Start
1️⃣ Clone the Repository
git clone https://github.com/your-username/velora.git
cd velora

#  2️⃣ Create Virtual Environment
python -m venv venv


#  Activate it:

Windows (PowerShell):

.\venv\Scripts\activate


Mac/Linux:

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add API Key

Create a .env file in the project root:

GEMINI_API_KEY=your_api_key_here

5️⃣ Run the App
streamlit run app.py

#  🛠️ Tech Stack

Frontend/UI: Streamlit, Matplotlib, Seaborn

Market Data: yFinance API (real-time stock prices), Yahoo Finance News

AI Models:

Gemini 2.5 Pro (Google GenAI)

HuggingFace Transformers

FinBERT (financial sentiment analysis)

Vector Search: FAISS + SentenceTransformers

Environment: Python 3.10+, dotenv for API key management


#  🚀 Future Enhancements

📌 Portfolio optimization tools (Sharpe Ratio, Efficient Frontier, VaR)

📌 Risk analysis dashboards

📌 Expanded news sources (Bloomberg, Reuters, CNBC)

📌 User portfolio tracking & price alerts

📌 Deployment on Streamlit Cloud / Docker for wider accessibility

#  🤝 Contributing

Contributions are welcome! 🎉

Fork the repository

Create a feature branch (git checkout -b feature-xyz)

Commit changes (git commit -m "Added feature xyz")

Push branch (git push origin feature-xyz)

Open a Pull Request



#  🙌 Acknowledgements

Streamlit
 for UI

yFinance
 for stock data

HuggingFace
 for Transformers & FinBERT

FAISS
 for vector search

Google Gemini
 for AI insights
