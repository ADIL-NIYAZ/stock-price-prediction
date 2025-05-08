from flask import Flask, request, jsonify, render_template
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import os
import logging
from alpha_vantage.timeseries import TimeSeries

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Alpha Vantage API key (stored securely via environment variable)
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'DXA8V0DBY29OE8C3')
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# NewsAPI key (stored securely via environment variable)
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '4d57cb75b7c4443b836bee326b524663')

# Mapping of company names to their stock symbols
stock_symbol_mapping = {
    'Apple': 'AAPL',
    'Tesla': 'TSLA',
    'Amazon': 'AMZN',
    'NVIDIA': 'NVDA',
    'Infosys': 'INFY',

}

# Function to fetch stock data
def fetch_stock_data(stock_symbol, period='1y'):
    try:
        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period=period)
        if hist.empty:
            raise ValueError("No data found with yfinance; attempting Alpha Vantage.")

        return hist

    except Exception as yf_error:
        app.logger.warning(f"yfinance failed for {stock_symbol}, attempting Alpha Vantage: {str(yf_error)}")
        try:
            data, meta_data = ts.get_daily_adjusted(symbol=stock_symbol, outputsize='compact')
            period_days = {'1y': 252, '5y': 1260, '10y': 2520}
            tail_size = period_days.get(period, 252)  # Default to 1 year
            data = data.tail(tail_size)
            data.rename(columns={'5. adjusted close': 'Close'}, inplace=True)
            return data[['Close']]

        except Exception as av_error:
            raise Exception(f"Error fetching stock data for {stock_symbol}: {str(av_error)}")

# Function to fetch latest news articles using NewsAPI
def get_latest_news(stock_symbol):
    try:
        url = f'https://newsapi.org/v2/everything?q={stock_symbol}&sortBy=publishedAt&apiKey={NEWS_API_KEY}'
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("Failed to fetch news articles.")

        articles_data = response.json()
        articles = [
            {'title': article['title'], 'link': article['url']}
            for article in articles_data['articles'][:5]  # Top 5 articles
        ]

        app.logger.info(f"Fetched articles for {stock_symbol}: {articles}")
        return articles
    except Exception as e:
        app.logger.error(f"Error fetching news articles: {e}")
        return []

# Function to get sentiment score
def analyze_sentiment(articles):
    if not articles:
        app.logger.info("No articles available for sentiment analysis.")
        return 0

    sentiment_score = sum(
        analyzer.polarity_scores(article['title'])['compound'] for article in articles
    )
    average_sentiment = sentiment_score / len(articles)
    app.logger.info(f"Average sentiment score: {average_sentiment}")
    return average_sentiment

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to get stock data and sentiment
@app.route('/predict', methods=['GET'])
def predict():
    company_name = request.args.get('stock', default='AAPL', type=str)
    period = request.args.get('period', default='1y', type=str)
    app.logger.info(f"Received company_name: {company_name}, period: {period}")

    stock_symbol = stock_symbol_mapping.get(company_name, company_name)

    try:
        # Fetch stock data
        stock_data = fetch_stock_data(stock_symbol, period)

        # Fetch latest news articles
        articles = get_latest_news(stock_symbol)

        # Analyze sentiment
        sentiment_score = analyze_sentiment(articles)

        # Determine recommendation
        if sentiment_score > 0.5:
            recommendation = "Buy"
        else:
            recommendation = "Don't Buy"

        # Generate stock chart
        plt.figure(figsize=(10, 5))
        stock_data['Close'].plot(title=f'{company_name} Stock Price', xlabel='Date', ylabel='Price ($)')
        plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)

        filename = f"stock_chart_{stock_symbol}_{period}.png"
        chart_path = os.path.join(app.static_folder, filename)
        plt.savefig(chart_path)
        plt.close()

        app.logger.info(f"Returning data: stock_symbol={stock_symbol}, sentiment_score={sentiment_score}, recommendation={recommendation}")

        # Return JSON response
        return jsonify({
            'stock_symbol': stock_symbol,
            'period': period,
            'sentiment_score': sentiment_score,
            'recommendation': recommendation,
            'articles': articles,
            'chart_url': f'/static/{filename}'
        })
    except Exception as e:
        app.logger.error(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
