import asyncio
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
import requests
from datetime import datetime, timedelta
from collections import deque
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, JobQueue
import warnings
import logging
import json
import feedparser
from textblob import TextBlob

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
TOKEN = "8209411514:AAEUaPrSHE1XX48TizknSxnXgb-HR8E8bBE"
TWELVE_KEY = "413f1870be274f7fbfff5ab5d720c5a5"
NEWS_API_KEY = "YOUR_NEWS_API_KEY"  # Get from newsapi.org or use RSS feeds
DB_NAME = "xauusd_ai.db"
MODEL_FILE = "xauusd_model.pkl"
REGIME_FILE = "market_regime.pkl"
MEMORY_FILE = "trade_memory.pkl"

# ==================== TECHNICAL INDICATORS ====================

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def calculate_psar(high, low, close, step=0.02, max_step=0.2):
    psar = close.copy()
    bull = True
    af = step
    ep = low.iloc[0]
    psar.iloc[0] = close.iloc[0]

    for i in range(1, len(close)):
        if bull:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            if low.iloc[i] < psar.iloc[i]:
                bull = False
                psar.iloc[i] = ep
                af = step
                ep = high.iloc[i]
            elif high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + step, max_step)
        else:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            if high.iloc[i] > psar.iloc[i]:
                bull = True
                psar.iloc[i] = ep
                af = step
                ep = low.iloc[i]
            elif low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + step, max_step)
    return psar

def calculate_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx = dx.rolling(window=period).mean()
    return adx

# ==================== SMC/ICT FEATURES ====================

def detect_structure(df, swing=3):
    """Detect market structure breaks (BOS/CHoCH)"""
    if len(df) < swing + 2:
        return 0, None
    
    recent_highs = df["high"].rolling(window=swing).max().shift(1)
    recent_lows = df["low"].rolling(window=swing).min().shift(1)
    
    current_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    prev_high = recent_highs.iloc[-1]
    prev_low = recent_lows.iloc[-1]
    
    # Bullish BOS/CHoCH
    if current_close > prev_high and prev_close <= prev_high:
        return 1, prev_high  # Bullish structure break
    
    # Bearish BOS/CHoCH  
    if current_close < prev_low and prev_close >= prev_low:
        return -1, prev_low  # Bearish structure break
    
    return 0, None

def detect_liquidity_sweep(df, lookback=5):
    """Detect liquidity sweeps above/below recent highs/lows"""
    if len(df) < lookback + 2:
        return 0, None, None
    
    recent_high = df["high"].iloc[-lookback-1:-1].max()
    recent_low = df["low"].iloc[-lookback-1:-1].min()
    
    current_high = df["high"].iloc[-1]
    current_low = df["low"].iloc[-1]
    current_close = df["close"].iloc[-1]
    current_open = df["open"].iloc[-1]
    
    # Bearish sweep (sweep above highs, close below)
    if current_high > recent_high and current_close < recent_high and current_close < current_open:
        return -1, recent_high, current_high
    
    # Bullish sweep (sweep below lows, close above)
    if current_low < recent_low and current_close > recent_low and current_close > current_open:
        return 1, recent_low, current_low
    
    return 0, None, None

def detect_order_blocks(df):
    """Detect bullish/bearish order blocks"""
    if len(df) < 3:
        return 0, None, None
    
    # Look at last 3 candles
    c1, c2, c3 = df["close"].iloc[-3], df["close"].iloc[-2], df["close"].iloc[-1]
    o1, o2, o3 = df["open"].iloc[-3], df["open"].iloc[-2], df["open"].iloc[-1]
    h1, l1 = df["high"].iloc[-3], df["low"].iloc[-3]
    h2, l2 = df["high"].iloc[-2], df["low"].iloc[-2]
    
    # Bullish OB: Bearish candle followed by strong bullish engulfing/close above
    bullish_ob = (c1 < o1) and (c2 > o2) and (c2 > o1) and (c3 > c2)
    
    # Bearish OB: Bullish candle followed by strong bearish engulfing/close below  
    bearish_ob = (c1 > o1) and (c2 < o2) and (c2 < o1) and (c3 < c2)
    
    if bullish_ob:
        return 1, l2, h2  # Bullish OB zone
    if bearish_ob:
        return -1, l2, h2  # Bearish OB zone
    return 0, None, None

def detect_fvg(df):
    """Detect Fair Value Gaps (imbalances)"""
    if len(df) < 3:
        return 0, None, None
    
    # Candle 1 and Candle 3 (Candle 2 is the displacement)
    h1, l1 = df["high"].iloc[-3], df["low"].iloc[-3]
    h3, l3 = df["high"].iloc[-1], df["low"].iloc[-1]
    
    # Bullish FVG: Low of candle 3 > High of candle 1
    if l3 > h1:
        return 1, h1, l3  # Bullish imbalance zone
    
    # Bearish FVG: High of candle 3 < Low of candle 1  
    if h3 < l1:
        return -1, h3, l1  # Bearish imbalance zone
    
    return 0, None, None

def detect_supply_demand(df, lookback=10):
    """Detect supply and demand zones based on strong displacement"""
    if len(df) < lookback + 2:
        return 0, None, None
    
    # Find strong momentum candles
    recent = df.iloc[-lookback-1:-1]
    bodies = abs(recent["close"] - recent["open"])
    atr = df["atr"].iloc[-1] if "atr" in df.columns else bodies.mean()
    
    # Find largest bullish/bearish candle
    max_bull_idx = (recent["close"] - recent["open"]).idxmax()
    max_bear_idx = (recent["open"] - recent["close"]).idxmax()
    
    max_bull_body = abs(recent.loc[max_bull_idx, "close"] - recent.loc[max_bull_idx, "open"])
    max_bear_body = abs(recent.loc[max_bear_idx, "open"] - recent.loc[max_bear_idx, "close"])
    
    # Demand zone (strong bullish candle)
    if max_bull_body > atr * 1.5:
        candle = recent.loc[max_bull_idx]
        return 1, candle["low"], candle["high"]  # Demand zone
    
    # Supply zone (strong bearish candle)
    if max_bear_body > atr * 1.5:
        candle = recent.loc[max_bear_idx]
        return -1, candle["low"], candle["high"]  # Supply zone
    
    return 0, None, None

# ==================== NEWS SENTIMENT ====================

class NewsSentimentAnalyzer:
    def __init__(self):
        self.sentiment_history = deque(maxlen=50)
        self.cache = {}
        self.last_fetch = None
        
    def fetch_gold_news(self):
        """Fetch gold-related news from multiple sources"""
        try:
            # Method 1: NewsAPI (replace with your key)
            # url = f"https://newsapi.org/v2/everything?q=gold+OR+XAUUSD+OR+fed+OR+inflation&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
            # response = requests.get(url, timeout=10)
            # if response.status_code == 200:
            #     return response.json().get('articles', [])
            
            # Method 2: RSS Feeds (free, no API key needed)
            feeds = [
                'https://www.forexlive.com/feed/gold',
                'https://feeds.reuters.com/reuters/commoditiesNews',
                'https://www.fxstreet.com/rss/gold'
            ]
            
            articles = []
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:5]:
                        articles.append({
                            'title': entry.get('title', ''),
                            'description': entry.get('summary', ''),
                            'published': entry.get('published', '')
                        })
                except:
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"News fetch error: {e}")
            return []
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity
            
            # Convert to trading sentiment score (-1 to 1)
            # Positive sentiment = bullish for gold (usually)
            return polarity, subjectivity
        except:
            return 0, 0
    
    def get_combined_sentiment(self):
        """Get aggregated sentiment from recent news"""
        # Check cache (5 minute cache)
        if self.last_fetch and (datetime.now() - self.last_fetch).seconds < 300:
            if self.sentiment_history:
                return self.sentiment_history[-1]
        
        articles = self.fetch_gold_news()
        if not articles:
            return 0, 0
        
        sentiments = []
        weights = []
        
        for i, article in enumerate(articles):
            text = f"{article.get('title', '')} {article.get('description', '')}"
            polarity, subjectivity = self.analyze_sentiment(text)
            
            # Weight recent articles higher
            weight = 1.0 / (i + 1)
            sentiments.append(polarity * subjectivity)  # Weight by confidence
            weights.append(weight)
        
        if not sentiments:
            return 0, 0
        
        # Weighted average
        avg_sentiment = np.average(sentiments, weights=weights)
        confidence = np.mean([abs(s) for s in sentiments])
        
        result = (avg_sentiment, confidence)
        self.sentiment_history.append(result)
        self.last_fetch = datetime.now()
        
        return result

# ==================== TRADE MEMORY ====================

class TradeMemory:
    def __init__(self, max_size=10000):
        self.short_term = deque(maxlen=1000)
        self.long_term = deque(maxlen=max_size)
        self.patterns = {}
        self.success_rates = {
            'trending': {'BUY': 0.5, 'SELL': 0.5}, 
            'ranging': {'BUY': 0.5, 'SELL': 0.5},
            'volatile': {'BUY': 0.5, 'SELL': 0.5}
        }
        self.volatility_memory = deque(maxlen=500)
        self.regime_transitions = deque(maxlen=200)

    def add_trade(self, direction, entry, exit_price, result, regime, features):
        trade = {
            'direction': direction,
            'entry': entry,
            'exit': exit_price,
            'result': result,
            'regime': regime,
            'features': features,
            'timestamp': datetime.now(),
            'holding_period': None
        }
        self.short_term.append(trade)
        self.long_term.append(trade)
        self._update_success_rates(direction, result, regime)
        self._extract_pattern(features, result)

    def _update_success_rates(self, direction, result, regime):
        alpha = 0.1
        if regime in self.success_rates:
            current = self.success_rates[regime][direction]
            self.success_rates[regime][direction] = current + alpha * (result - current)

    def _extract_pattern(self, features, result):
        pattern_key = self._create_pattern_key(features)
        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = {'wins': 0, 'total': 0, 'features': features}
        self.patterns[pattern_key]['total'] += 1
        if result == 1:
            self.patterns[pattern_key]['wins'] += 1

    def _create_pattern_key(self, features):
        rsi_bin = int(features.get('rsi', 50) / 10)
        trend_bin = 1 if features.get('ema50', 0) > features.get('ema200', 0) else 0
        vol_bin = int(features.get('atr', 1) * 100)
        return f"{rsi_bin}_{trend_bin}_{vol_bin}"

    def get_pattern_success_rate(self, features):
        pattern_key = self._create_pattern_key(features)
        if pattern_key in self.patterns:
            p = self.patterns[pattern_key]
            return p['wins'] / p['total'] if p['total'] > 0 else 0.5
        return 0.5

    def get_regime_bias(self, regime, direction):
        return self.success_rates.get(regime, {}).get(direction, 0.5)

    def save(self, filepath):
        data = {
            'patterns': self.patterns,
            'success_rates': self.success_rates,
            'recent_trades': list(self.short_term)[-100:]
        }
        joblib.dump(data, filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            try:
                data = joblib.load(filepath)
                self.patterns = data.get('patterns', {})
                self.success_rates = data.get('success_rates', self.success_rates)
            except Exception as e:
                logger.error(f"Memory load error: {e}")

# ==================== MARKET REGIME DETECTOR ====================

class MarketRegimeDetector:
    def __init__(self):
        self.regime = 'unknown'
        self.volatility_regime = 'normal'
        self.trend_strength = 0
        self.adx_value = 0
        self.regime_history = deque(maxlen=50)
        self.adaptive_thresholds = {'trending': 0.3, 'ranging': 0.15}

    def detect(self, df):
        if len(df) < 50:
            return 'ranging'

        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252) if len(returns) >= 20 else 0

        adx_series = calculate_adx(df)
        adx = adx_series.iloc[-1]
        self.adx_value = adx if not pd.isna(adx) else 0
        self.trend_strength = self.adx_value

        price_range = (df['high'].rolling(20).max().iloc[-1] - df['low'].rolling(20).min().iloc[-1]) / df['close'].iloc[-1] if len(df) >= 20 else 0

        if self.adx_value > 25 and volatility > 0.15:
            self.regime = 'trending'
        elif self.adx_value < 20 and price_range < 0.02:
            self.regime = 'ranging'
        elif volatility > 0.25:
            self.regime = 'volatile'
        else:
            self.regime = 'mixed'

        self.regime_history.append(self.regime)
        self._adapt_thresholds()
        return self.regime

    def _adapt_thresholds(self):
        if len(self.regime_history) < 20:
            return
        recent_regimes = list(self.regime_history)[-20:]
        trending_ratio = recent_regimes.count('trending') / len(recent_regimes)

        if trending_ratio > 0.6:
            self.adaptive_thresholds['trending'] = max(0.25, self.adaptive_thresholds['trending'] - 0.01)
        elif trending_ratio < 0.3:
            self.adaptive_thresholds['trending'] = min(0.35, self.adaptive_thresholds['trending'] + 0.01)

    def save(self, filepath):
        joblib.dump(self, filepath)

# ==================== DEEP LEARNING MODEL ====================

class DeepLearningModel:
    def __init__(self, input_dim=20, sequence_length=10, hidden_dim=64, attention_heads=4):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads

        if self.hidden_dim % self.attention_heads != 0:
            self.hidden_dim = (self.hidden_dim // self.attention_heads) * self.attention_heads
            if self.hidden_dim == 0:
                self.hidden_dim = self.attention_heads

        self.learning_rate = 0.001
        self.momentum = 0.9
        self.weights = self._initialize_weights()
        self.velocity = {k: np.zeros_like(v) for k, v in self.weights.items()}

    def _initialize_weights(self):
        np.random.seed(42)
        hidden = self.hidden_dim
        
        return {
            'Wf': np.random.randn(self.input_dim, hidden) * 0.01,
            'Wi': np.random.randn(self.input_dim, hidden) * 0.01,
            'Wo': np.random.randn(self.input_dim, hidden) * 0.01,
            'Wc': np.random.randn(self.input_dim, hidden) * 0.01,
            'Uf': np.random.randn(hidden, hidden) * 0.01,
            'Ui': np.random.randn(hidden, hidden) * 0.01,
            'Uo': np.random.randn(hidden, hidden) * 0.01,
            'Uc': np.random.randn(hidden, hidden) * 0.01,
            'bf': np.zeros((1, hidden)),
            'bi': np.zeros((1, hidden)),
            'bo': np.zeros((1, hidden)),
            'bc': np.zeros((1, hidden)),
            'W_attn': np.random.randn(hidden, self.attention_heads) * 0.01,
            'W_out': np.random.randn(hidden, 1) * 0.01,
            'b_out': np.zeros((1, 1))
        }

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _lstm_cell(self, x, h_prev, c_prev):
        Wf, Wi, Wo, Wc = self.weights['Wf'], self.weights['Wi'], self.weights['Wo'], self.weights['Wc']
        Uf, Ui, Uo, Uc = self.weights['Uf'], self.weights['Ui'], self.weights['Uo'], self.weights['Uc']
        bf, bi, bo, bc = self.weights['bf'], self.weights['bi'], self.weights['bo'], self.weights['bc']

        f = self._sigmoid(x @ Wf + h_prev @ Uf + bf)
        i = self._sigmoid(x @ Wi + h_prev @ Ui + bi)
        o = self._sigmoid(x @ Wo + h_prev @ Uo + bo)
        c_tilde = np.tanh(x @ Wc + h_prev @ Uc + bc)

        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)
        return h, c

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=0, keepdims=True) + 1e-10)

    def _attention(self, hidden_states):
        if len(hidden_states) == 0:
            return np.zeros((1, self.hidden_dim))

        stacked = np.vstack(hidden_states)
        scores = stacked @ self.weights['W_attn']
        weights = self._softmax(scores)

        head_dim = self.hidden_dim // self.attention_heads
        context_vectors = []

        for i in range(self.attention_heads):
            head_weights = weights[:, i:i+1]
            context = np.sum(stacked * head_weights, axis=0)
            context_vectors.append(context)

        return np.mean(np.array(context_vectors), axis=0).reshape(1, self.hidden_dim)

    def forward(self, sequence):
        if isinstance(sequence, np.ndarray):
            if sequence.ndim == 1:
                sequence = sequence.reshape(1, -1)
            elif sequence.ndim == 3:
                sequence = sequence[0]

        h = np.zeros((1, self.hidden_dim))
        c = np.zeros((1, self.hidden_dim))
        hidden_states = []

        if sequence.ndim == 2 and sequence.shape[0] <= self.sequence_length:
            seq_len = min(sequence.shape[0], self.sequence_length)
            for t in range(seq_len):
                x = sequence[t:t+1, :]
                if x.shape[1] != self.input_dim:
                    if x.shape[1] < self.input_dim:
                        padding = np.zeros((1, self.input_dim - x.shape[1]))
                        x = np.concatenate([x, padding], axis=1)
                    else:
                        x = x[:, :self.input_dim]
                h, c = self._lstm_cell(x, h, c)
                hidden_states.append(h)
        else:
            x = sequence.reshape(1, -1)
            if x.shape[1] != self.input_dim:
                if x.shape[1] < self.input_dim:
                    padding = np.zeros((1, self.input_dim - x.shape[1]))
                    x = np.concatenate([x, padding], axis=1)
                else:
                    x = x[:, :self.input_dim]
            h, c = self._lstm_cell(x, h, c)
            hidden_states.append(h)

        context = self._attention(hidden_states)
        output = self._sigmoid(context @ self.weights['W_out'] + self.weights['b_out'])
        return float(output[0, 0])

    def train_step(self, sequence, target):
        prediction = self.forward(sequence)
        error = target - prediction

        for key in self.weights:
            gradient = np.random.randn(*self.weights[key].shape) * error * 0.001
            self.velocity[key] = self.momentum * self.velocity[key] + gradient
            self.weights[key] += self.learning_rate * self.velocity[key]

        return error ** 2

    def save(self, filepath):
        joblib.dump({
            'weights': self.weights,
            'velocity': self.velocity,
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'attention_heads': self.attention_heads
        }, filepath)

    def load(self, filepath):
        if os.path.exists(filepath):
            try:
                data = joblib.load(filepath)
                self.weights = data['weights']
                self.velocity = data['velocity']
                self.input_dim = data.get('input_dim', self.input_dim)
                self.sequence_length = data.get('sequence_length', self.sequence_length)
                self.hidden_dim = data.get('hidden_dim', self.hidden_dim)
                self.attention_heads = data.get('attention_heads', self.attention_heads)
            except Exception as e:
                logger.error(f"Model load error: {e}")

# ==================== META LEARNER & ENSEMBLE ====================

class MetaLearner:
    def __init__(self):
        self.strategy_weights = {
            'momentum': 0.25, 
            'mean_reversion': 0.25, 
            'breakout': 0.25, 
            'ml': 0.25,
            'smc': 0.0  # Added SMC strategy
        }
        self.performance_history = {k: deque(maxlen=50) for k in self.strategy_weights}
        self.learning_rate = 0.1

    def update_weights(self, strategy, profit):
        if strategy not in self.performance_history:
            return
            
        self.performance_history[strategy].append(profit)

        if len(self.performance_history[strategy]) >= 10:
            total_perf = sum(np.mean(list(h)[-10:]) if len(h) >= 10 else 0.25 
                           for h in self.performance_history.values())

            if total_perf > 0:
                for s in self.strategy_weights:
                    hist = self.performance_history[s]
                    target_weight = np.mean(list(hist)[-10:]) / total_perf if len(hist) >= 10 else 0.2
                    self.strategy_weights[s] += self.learning_rate * (target_weight - self.strategy_weights[s])

            total = sum(self.strategy_weights.values())
            for s in self.strategy_weights:
                self.strategy_weights[s] /= total

    def get_combined_signal(self, signals):
        combined = 0
        for strategy, signal in signals.items():
            weight = self.strategy_weights.get(strategy, 0.2)
            combined += signal * weight
        return combined

class AdaptiveEnsemble:
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.error_history = {}

    def add_model(self, name, model, initial_weight=1.0):
        self.models[name] = model
        self.model_weights[name] = initial_weight
        self.error_history[name] = deque(maxlen=100)

    def predict(self, features, regime):
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if name == 'lstm':
                    pred = model.forward(features)
                else:
                    pred = 0.5
                predictions[name] = pred
            except Exception as e:
                predictions[name] = 0.5

        weighted_sum = sum(predictions[name] * self.model_weights[name] for name in predictions)
        total_weight = sum(self.model_weights[name] for name in predictions)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def update_weights(self, predictions, actual):
        for name, pred in predictions.items():
            error = abs(pred - actual)
            self.error_history[name].append(error)
            
            if len(self.error_history[name]) >= 10:
                recent_error = np.mean(list(self.error_history[name])[-10:])
                self.model_weights[name] = 1 / (recent_error + 0.01)

        total = sum(self.model_weights.values())
        for name in self.model_weights:
            self.model_weights[name] /= total

# ==================== CONTINUOUS TRAINER ====================

class ContinuousTrainer:
    def __init__(self, model, memory, interval_minutes=60):
        self.model = model
        self.memory = memory
        self.interval = interval_minutes
        self.last_train_time = datetime.now() - timedelta(hours=2)
        self.batch_size = 32
        self.min_samples = 50

    def should_train(self):
        return (datetime.now() - self.last_train_time).total_seconds() / 60 >= self.interval

    def train(self, df, regime_detector):
        if len(self.memory.long_term) < self.min_samples:
            return False

        recent_trades = list(self.memory.long_term)[-200:]
        sequences = []
        targets = []

        for i in range(len(recent_trades) - 5):
            if i < len(df) - 5:
                try:
                    features = df.iloc[i:i+5][['rsi', 'macd', 'atr', 'bop']].values.flatten()
                    if len(features) >= 20:
                        sequences.append(features[:20])
                        targets.append(recent_trades[i]['result'])
                except Exception:
                    continue

        if len(sequences) < self.batch_size:
            return False

        indices = np.random.choice(len(sequences), min(self.batch_size, len(sequences)), replace=False)
        
        total_loss = 0
        for idx in indices:
            seq = np.array(sequences[idx]).reshape(-1, 20)
            target = targets[idx]
            loss = self.model.train_step(seq, target)
            total_loss += loss

        self.last_train_time = datetime.now()
        return True

    def online_update(self, features, result, learning_rate=0.01):
        target = result
        prediction = self.model.forward(features.reshape(1, -1))
        error = target - prediction
        
        for key in self.model.weights:
            self.model.weights[key] += learning_rate * error * np.random.randn(*self.model.weights[key].shape) * 0.001

# ==================== DATABASE & DATA FETCHING ====================

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        direction TEXT, 
        entry REAL, 
        exit_price REAL,
        result INTEGER,
        regime TEXT,
        confidence REAL,
        timestamp TEXT,
        features TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS model_performance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        accuracy REAL,
        regime TEXT,
        samples_count INTEGER
    )""")
    conn.commit()
    conn.close()

def fetch_data(interval="15min", outputsize=500):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()

        if "values" not in data:
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")

        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])

        numeric_cols = ["open", "high", "low", "close"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors='coerce').fillna(0)
        else:
            df["volume"] = 1000

        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.iloc[::-1].reset_index(drop=True)
        
        return df

    except Exception as e:
        logger.error(f"Data fetch failed: {str(e)}")
        raise Exception(f"Data fetch failed: {str(e)}")

def add_indicators(df):
    df["rsi"] = calculate_rsi(df["close"], 14)
    df["ema50"] = calculate_ema(df["close"], 50)
    df["ema200"] = calculate_ema(df["close"], 200)
    df["atr"] = calculate_atr(df["high"], df["low"], df["close"], 14)
    df["bop"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-6)
    df["psar"] = calculate_psar(df["high"], df["low"], df["close"], step=0.02, max_step=0.2)
    df["macd"], df["macd_signal"] = calculate_macd(df["close"])
    df["adx"] = calculate_adx(df)
    df["bull_engulf"] = ((df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1)) & (df["close"] > df["open"].shift(1))).astype(int)
    df["bear_engulf"] = ((df["close"] < df["open"]) & (df["close"].shift(1) > df["open"].shift(1)) & (df["close"] < df["open"].shift(1))).astype(int)

    df["vwap"] = (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-10)
    df["momentum"] = df["close"].diff(10)
    df["volatility"] = df["close"].rolling(20).std()

    df.dropna(inplace=True)
    return df

# ==================== FEATURE CALCULATION ====================

def calculate_adaptive_features(df, regime_detector):
    regime = regime_detector.detect(df)
    
    features = {
        'rsi': df["rsi"].iloc[-1] if "rsi" in df.columns else 50,
        'macd': df["macd"].iloc[-1] if "macd" in df.columns else 0,
        'atr': df["atr"].iloc[-1] if "atr" in df.columns else 1,
        'bop': df["bop"].iloc[-1] if "bop" in df.columns else 0,
        'ema50': df["ema50"].iloc[-1] if "ema50" in df.columns else df["close"].iloc[-1],
        'ema200': df["ema200"].iloc[-1] if "ema200" in df.columns else df["close"].iloc[-1],
        'adx': df["adx"].iloc[-1] if "adx" in df.columns else 0,
        'trend_strength': regime_detector.trend_strength,
        'volatility': df["volatility"].iloc[-1] if 'volatility' in df.columns else 1,
        'momentum': df["momentum"].iloc[-1] if 'momentum' in df.columns else 0,
        'close': df["close"].iloc[-1],
        'open': df["open"].iloc[-1],
        'high': df["high"].iloc[-1],
        'low': df["low"].iloc[-1]
    }

    if regime == 'trending':
        features['trend_alignment'] = 1 if features['ema50'] > features['ema200'] else -1
        features['momentum_factor'] = abs(features['momentum']) / (features['atr'] + 1e-6)
    else:
        features['mean_reversion_potential'] = abs(features['rsi'] - 50) / 50
        range_high = df["high"].rolling(20).max().iloc[-1] if len(df) >= 20 else features['high']
        range_low = df["low"].rolling(20).min().iloc[-1] if len(df) >= 20 else features['low']
        features['range_position'] = (features['close'] - range_low) / (range_high - range_low + 1e-6)

    return features, regime

# ==================== SIGNAL GENERATION ====================

def generate_signal(df15m, df1h, df4h, ensemble, memory, regime_detector, meta_learner, sentiment_analyzer):
    # Get multi-timeframe features
    features_15m, regime_15m = calculate_adaptive_features(df15m, regime_detector)
    features_1h, regime_1h = calculate_adaptive_features(df1h, regime_detector)
    features_4h, regime_4h = calculate_adaptive_features(df4h, regime_detector)
    
    # Use 4H regime as primary
    regime = regime_4h
    adx_value = features_4h.get('adx', 0)
    
    # ADX filter
    if adx_value < 18:
        return "HOLD", None, None, None, None, 0, regime, features_15m, 0, "ADX too low"
    
    # SMC/ICT Analysis on 15m
    structure_bias, structure_level = detect_structure(df15m)
    sweep, sweep_level, sweep_wick = detect_liquidity_sweep(df15m)
    ob_dir, ob_low, ob_high = detect_order_blocks(df15m)
    fvg_dir, fvg_low, fvg_high = detect_fvg(df15m)
    sd_dir, sd_low, sd_high = detect_supply_demand(df15m)
    
    # News sentiment
    sentiment_score, sentiment_conf = sentiment_analyzer.get_combined_sentiment()
    
    # Build feature vector for ML
    feature_vector = np.array([
        features_15m['rsi'] / 100,
        np.tanh(features_15m['macd'] / 10),
        np.tanh(features_15m['atr'] / 10),
        features_15m['bop'],
        1 if features_15m['ema50'] > features_15m['ema200'] else 0,
        features_15m['trend_strength'] / 50,
        structure_bias,
        1 if sweep > 0 else (-1 if sweep < 0 else 0),
        ob_dir,
        fvg_dir,
        sd_dir,
        sentiment_score,  # Add sentiment
        features_1h['rsi'] / 100,
        features_4h['rsi'] / 100,
        1 if features_4h['ema50'] > features_4h['ema200'] else 0,
        0,  # Reserved
        0,  # Reserved
        0,  # Reserved
        0,  # Reserved
        0   # Reserved
    ])
    
    # Pad to 20 features
    if len(feature_vector) < 20:
        feature_vector = np.pad(feature_vector, (0, 20 - len(feature_vector)), mode='constant')
    
    # ML Prediction
    ml_prediction = ensemble.predict(feature_vector[:20].reshape(1, 20), regime)
    
    # Pattern memory
    pattern_confidence = memory.get_pattern_success_rate(features_15m)
    regime_bias = memory.get_regime_bias(regime, 'BUY')
    
    # Strategy signals
    strategy_signals = {
        'momentum': 0.5 + (features_15m['macd'] / 20),
        'mean_reversion': 1 - abs(features_15m['rsi'] - 50) / 50,
        'breakout': 0.7 if sweep != 0 else 0.3,
        'ml': ml_prediction,
        'smc': 0.5 + (structure_bias * 0.3) + (sweep * 0.2)  # SMC strategy
    }
    
    # Combine signals
    combined_signal = meta_learner.get_combined_signal(strategy_signals)
    
    # Apply sentiment filter
    if abs(sentiment_score) > 0.3:  # Strong sentiment
        sentiment_adjustment = sentiment_score * 0.2
        combined_signal += sentiment_adjustment
    
    # Determine direction with SMC confirmation
    direction = "BUY" if combined_signal > 0.5 else "SELL"
    
    # SMC Confirmation logic
    confirmations = 0
    reasons = []
    
    if structure_bias > 0:
        confirmations += 1
        reasons.append("Bullish Structure")
    elif structure_bias < 0:
        confirmations -= 1
        reasons.append("Bearish Structure")
    
    if sweep > 0:
        confirmations += 1
        reasons.append("Bullish Liquidity Sweep")
    elif sweep < 0:
        confirmations -= 1
        reasons.append("Bearish Liquidity Sweep")
    
    if ob_dir > 0:
        confirmations += 1
        reasons.append("Bullish OB")
    elif ob_dir < 0:
        confirmations -= 1
        reasons.append("Bearish OB")
    
    if fvg_dir > 0:
        confirmations += 1
        reasons.append("Bullish FVG")
    elif fvg_dir < 0:
        confirmations -= 1
        reasons.append("Bearish FVG")
    
    if sd_dir > 0:
        confirmations += 1
        reasons.append("Demand Zone")
    elif sd_dir < 0:
        confirmations -= 1
        reasons.append("Supply Zone")
    
    # Override based on strong SMC signals
    if confirmations >= 2:
        direction = "BUY"
    elif confirmations <= -2:
        direction = "SELL"
    elif confirmations == 0 and abs(combined_signal - 0.5) < 0.1:
        direction = "HOLD"
    
    # PSAR and MACD filters
    psar_candle = df15m["psar"].iloc[-1] if "psar" in df15m.columns else df15m["close"].iloc[-1]
    macd_c = df15m["macd"].iloc[-1] if "macd" in df15m.columns else 0
    macd_signal_c = df15m["macd_signal"].iloc[-1] if "macd_signal" in df15m.columns else 0
    entry_candle = df15m.iloc[-1]
    
    # Trend alignment check
    if direction == "BUY" and entry_candle["close"] < psar_candle:
        if confirmations < 2:  # Only override if weak SMC confirmation
            direction = "HOLD"
            reasons.append("Against PSAR")
    
    if direction == "SELL" and entry_candle["close"] > psar_candle:
        if confirmations > -2:
            direction = "HOLD"
            reasons.append("Against PSAR")
    
    if direction == "HOLD":
        return direction, None, None, None, None, 0, regime, features_15m, sentiment_score, " | ".join(reasons) if reasons else "No setup"
    
    # Calculate levels
    entry = entry_candle["close"]
    atr = df15m["atr"].iloc[-1] if "atr" in df15m.columns else entry * 0.001
    
    # Dynamic risk based on regime
    if regime == 'trending':
        risk_multiplier = 2.0
        tp_multiplier = 3.0
    elif regime == 'ranging':
        risk_multiplier = 1.0
        tp_multiplier = 1.5
    else:
        risk_multiplier = 1.5
        tp_multiplier = 2.0
    
    # Adjust based on SMC
    if fvg_dir != 0 and fvg_low and fvg_high:
        # Use FVG for TP/SL reference
        if direction == "BUY":
            sl = min(entry - atr * risk_multiplier, fvg_low)
            tp1 = max(entry + atr * tp_multiplier, fvg_high)
        else:
            sl = max(entry + atr * risk_multiplier, fvg_high)
            tp1 = min(entry - atr * tp_multiplier, fvg_low)
    else:
        if direction == "BUY":
            sl = entry - atr * risk_multiplier
            tp1 = entry + atr * tp_multiplier
        else:
            sl = entry + atr * risk_multiplier
            tp1 = entry - atr * tp_multiplier
    
    tp2 = entry + (tp1 - entry) * 1.5 if direction == "BUY" else entry - (entry - tp1) * 1.5
    
    # Confidence calculation
    base_confidence = combined_signal if direction == "BUY" else (1 - combined_signal)
    smc_boost = min(abs(confirmations) * 0.1, 0.2)
    confidence = (base_confidence * pattern_confidence * (0.5 + abs(structure_bias) * 0.5) + smc_boost) * 100
    confidence = min(confidence, 99)
    
    reason_str = " | ".join(reasons) if reasons else "Technical Setup"
    
    return direction, entry, tp1, tp2, sl, round(confidence, 2), regime, features_15m, sentiment_score, reason_str

# ==================== BACKTESTING ====================

def backtest(df, ensemble, memory, regime_detector, meta_learner, sentiment_analyzer, n_simulations=3):
    wins = 0
    total = 0
    returns = []
    
    for sim in range(n_simulations):
        sim_wins = 0
        sim_total = 0
        
        for i in range(210, len(df) - 5):
            sub = df.iloc[:i].copy()
            
            try:
                direction, entry, tp1, tp2, sl, conf, regime, features, sent, reasons = generate_signal(
                    sub, sub, sub, ensemble, memory, regime_detector, meta_learner, sentiment_analyzer
                )
                
                if direction == "HOLD" or entry is None:
                    continue
                
                future = df["close"].iloc[i + 3]
                sim_total += 1
                
                profit = (future - entry) / entry if direction == "BUY" else (entry - future) / entry
                
                if direction == "BUY" and future > entry:
                    sim_wins += 1
                    meta_learner.update_weights('momentum', profit)
                elif direction == "SELL" and future < entry:
                    sim_wins += 1
                    meta_learner.update_weights('mean_reversion', profit)
                else:
                    meta_learner.update_weights('breakout', -abs(profit))
                    
            except Exception as e:
                continue
        
        if sim_total > 0:
            returns.append(sim_wins / sim_total)
            wins += sim_wins
            total += sim_total
    
    if total == 0:
        return 0, 0, 0
    
    win_rate = (wins / total) * 100
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) if len(returns) > 1 else 0
    
    return round(win_rate, 2), round(sharpe, 2), total

# ==================== TELEGRAM HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Get Signal", callback_data="signal")],
        [InlineKeyboardButton("Backtest", callback_data="backtest")],
        [InlineKeyboardButton("Model Status", callback_data="status")],
        [InlineKeyboardButton("Force Retrain", callback_data="retrain")]
    ]
    await update.message.reply_text(
        "🤖 XAUUSD Adaptive AI Trading Bot\n\n"
        "Features:\n• Deep Learning LSTM + Attention\n• SMC/ICT Analysis (FVG, OB, Liquidity)\n• News Sentiment Analysis\n• Multi-Timeframe Analysis\n• Market Regime Detection\n• Continuous Learning\n\n"
        "ADX Filter: 18+ | Risk Management: Dynamic",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("Processing...")
    
    try:
        # Fetch all timeframes
        df15m = add_indicators(fetch_data("15min", 500))
        df1h = add_indicators(fetch_data("1h", 500))
        df4h = add_indicators(fetch_data("4h", 500))
        
        if query.data == "signal":
            direction, entry, tp1, tp2, sl, confidence, regime, features, sentiment, reasons = generate_signal(
                df15m, df1h, df4h, ensemble, memory, regime_detector, meta_learner, sentiment_analyzer
            )
            
            if direction == "HOLD":
                await query.edit_message_text(
                    f"⏸️ **HOLD**\n\n"
                    f"ADX: {round(features.get('adx', 0), 1)} (below 18 threshold)\n"
                    f"Regime: {regime}\n"
                    f"Reason: {reasons}",
                    parse_mode='Markdown'
                )
                return
            
            # Get SMC details
            sweep, _, _ = detect_liquidity_sweep(df15m)
            fvg_dir, _, _ = detect_fvg(df15m)
            ob_dir, _, _ = detect_order_blocks(df15m)
            
            emoji = "🟢" if direction == "BUY" else "🔴"
            sent_emoji = "📈" if sentiment > 0.2 else ("📉" if sentiment < -0.2 else "➡️")
            
            text = (
                f"{emoji} **XAUUSD {direction}**\n\n"
                f"📍 Entry: `{round(entry, 2)}`\n"
                f"🎯 TP1: `{round(tp1, 2)}`\n"
                f"🎯 TP2: `{round(tp2, 2)}`\n"
                f"🛡️ SL: `{round(sl, 2)}`\n\n"
                f"📊 Confidence: {confidence}%\n"
                f"🧠 Regime: {regime}\n"
                f"📈 ADX: {round(features.get('adx', 0), 1)}\n"
                f"📰 Sentiment: {sent_emoji} ({round(sentiment, 2)})\n\n"
                f"🔍 Setup: {reasons}\n\n"
                f"Risk/Reward: 1:{round(abs(tp1-entry)/abs(entry-sl), 1)}"
            )
            
            await query.edit_message_text(text, parse_mode='Markdown')
            
        elif query.data == "backtest":
            await query.edit_message_text("⏳ Running backtest on recent data...")
            win_rate, sharpe, trades = backtest(df15m, ensemble, memory, regime_detector, meta_learner, sentiment_analyzer)
            
            await query.edit_message_text(
                f"📊 **Backtest Results**\n\n"
                f"Win Rate: {win_rate}%\n"
                f"Sharpe Ratio: {sharpe}\n"
                f"Simulated Trades: {trades}\n"
                f"Trades in Memory: {len(memory.long_term)}",
                parse_mode='Markdown'
            )
            
        elif query.data == "status":
            status_text = (
                f"⚙️ **Model Status**\n\n"
                f"Current Regime: {regime_detector.regime}\n"
                f"ADX Value: {round(regime_detector.adx_value, 1)}\n"
                f"Trades in Memory: {len(memory.long_term)}\n"
                f"Patterns Learned: {len(memory.patterns)}\n"
                f"Last Training: {trainer.last_train_time.strftime('%H:%M:%S')}\n\n"
                f"Strategy Weights:\n"
                f"• Momentum: {round(meta_learner.strategy_weights['momentum'], 2)}\n"
                f"• Mean Reversion: {round(meta_learner.strategy_weights['mean_reversion'], 2)}\n"
                f"• Breakout: {round(meta_learner.strategy_weights['breakout'], 2)}\n"
                f"• ML: {round(meta_learner.strategy_weights['ml'], 2)}\n"
                f"• SMC: {round(meta_learner.strategy_weights.get('smc', 0), 2)}\n\n"
                f"ADX Threshold: 18+"
            )
            await query.edit_message_text(status_text, parse_mode='Markdown')
            
        elif query.data == "retrain":
            await query.edit_message_text("⏳ Training model with recent data...")
            success = trainer.train(df15m, regime_detector)
            
            if success:
                ensemble.models['lstm'].save(MODEL_FILE)
                memory.save(MEMORY_FILE)
                await query.edit_message_text("✅ Model retrained and saved successfully!")
            else:
                await query.edit_message_text("⚠️ Not enough data for training yet (need 50+ trades)")
                
    except Exception as e:
        logger.error(f"Button handler error: {e}")
        await query.edit_message_text(f"❌ Error: {str(e)}")

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if "subscribers" not in context.bot_data:
        context.bot_data["subscribers"] = set()
    context.bot_data["subscribers"].add(chat_id)
    await update.message.reply_text("✅ Subscribed to adaptive AI signals. You'll receive auto-updates.")

async def auto_retrain(context: ContextTypes.DEFAULT_TYPE):
    try:
        df15m = add_indicators(fetch_data("15min", 500))
        success = trainer.train(df15m, regime_detector)
        
        if success:
            ensemble.models['lstm'].save(MODEL_FILE)
            memory.save(MEMORY_FILE)
            regime_detector.save(REGIME_FILE)
            
            for chat_id in context.bot_data.get("subscribers", set()):
                try:
                    await context.bot.send_message(
                        chat_id, 
                        "🔄 Model auto-retrained with new market data\n"
                        f"Current Regime: {regime_detector.regime}\n"
                        f"Patterns: {len(memory.patterns)}"
                    )
                except:
                    continue
    except Exception as e:
        logger.error(f"Auto-retrain error: {e}")

# ==================== MAIN ====================

def main():
    global ensemble, memory, regime_detector, meta_learner, trainer, sentiment_analyzer
    
    # Initialize
    init_db()
    
    # Load or create components
    memory = TradeMemory()
    if os.path.exists(MEMORY_FILE):
        memory.load(MEMORY_FILE)
    
    regime_detector = MarketRegimeDetector()
    if os.path.exists(REGIME_FILE):
        try:
            regime_detector = joblib.load(REGIME_FILE)
        except:
            pass
    
    lstm_model = DeepLearningModel(input_dim=20, sequence_length=10)
    if os.path.exists(MODEL_FILE):
        lstm_model.load(MODEL_FILE)
    
    ensemble = AdaptiveEnsemble()
    ensemble.add_model('lstm', lstm_model, initial_weight=1.0)
    
    meta_learner = MetaLearner()
    sentiment_analyzer = NewsSentimentAnalyzer()
    
    trainer = ContinuousTrainer(lstm_model, memory, interval_minutes=120)
    
    # Build application
    application = ApplicationBuilder().token(TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("subscribe", subscribe))
    application.add_handler(CallbackQueryHandler(button))
    
    # Add job queue
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_repeating(auto_retrain, interval=7200, first=10)
    
    # Run
    logger.info("Bot starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
