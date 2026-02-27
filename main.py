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
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
import warnings
warnings.filterwarnings('ignore')

TOKEN = "8209411514:AAEUaPrSHE1XX48TizknSxnXgb-HR8E8bBE"
TWELVE_KEY = "413f1870be274f7fbfff5ab5d720c5a5"
DB_NAME = "xauusd_ai.db"
MODEL_FILE = "xauusd_model.pkl"
REGIME_FILE = "market_regime.pkl"
MEMORY_FILE = "trade_memory.pkl"

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
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
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx = dx.rolling(window=period).mean()
    
    return adx

class TradeMemory:
    def __init__(self, max_size=10000):
        self.short_term = deque(maxlen=1000)
        self.long_term = deque(maxlen=max_size)
        self.patterns = {}
        self.success_rates = {'trending': {'BUY': 0.5, 'SELL': 0.5}, 'ranging': {'BUY': 0.5, 'SELL': 0.5}}
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
            data = joblib.load(filepath)
            self.patterns = data.get('patterns', {})
            self.success_rates = data.get('success_rates', self.success_rates)

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
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        adx_series = calculate_adx(df)
        adx = adx_series.iloc[-1]
        self.adx_value = adx if not pd.isna(adx) else 0
        self.trend_strength = self.adx_value
        
        price_range = (df['high'].rolling(20).max().iloc[-1] - df['low'].rolling(20).min().iloc[-1]) / df['close'].iloc[-1]
        
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

class DeepLearningModel:
    def __init__(self, input_dim=20, sequence_length=10, hidden_dim=64, attention_heads=4):
        # Define ALL attributes FIRST before using them
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        
        # Ensure hidden_dim is divisible by attention_heads for proper attention mechanism
        if self.hidden_dim % self.attention_heads != 0:
            # Adjust hidden_dim to be divisible by attention_heads
            self.hidden_dim = (self.hidden_dim // self.attention_heads) * self.attention_heads
            if self.hidden_dim == 0:
                self.hidden_dim = self.attention_heads
        
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.velocity = None  # Will be initialized in _initialize_weights
        
        # Now initialize weights AFTER all attributes are defined
        self.weights = self._initialize_weights()
        self.velocity = {k: np.zeros_like(v) for k, v in self.weights.items()}
        
    def _initialize_weights(self):
        np.random.seed(42)
        hidden = self.hidden_dim  # Use the instance attribute
        head_dim = hidden // self.attention_heads
        
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
            'W_out': np.random.randn(hidden, 1) * 0.01,  # Changed: use full hidden dim, not multiplied by heads
            'b_out': np.zeros((1, 1))
        }
        
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
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
    def _attention(self, hidden_states):
        if len(hidden_states) == 0:
            return np.zeros((1, self.hidden_dim))
        
        stacked = np.vstack(hidden_states)  # Shape: (seq_len, hidden_dim)
        scores = stacked @ self.weights['W_attn']  # Shape: (seq_len, attention_heads)
        weights = self._softmax(scores)  # Shape: (seq_len, attention_heads)
        
        # Multi-head attention: compute context for each head
        head_dim = self.hidden_dim // self.attention_heads
        context_vectors = []
        
        for i in range(self.attention_heads):
            # Get weights for this head: (seq_len, 1)
            head_weights = weights[:, i:i+1]
            # Weighted sum of hidden states: (hidden_dim,)
            context = np.sum(stacked * head_weights, axis=0)
            context_vectors.append(context)
        
        # Concatenate all head contexts
        full_context = np.concatenate(context_vectors).reshape(1, -1)
        
        # Project back to hidden_dim if needed (optional, here we just use the concatenated version)
        # For simplicity, we'll use the average or just the concatenated version
        # But W_out expects (hidden_dim, 1), so we need to match dimensions
        
        # Actually, let's use mean of contexts to match W_out dimensions
        return np.mean(np.array(context_vectors), axis=0).reshape(1, self.hidden_dim)
        
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=0, keepdims=True) + 1e-10)
        
    def forward(self, sequence):
        # sequence shape: (batch_size, sequence_length, input_dim) or flattened
        # Handle different input shapes
        if isinstance(sequence, np.ndarray):
            if sequence.ndim == 1:
                # Single timestep: reshape to (1, input_dim)
                sequence = sequence.reshape(1, -1)
            elif sequence.ndim == 2 and sequence.shape[0] == 1:
                # Already (1, features) - single timestep
                pass
            elif sequence.ndim == 2:
                # (sequence_length, input_dim) - multiple timesteps
                pass
            elif sequence.ndim == 3:
                # (batch, seq, features) - take first batch
                sequence = sequence[0]
        
        h = np.zeros((1, self.hidden_dim))
        c = np.zeros((1, self.hidden_dim))
        hidden_states = []
        
        # Ensure sequence is iterable of timesteps
        if sequence.ndim == 2 and sequence.shape[0] <= self.sequence_length:
            # sequence is (seq_len, input_dim)
            seq_len = min(sequence.shape[0], self.sequence_length)
            for t in range(seq_len):
                x = sequence[t:t+1, :]  # Shape: (1, input_dim)
                if x.shape[1] != self.input_dim:
                    # Pad if necessary
                    if x.shape[1] < self.input_dim:
                        padding = np.zeros((1, self.input_dim - x.shape[1]))
                        x = np.concatenate([x, padding], axis=1)
                    else:
                        x = x[:, :self.input_dim]
                h, c = self._lstm_cell(x, h, c)
                hidden_states.append(h)
        else:
            # Single timestep or reshaped
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
            data = joblib.load(filepath)
            self.weights = data['weights']
            self.velocity = data['velocity']
            # Restore architecture parameters if saved
            self.input_dim = data.get('input_dim', self.input_dim)
            self.sequence_length = data.get('sequence_length', self.sequence_length)
            self.hidden_dim = data.get('hidden_dim', self.hidden_dim)
            self.attention_heads = data.get('attention_heads', self.attention_heads)

class MetaLearner:
    def __init__(self):
        self.strategy_weights = {'momentum': 0.25, 'mean_reversion': 0.25, 'breakout': 0.25, 'ml': 0.25}
        self.performance_history = {k: deque(maxlen=50) for k in self.strategy_weights}
        self.learning_rate = 0.1
        
    def update_weights(self, strategy, profit):
        self.performance_history[strategy].append(profit)
        
        if len(self.performance_history[strategy]) >= 10:
            avg_perf = np.mean(list(self.performance_history[strategy])[-10:])
            
            total_perf = sum(np.mean(list(h)[-10:]) if len(h) >= 10 else 0 
                           for h in self.performance_history.values())
            
            if total_perf > 0:
                for s in self.strategy_weights:
                    target_weight = np.mean(list(self.performance_history[s])[-10:]) / total_perf if len(self.performance_history[s]) >= 10 else 0.25
                    self.strategy_weights[s] += self.learning_rate * (target_weight - self.strategy_weights[s])
                    
            total = sum(self.strategy_weights.values())
            for s in self.strategy_weights:
                self.strategy_weights[s] /= total
                
    def get_combined_signal(self, signals):
        combined = 0
        for strategy, signal in signals.items():
            weight = self.strategy_weights.get(strategy, 0.25)
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
                    pred = model.predict_proba(features.reshape(1, -1))[0][1] if hasattr(model, 'predict_proba') else 0.5
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
    
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["momentum"] = df["close"].diff(10)
    df["volatility"] = df["close"].rolling(20).std()
    
    df.dropna(inplace=True)
    return df

def detect_structure(df, swing=3):
    if len(df) < swing + 1:
        return 0
    highs = df["high"].iloc[-swing:]
    lows = df["low"].iloc[-swing:]
    
    if df["close"].iloc[-1] > max(highs[:-1]) and df["close"].iloc[-2] < max(highs[:-1]):
        return 1
    if df["close"].iloc[-1] < min(lows[:-1]) and df["close"].iloc[-2] > min(lows[:-1]):
        return -1
    return 0

def detect_liquidity_sweep(df, threshold=0.5):
    if len(df) < 5:
        return 0, None
    
    prev_high = df["high"].iloc[-3:-1].max()
    prev_low = df["low"].iloc[-3:-1].min()
    
    ch = df["high"].iloc[-1]
    cl = df["low"].iloc[-1]
    cc = df["close"].iloc[-1]
    co = df["open"].iloc[-1]
    
    body = abs(cc - co)
    atr = df["atr"].iloc[-2] if len(df) > 1 else 1
    
    upper_sweep = ch > prev_high and cc < prev_high and body > threshold * atr
    lower_sweep = cl < prev_low and cc > prev_low and body > threshold * atr
    
    if upper_sweep:
        return -1, prev_high
    if lower_sweep:
        return 1, prev_low
    return 0, None

def detect_order_blocks(df):
    if len(df) < 3:
        return 0, None, None
    
    c1, c2, c3 = df["close"].iloc[-3], df["close"].iloc[-2], df["close"].iloc[-1]
    o1, o2, o3 = df["open"].iloc[-3], df["open"].iloc[-2], df["open"].iloc[-1]
    h2, l2 = df["high"].iloc[-2], df["low"].iloc[-2]
    
    bullish_ob = (c2 > o2) and (c1 < o1) and (c2 > o1) and (c3 > c2)
    bearish_ob = (c2 < o2) and (c1 > o1) and (c2 < o1) and (c3 < c2)
    
    if bullish_ob:
        return 1, l2, h2
    if bearish_ob:
        return -1, l2, h2
    return 0, None, None

def detect_fvg(df):
    if len(df) < 3:
        return 0, None, None
    
    h1, l1 = df["high"].iloc[-3], df["low"].iloc[-3]
    h2, l2 = df["high"].iloc[-2], df["low"].iloc[-2]
    h3, l3 = df["high"].iloc[-1], df["low"].iloc[-1]
    
    bullish_fvg = l3 > h1
    bearish_fvg = h3 < l1
    
    if bullish_fvg:
        return 1, h1, l3
    if bearish_fvg:
        return -1, h3, l1
    return 0, None, None

def supply_demand(df):
    if len(df) < 3:
        return 0
    body = abs(df["close"].iloc[-2] - df["open"].iloc[-2])
    atr = df["atr"].iloc[-2] if len(df) > 1 else 1
    
    if body > atr * 1.5:
        if df["close"].iloc[-2] > df["open"].iloc[-2]:
            return 1
        return -1
    return 0

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
        'momentum': df["momentum"].iloc[-1] if 'momentum' in df.columns else 0
    }
    
    if regime == 'trending':
        features['trend_alignment'] = 1 if features['ema50'] > features['ema200'] else -1
        features['momentum_factor'] = abs(features['momentum']) / (features['atr'] + 1e-6)
    else:
        features['mean_reversion_potential'] = abs(features['rsi'] - 50) / 50
        features['range_position'] = (df["close"].iloc[-1] - df["low"].rolling(20).min().iloc[-1]) / \
                                    (df["high"].rolling(20).max().iloc[-1] - df["low"].rolling(20).min().iloc[-1] + 1e-6)
    
    return features, regime

def generate_signal(df15m, df1h, df4h, ensemble, memory, regime_detector, meta_learner):
    features_15m, regime_15m = calculate_adaptive_features(df15m, regime_detector)
    features_1h, regime_1h = calculate_adaptive_features(df1h, regime_detector)
    features_4h, regime_4h = calculate_adaptive_features(df4h, regime_detector)
    
    regime = regime_4h
    adx_value = features_4h.get('adx', 0)
    
    if adx_value < 18:
        return "HOLD", None, None, None, None, 0, regime, features_15m
    
    bias = detect_structure(df4h)
    structure_15m = detect_structure(df15m)
    sweep, sweep_level = detect_liquidity_sweep(df15m)
    ob_dir, ob_low, ob_high = detect_order_blocks(df15m)
    fvg_dir, fvg_low, fvg_high = detect_fvg(df15m)
    zone = supply_demand(df15m)
    
    feature_vector = np.array([
        features_15m['rsi'] / 100,
        features_15m['macd'] / 10,
        features_15m['atr'] / 10,
        features_15m['bop'],
        1 if features_15m['ema50'] > features_15m['ema200'] else 0,
        features_15m['trend_strength'] / 50,
        bias,
        structure_15m,
        sweep,
        ob_dir,
        fvg_dir,
        zone
    ])
    
    if len(feature_vector) < 20:
        feature_vector = np.pad(feature_vector, (0, 20 - len(feature_vector)), mode='constant')
    
    ml_prediction = ensemble.predict(feature_vector[:20].reshape(1, 20), regime)
    
    pattern_confidence = memory.get_pattern_success_rate(features_15m)
    regime_bias = memory.get_regime_bias(regime, 'BUY')
    
    strategy_signals = {
        'momentum': 0.5 + (features_15m['macd'] / 20),
        'mean_reversion': 1 - abs(features_15m['rsi'] - 50) / 50,
        'breakout': 0.7 if sweep != 0 else 0.3,
        'ml': ml_prediction
    }
    
    combined_signal = meta_learner.get_combined_signal(strategy_signals)
    
    direction = "BUY" if combined_signal > 0.5 else "SELL"
    
    if bias < 0:
        direction = "SELL"
    elif bias > 0:
        direction = "BUY"
    
    if structure_15m < 0:
        direction = "SELL"
    elif structure_15m > 0:
        direction = "BUY"
    
    if sweep == -1:
        direction = "SELL"
    elif sweep == 1:
        direction = "BUY"
    
    if ob_dir != 0:
        direction = "BUY" if ob_dir == 1 else "SELL"
    
    if fvg_dir != 0:
        direction = "BUY" if fvg_dir == 1 else "SELL"
    
    if zone == -1:
        direction = "SELL"
    elif zone == 1:
        direction = "BUY"
    
    psar_candle = df15m["psar"].iloc[-1] if "psar" in df15m.columns else df15m["close"].iloc[-1]
    macd_c = df15m["macd"].iloc[-1] if "macd" in df15m.columns else 0
    macd_signal_c = df15m["macd_signal"].iloc[-1] if "macd_signal" in df15m.columns else 0
    entry_candle = df15m.iloc[-1]
    
    if direction == "BUY" and entry_candle["close"] < psar_candle:
        direction = "SELL"
    elif direction == "SELL" and entry_candle["close"] > psar_candle:
        direction = "BUY"
    
    if direction == "BUY" and macd_c < macd_signal_c:
        direction = "SELL"
    elif direction == "SELL" and macd_c > macd_signal_c:
        direction = "BUY"
    
    entry = entry_candle["close"]
    atr = df15m["atr"].iloc[-1] if "atr" in df15m.columns else entry * 0.001
    
    if regime == 'trending':
        risk_multiplier = 2.0
        tp_multiplier = 3.0
    elif regime == 'ranging':
        risk_multiplier = 1.0
        tp_multiplier = 1.5
    else:
        risk_multiplier = 1.5
        tp_multiplier = 2.0
    
    tp1 = entry + atr * tp_multiplier if direction == "BUY" else entry - atr * tp_multiplier
    tp2 = entry + atr * tp_multiplier * 1.5 if direction == "BUY" else entry - atr * tp_multiplier * 1.5
    sl = entry - atr * risk_multiplier if direction == "BUY" else entry + atr * risk_multiplier
    
    confidence = combined_signal * pattern_confidence * (0.5 + abs(bias) * 0.5)
    confidence = min(confidence * 100, 99)
    
    return direction, entry, tp1, tp2, sl, round(confidence, 2), regime, features_15m

def backtest(df, ensemble, memory, regime_detector, meta_learner, n_simulations=5):
    wins = 0
    total = 0
    returns = []
    
    for sim in range(n_simulations):
        sim_wins = 0
        sim_total = 0
        
        for i in range(210, len(df) - 5):
            sub = df.iloc[:i].copy()
            
            try:
                direction, entry, tp1, tp2, sl, conf, regime, features = generate_signal(
                    sub, sub, sub, ensemble, memory, regime_detector, meta_learner
                )
                
                if direction == "HOLD":
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
        return 0, 0
        
    win_rate = (wins / total) * 100
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) if len(returns) > 1 else 0
    
    return round(win_rate, 2), round(sharpe, 2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Get Signal", callback_data="signal")],
        [InlineKeyboardButton("Backtest", callback_data="backtest")],
        [InlineKeyboardButton("Model Status", callback_data="status")],
        [InlineKeyboardButton("Force Retrain", callback_data="retrain")]
    ]
    await update.message.reply_text(
        "XAUUSD Adaptive AI Trading Bot\nDeep Learning + Meta Learning + Continuous Training\nADX Filter: 18+",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    try:
        df15m = add_indicators(fetch_data("15min", 500))
        df1h = add_indicators(fetch_data("1h", 500))
        df4h = add_indicators(fetch_data("4h", 500))
        
        if query.data == "signal":
            direction, entry, tp1, tp2, sl, confidence, regime, features = generate_signal(
                df15m, df1h, df4h, ensemble, memory, regime_detector, meta_learner
            )
            
            if direction == "HOLD":
                await query.edit_message_text(f"HOLD - ADX {round(features.get('adx', 0), 1)} below 18 threshold")
                return
            
            text = f"""XAUUSD {direction}
Entry: {round(entry, 2)}
TP1: {round(tp1, 2)}
TP2: {round(tp2, 2)}
SL: {round(sl, 2)}
Confidence: {confidence}%
Regime: {regime}
ADX: {round(features.get('adx', 0), 1)}
ML Signal: {round(ensemble.predict(np.array(list(features.values())[:12] + [0] * 8).reshape(1, 20), regime) * 100, 1)}%"""
            
            await query.edit_message_text(text)
            
        elif query.data == "backtest":
            win_rate, sharpe = backtest(df15m, ensemble, memory, regime_detector, meta_learner)
            await query.edit_message_text(f"Win Rate: {win_rate}%\nSharpe Ratio: {sharpe}\nTrades in memory: {len(memory.long_term)}")
            
        elif query.data == "status":
            status_text = f"""Model Status:
Regime: {regime_detector.regime}
Trades in memory: {len(memory.long_term)}
Patterns learned: {len(memory.patterns)}
Last training: {trainer.last_train_time.strftime('%H:%M:%S')}
Strategy weights: {meta_learner.strategy_weights}
ADX Threshold: 18+"""
            await query.edit_message_text(status_text)
            
        elif query.data == "retrain":
            success = trainer.train(df15m, regime_detector)
            if success:
                await query.edit_message_text("Model retrained successfully!")
            else:
                await query.edit_message_text("Not enough data for training yet.")
                
    except Exception as e:
        await query.edit_message_text(f"Error: {str(e)}")

async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if "subscribers" not in context.application.bot_data:
        context.application.bot_data["subscribers"] = []
    if chat_id not in context.application.bot_data["subscribers"]:
        context.application.bot_data["subscribers"].append(chat_id)
    await update.message.reply_text("Subscribed to adaptive AI signals")

async def auto_retrain(context: ContextTypes.DEFAULT_TYPE):
    try:
        df15m = add_indicators(fetch_data("15min", 500))
        success = trainer.train(df15m, regime_detector)
        
        if success:
            ensemble.models['lstm'].save(MODEL_FILE)
            memory.save(MEMORY_FILE)
            
            for chat_id in context.bot_data.get("subscribers", []):
                await context.bot.send_message(chat_id, "Model auto-retrained with new data")
    except Exception as e:
        pass

init_db()

memory = TradeMemory()
if os.path.exists(MEMORY_FILE):
    memory.load(MEMORY_FILE)

regime_detector = MarketRegimeDetector()
if os.path.exists(REGIME_FILE):
    regime_detector = joblib.load(REGIME_FILE)

lstm_model = DeepLearningModel(input_dim=20, sequence_length=10)
if os.path.exists(MODEL_FILE):
    lstm_model.load(MODEL_FILE)

ensemble = AdaptiveEnsemble()
ensemble.add_model('lstm', lstm_model, initial_weight=1.0)

meta_learner = MetaLearner()

trainer = ContinuousTrainer(lstm_model, memory, interval_minutes=60)

app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("subscribe", subscribe))
app.add_handler(CallbackQueryHandler(button))

application = Application.builder().token(TOKEN).build()

application.job_queue.run_repeating(
    auto_retrain,
    interval=8600,
)

application.run_polling()
