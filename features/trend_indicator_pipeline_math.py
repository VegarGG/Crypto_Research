# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from arcticdb.version_store.helper import ArcticMemoryConfig
from arcticdb import Arctic

# DB configuration
DB_PATH = '/Users/zway/Desktop/BTC_Project/DB'


# Moving Average Pipeline, constructed by using math formula
class MovingAveragePipeline:
    def __init__(self, lib_name='trend_indicators', store_path='arctic_store'):
        # connect to ArcticDB
        self.arctic = Arctic(store_path)
        if lib_name not in self.arctic.list_libraries():
            self.arctic.create_library(lib_name)
        self.library = self.arctic[lib_name]

    def compute_sma(self, df, days=7):
        window = days * 1440
        df[f'sma_{days}d'] = df['Close'].rolling(window=window).mean()
        return df

    def compute_ema(self, df, days=7):
        span = days * 1440
        df[f'ema_{days}d'] = df['Close'].ewm(span=span, adjust=False).mean()
        return df

    def compute_adx(self, df, days=14):
        window = days * 1440
        # 1. Calculate directional movements
        df['up_move'] = df['High'] - df['High'].shift(1)
        df['down_move'] = df['Low'].shift(1) - df['Low']
        
        df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

        # 2. True Range
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift(1))
        df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # 3. Smoothed TR, +DM, -DM using Wilder's EMA (alpha=1/window)
        alpha = 1 / window
        df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
        df['+DM_smooth'] = pd.Series(df['+DM']).ewm(alpha=alpha, adjust=False).mean()
        df['-DM_smooth'] = pd.Series(df['-DM']).ewm(alpha=alpha, adjust=False).mean()

        # 4. +DI and -DI
        df['+DI'] = 100 * (df['+DM_smooth'] / df['ATR'])
        df['-DI'] = 100 * (df['-DM_smooth'] / df['ATR'])

        # 5. DX and ADX
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        df[f'adx_{days}d'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()

        # Clean up intermediate columns
        df.drop(columns=['up_move', 'down_move', '+DM', '-DM', 'tr1', 'tr2', 'tr3',
                        'TR', 'ATR', '+DM_smooth', '-DM_smooth', '+DI', '-DI', 'DX'], inplace=True)

        return df
    
    def plot_indicators(self, df, sma_days=7, ema_days=20, adx_window=14):
        # Plot SMA and EMA
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
        plt.plot(df.index, df[f'sma_{sma_days}d'], label=f'SMA {sma_days}d')
        plt.plot(df.index, df[f'ema_{ema_days}d'], label=f'EMA {ema_days}d')
        plt.title('Trend Indicators: Close Price, SMA, EMA')
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot ADX
        plt.figure(figsize=(14, 4))
        plt.plot(df.index, df[f'adx_{adx_window}d'], label=f'ADX {adx_window}d', color='purple')
        plt.axhline(25, color='gray', linestyle='--', label='Trend Threshold')
        plt.title(f'Average Directional Index (ADX {adx_window})')
        plt.xlabel('Timestamp')
        plt.ylabel('ADX')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def run(self, df: pd.DataFrame, symbol: str, sma_windows=[7], ema_spans=[7], adx_windows=[7]):
        df = df.copy()

        # Compute indicators
        for w in sma_windows:
            df = self.compute_sma(df, days=w)
        for s in ema_spans:
            df = self.compute_ema(df, days=s)
        for d in adx_windows:
            df = self.compute_adx(df, days=d)

        # Save to ArcticDB
        self.library.write(symbol, df)
        print(f"[INFO] Written trend indicators for {symbol} to ArcticDB")
        
        # Now plot
        self.plot_indicators(
            df,
            sma_days=sma_windows[0],
            ema_days=ema_spans[0],
            adx_window=adx_windows[0]
        )
        
        return df

# Momentum Indicator Pipeline
class MomentumIndicatorPipeline:
    def __init__(self, lib_name='momentum_indicators', store_path='arctic_store'):
        # connect to ArcticDB
        self.arctic = Arctic(store_path)
        if lib_name not in self.arctic.list_libraries():
            self.arctic.create_library(lib_name)
        self.library = self.arctic[lib_name]

    def compute_rsi(self, df, days=14):
        window = days * 1440
        df = df.copy()
        
        # compute price change
        delta = df['Close'].diff()
        
        # Gains & Losses
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing (EMA with alpha = 1/window)
        avg_gain = pd.Series(gain).ewm(alpha=1/window, adjust=False).mean()
        avg_loss = pd.Series(loss).ewm(alpha=1/window, adjust=False).mean()
        
        # Compute RS & RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        df[f'rsi_{days}d'] = rsi.values
        
        return df

    def compute_stochastic(self, df, days=14, smooth_days=3):
        window = days * 1440
        smooth = smooth_days * 1440
        
        df = df.copy()
        
        # Rolling high and low for %K
        low_min = df['Low'].rolling(window=window).min()
        high_max = df['High'].rolling(window=window).max()
        
        # %K line (Fast Stochastic)
        df[f'stoch_k_{days}d'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        
        # %D line 
        df[f'stoch_d_{days}d'] = df[f'stoch_k_{days}d'].rolling(window=smooth).mean()
        
        return df

    def compute_macd(self, df, fast_days=12, slow_days=26, signal_days=9):
        df = df.copy()
        
        fast_span = fast_days * 1440
        slow_span = slow_days * 1440
        signal_span = signal_days * 1440
        
        # EMA fast and slow
        ema_fast = df['Close'].ewm(span=fast_span, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow_span, adjust=False).mean()
        
        # MACD line
        df['macd'] = ema_fast - ema_slow
        
        # Signal line
        df['macd_signal'] = df['macd'].ewm(span=signal_span, adjust=False).mean()
        
        # Histogram
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        return df
    
    def plot_indicators(self, df, rsi_days=14, stoch_days=14, macd_days=(12, 26, 9)):
        # RSI
        plt.figure(figsize=(14, 4))
        plt.plot(df.index, df[f'rsi_{rsi_days}d'], label=f'RSI {rsi_days}d', color='orange')
        plt.axhline(70, color='red', linestyle='--', label='Overbought')
        plt.axhline(30, color='green', linestyle='--', label='Oversold')
        plt.title('Relative Strength Index (RSI)')
        plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

        # Stochastic Oscillator
        plt.figure(figsize=(14, 4))
        plt.plot(df.index, df[f'stoch_k_{stoch_days}d'], label=f'%K {stoch_days}d', color='blue')
        plt.plot(df.index, df[f'stoch_d_{stoch_days}d'], label=f'%D {stoch_days}d', color='magenta')
        plt.axhline(80, color='red', linestyle='--', label='Overbought')
        plt.axhline(20, color='green', linestyle='--', label='Oversold')
        plt.title('Stochastic Oscillator')
        plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

        # MACD
        fast, slow, signal = macd_days
        plt.figure(figsize=(14, 5))
        plt.plot(df.index, df['macd'], label=f'MACD ({fast}, {slow})', color='blue')
        plt.plot(df.index, df['macd_signal'], label=f'Signal ({signal})', color='red')
        plt.bar(df.index, df['macd_diff'], label='Histogram', color='gray', alpha=0.4)
        plt.title('MACD')
        plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    def run(self, df: pd.DataFrame, symbol: str, rsi_windows=[14], stoch_windows=[14], macd_params=(12, 26, 9)):
        df = df.copy()
        for w in rsi_windows:
            df = self.compute_rsi(df, days=w)
        for w in stoch_windows:
            df = self.compute_stochastic(df, days=w)
        df = self.compute_macd(df, *macd_params)

        # Save to ArcticDB
        self.library.write(symbol, df)
        print(f"[INFO] Written momentum indicators for {symbol} to ArcticDB")
        
        # Plot the indicators automatically
        self.plot_indicators(
            df,
            rsi_days=rsi_windows[0],
            stoch_days=stoch_windows[0],
            macd_days=macd_params
            )
        
        return df

