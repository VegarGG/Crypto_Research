# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ta.trend import EMAIndicator, SMAIndicator, ADXIndicator

from ta.momentum import RSIIndicator, StochasticOscillator

from ta.trend import MACD

from ta.volatility import BollingerBands, AverageTrueRange

from arcticdb.version_store.helper import ArcticMemoryConfig
from arcticdb import Arctic

# DB Configuration
DB_PATH = '/Users/zway/Desktop/BTC_Project/DB'

# Moving average indicator pipeline
class MovingAveragePipeline:
    def __init__(self, lib_name='trend_indicators', store_path='arctic_store'):
        # connect to ArcticDB
        self.arctic = Arctic(store_path)
        if lib_name not in self.arctic.list_libraries():
            self.arctic.create_library(lib_name)
        self.library = self.arctic[lib_name]

    def compute_sma(self, df, days=7):
        window = days * 1440
        sma = SMAIndicator(close=df['Close'], window=window)
        df[f'sma_{days}d'] = sma.sma_indicator()
        return df

    def compute_ema(self, df, days=7):
        span = days * 1440
        ema = EMAIndicator(close=df['Close'], window=span)
        df[f'ema_{days}d'] = ema.ema_indicator()
        return df

    def compute_adx(self, df, days=14):
        window = days * 1440
        adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=window)
        df[f'adx_{days}d'] = adx.adx()
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

# Momentum indicator pipeline
class MomentumIndicatorPipeline:
    def __init__(self, lib_name='momentum_indicators', store_path='arctic_store'):
        # connect to ArcticDB
        self.arctic = Arctic(store_path)
        if lib_name not in self.arctic.list_libraries():
            self.arctic.create_library(lib_name)
        self.library = self.arctic[lib_name]

    def compute_rsi(self, df, days=14):
        window = days * 1440
        rsi = RSIIndicator(close=df['Close'], window=window)
        df[f'rsi_{days}d'] = rsi.rsi()
        return df

    def compute_stochastic(self, df, days=14, smooth_days=3):
        window = days * 1440
        smooth_window = smooth_days * 1440
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=window, smooth_window=smooth_window)
        df[f'stoch_k_{days}d'] = stoch.stoch()
        df[f'stoch_d_{days}d'] = stoch.stoch_signal()
        return df

    def compute_macd(self, df, fast_days=12, slow_days=26, signal_days=9):
        fast = fast_days * 1440
        slow = slow_days * 1440
        signal = signal_days * 1440
        macd = MACD(close=df['Close'], window_fast=fast, window_slow=slow, window_sign=signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()  # Histogram
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

# Volatitliy Indicators pipeline
class VolatilityIndicatorPipeline:
    def __init__(self, lib_name='volatility_indicators', store_path='arctic_store'):
        self.arctic = Arctic(store_path)
        if lib_name not in self.arctic.list_libraries():
            self.arctic.create_library(lib_name)
        self.library = self.arctic[lib_name]

    def compute_bollinger_bands(self, df, days=20, std=2):
        window = days * 1440
        bb = BollingerBands(close=df['Close'], window=window, window_dev=std)
        df[f'bb_mid_{days}d'] = bb.bollinger_mavg()
        df[f'bb_upper_{days}d'] = bb.bollinger_hband()
        df[f'bb_lower_{days}d'] = bb.bollinger_lband()
        return df

    def compute_atr(self, df, days=14):
        window = days * 1440
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=window)
        df[f'atr_{days}d'] = atr.average_true_range()
        return df

    def plot_indicators(self, df, bb_days=20, atr_days=14):
        # Bollinger Bands
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df['Close'], label='Close Price', alpha=0.6)
        plt.plot(df.index, df[f'bb_mid_{bb_days}d'], label='BB Mid')
        plt.plot(df.index, df[f'bb_upper_{bb_days}d'], label='BB Upper', color='red', linestyle='--')
        plt.plot(df.index, df[f'bb_lower_{bb_days}d'], label='BB Lower', color='green', linestyle='--')
        plt.title(f'Bollinger Bands ({bb_days}d)')
        plt.xlabel('Timestamp'); plt.ylabel('Price')
        plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

        # ATR
        plt.figure(figsize=(14, 4))
        plt.plot(df.index, df[f'atr_{atr_days}d'], label=f'ATR {atr_days}d', color='purple')
        plt.title(f'Average True Range (ATR {atr_days}d)')
        plt.xlabel('Timestamp'); plt.ylabel('ATR')
        plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    def run(self, df: pd.DataFrame, symbol: str, bb_days_list=[20], atr_days_list=[14]):
        df = df.copy()
        for d in bb_days_list:
            df = self.compute_bollinger_bands(df, days=d)
        for d in atr_days_list:
            df = self.compute_atr(df, days=d)

        self.library.write(symbol, df)
        print(f"[INFO] Written volatility indicators for {symbol} to ArcticDB")

        self.plot_indicators(df, bb_days=bb_days_list[0], atr_days=atr_days_list[0])
        return df

# Correlation Indicator Pipeline
class CorrelationIndicatorPipeline:
    def __init__(self, lib_name='correlation_indicators', store_path='arctic_store'):
        self.arctic = Arctic(store_path)
        if lib_name not in self.arctic.list_libraries():
            self.arctic.create_library(lib_name)
        self.library = self.arctic[lib_name]

    def compute_rolling_correlation(self, df1, df2, col1='BTC_Close', col2='SP500_Close', days=7):
        window = days
        merged = pd.merge(df1[[col1]], df2[[col2]], left_index=True, right_index=True)

        merged[f'corr_{days}d'] = merged[col1].rolling(window=window).corr(merged[col2])
        return merged[[f'corr_{days}d']]

    def plot_correlation(self, df, days=7):
        plt.figure(figsize=(14, 4))
        plt.plot(df.index, df[f'corr_{days}d'], label=f'Rolling Corr {days}d', color='teal')
        plt.axhline(0, linestyle='--', color='gray')
        plt.title(f'BTC vs SP500 {days}-day Rolling Correlation')
        plt.xlabel('Date'); plt.ylabel('Correlation')
        plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    def run(self, df_btc, df_sp500, symbol: str, days_list=[7], col1='Close', col2='Close'):
        df_btc = df_btc.copy()
        df_sp500 = df_sp500.copy()
        output_df = None

        for d in days_list:
            result = self.compute_rolling_correlation(df_btc, df_sp500, col1=col1, col2=col2, days=d)
            if output_df is None:
                output_df = result
            else:
                output_df = output_df.join(result, how='outer')

        # Store to ArcticDB
        self.library.write(symbol, output_df)
        print(f'[INFO] Written rolling correlation indicators for {symbol} to ArcticDB')

        # Plot only the first correlation (e.g., 7-day)
        self.plot_correlation(output_df, days=days_list[0])
        return output_df

# Fractal Dimension Indicator Pipeline    
class FractalDimensionPipeline:
    def __init__(self, lib_name='fractal_indicators', store_path='arctic_store'):
        self.arctic = Arctic(store_path)
        if lib_name not in self.arctic.list_libraries():
            self.arctic.create_library(lib_name)
        self.library = self.arctic[lib_name]

    @staticmethod
    def count_crossings_vectorized(prices, lower_bands, upper_bands):
        p1, p2 = prices[:-1], prices[1:]
        p1_matrix, p2_matrix = p1[None, :], p2[None, :]
        lower_matrix, upper_matrix = lower_bands[:, None], upper_bands[:, None]

        cross_up_lower = (p1_matrix < lower_matrix) & (lower_matrix < p2_matrix)
        cross_down_lower = (p1_matrix > lower_matrix) & (lower_matrix > p2_matrix)
        cross_up_upper = (p1_matrix < upper_matrix) & (upper_matrix < p2_matrix)
        cross_down_upper = (p1_matrix > upper_matrix) & (upper_matrix > p2_matrix)

        crossing = cross_down_lower | cross_down_upper | cross_up_lower | cross_up_upper
        return crossing.sum()

    def compute_keltner_fd(self, df_window):
        if df_window.shape[0] < 3:
            return np.nan

        prices = df_window['mid'].values
        mean_price = prices.mean()
        n_range = np.arange(1, 1001)
        deviations = n_range * 0.00001 * mean_price
        upper_bands = mean_price + deviations
        lower_bands = mean_price - deviations

        return self.count_crossings_vectorized(prices, lower_bands, upper_bands)

    def apply_fd(self, df, days=7):
        df = df.copy()
        df['mid'] = (df['High'] + df['Low']) / 2

        window = days  # assume daily frequency
        fd_series = df['mid'].rolling(window=window).apply(lambda x: self.compute_keltner_fd(df.loc[x.index]), raw=False)

        df[f'fd_{days}d'] = fd_series / 1000  # normalize to [0, 1]
        df.drop(columns=['mid'], inplace=True)
        return df

    def plot_fd(self, df, days=7):
        plt.figure(figsize=(14, 4))
        plt.plot(df.index, df[f'fd_{days}d'], label=f'Fractal Dimension {days}d', color='darkorange')
        plt.title(f'Fractal Dimension (Keltner Band) - {days} day window')
        plt.xlabel('Date'); plt.ylabel('Normalized FD')
        plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

    def run(self, df, symbol='BTC_FD', days_list=[7]):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        for d in days_list:
            df = self.apply_fd(df, days=d)

        self.library.write(symbol, df)
        print(f"[INFO] Written fractal dimension indicators for {symbol} to ArcticDB")

        self.plot_fd(df, days=days_list[0])
        return df
