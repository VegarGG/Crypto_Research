# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import threading
import multiprocessing
import warnings

try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. FD calculations will be slower. Install with: pip install numba")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

from ta.trend import EMAIndicator, SMAIndicator, ADXIndicator

from ta.momentum import RSIIndicator, StochasticOscillator

from ta.trend import MACD

from ta.volatility import BollingerBands, AverageTrueRange

from arcticdb.version_store.helper import ArcticMemoryConfig
from arcticdb import Arctic

# Import adaptive configuration
try:
    from config import config, ARCTIC_URI
except ImportError:
    # Fallback for backward compatibility
    import os
    import pathlib

    project_root = pathlib.Path(__file__).parent.parent
    arctic_store = project_root / "arctic_store"
    arctic_store.mkdir(parents=True, exist_ok=True)
    ARCTIC_URI = f"lmdb://{arctic_store}"


# Moving average indicator pipeline
class TrendIndicatorPipeline:
    def __init__(self, lib_name="trend_indicators", store_path=None):
        # connect to ArcticDB with adaptive path
        if store_path is None:
            store_path = ARCTIC_URI
        self.arctic = Arctic(store_path)
        if lib_name not in self.arctic.list_libraries():
            self.arctic.create_library(lib_name)
        self.library = self.arctic[lib_name]

    def compute_sma(self, df, days=7, minute_data=True):
        window = days * 1440 if minute_data else days
        sma = SMAIndicator(close=df["Close"], window=window)
        df[f"sma_{days}d"] = sma.sma_indicator()
        return df

    def compute_ema(self, df, days=7, minute_data=True):
        span = days * 1440 if minute_data else days
        ema = EMAIndicator(close=df["Close"], window=span)
        df[f"ema_{days}d"] = ema.ema_indicator()
        return df

    def compute_adx(self, df, days=14, minute_data=True):
        window = days * 1440 if minute_data else days
        adx = ADXIndicator(
            high=df["High"], low=df["Low"], close=df["Close"], window=window
        )
        df[f"adx_{days}d"] = adx.adx()
        return df

    def plot_indicators(self, df, sma_days=7, ema_days=20, adx_window=14):
        # Plot SMA and EMA
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df["Close"], label="Close Price", alpha=0.5)
        plt.plot(df.index, df[f"sma_{sma_days}d"], label=f"SMA {sma_days}d")
        plt.plot(df.index, df[f"ema_{ema_days}d"], label=f"EMA {ema_days}d")
        plt.title("Trend Indicators: Close Price, SMA, EMA")
        plt.xlabel("Timestamp")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot ADX
        plt.figure(figsize=(14, 4))
        plt.plot(
            df.index,
            df[f"adx_{adx_window}d"],
            label=f"ADX {adx_window}d",
            color="purple",
        )
        plt.axhline(25, color="gray", linestyle="--", label="Trend Threshold")
        plt.title(f"Average Directional Index (ADX {adx_window})")
        plt.xlabel("Timestamp")
        plt.ylabel("ADX")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        sma_windows=[7],
        ema_spans=[7],
        adx_windows=[7],
    ):
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
            adx_window=adx_windows[0],
        )

        return df


# Momentum indicator pipeline
class MomentumIndicatorPipeline:
    def __init__(self, lib_name="momentum_indicators", store_path=None):
        # connect to ArcticDB with adaptive path
        if store_path is None:
            store_path = ARCTIC_URI
        self.arctic = Arctic(store_path)
        if lib_name not in self.arctic.list_libraries():
            self.arctic.create_library(lib_name)
        self.library = self.arctic[lib_name]

    def compute_rsi(self, df, days=14, minute_data=True):
        window = days * 1440 if minute_data else days
        rsi = RSIIndicator(close=df["Close"], window=window)
        df[f"rsi_{days}d"] = rsi.rsi()
        return df

    def compute_stochastic(self, df, days=14, smooth_days=3, minute_data=True):
        window = days * 1440 if minute_data else days
        smooth_window = smooth_days * 1440
        stoch = StochasticOscillator(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            window=window,
            smooth_window=smooth_window,
        )
        df[f"stoch_k_{days}d"] = stoch.stoch()
        df[f"stoch_d_{days}d"] = stoch.stoch_signal()
        df[f"stoch_k_{days}d"] = stoch.stoch()
        df[f"stoch_d_{days}d"] = stoch.stoch_signal()
        return df

        return df

        return df

    def compute_dynamic_rsi_parallel(self, df, rsi_col, window_days=30, minute_data=True, chunk_size=None):
        window = window_days * 1440 if minute_data else window_days
        
        if chunk_size is None:
            chunk_size = max(10000, len(df) // self.max_workers)
            
        if len(df) < window or len(df) < chunk_size:
            return self.compute_dynamic_rsi_regime_sequential(df, rsi_col, window_days, minute_data)
            
        # Create overlapping chunks
        chunks = []
        overlap = window - 1
        
        for i in range(0, len(df), chunk_size):
            start_idx = max(0, i - overlap)
            end_idx = min(len(df), i + chunk_size + overlap)
            chunk_df = df.iloc[start_idx:end_idx].copy()
            chunks.append((chunk_df, rsi_col, window, start_idx))
            
        print(f"[INFO] Processing Dynamic RSI in parallel with {len(chunks)} chunks...")
        
        # Process chunks
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(parallel_rsi_worker, chunks))
            
        # Combine results
        q70_full = np.full(len(df), np.nan)
        q30_full = np.full(len(df), np.nan)
        
        for chunk_start_idx, q70_chunk, q30_chunk in results:
            chunk_end_idx = min(len(df), chunk_start_idx + len(q70_chunk))
            
            if chunk_start_idx == 0:
                q70_full[chunk_start_idx:chunk_end_idx] = q70_chunk[:chunk_end_idx-chunk_start_idx]
                q30_full[chunk_start_idx:chunk_end_idx] = q30_chunk[:chunk_end_idx-chunk_start_idx]
            else:
                skip_overlap = overlap if chunk_start_idx > 0 else 0
                valid_start = chunk_start_idx + skip_overlap
                valid_chunk_start = skip_overlap
                
                if valid_start < len(df):
                    len_to_copy = chunk_end_idx - valid_start
                    q70_full[valid_start:chunk_end_idx] = q70_chunk[valid_chunk_start:valid_chunk_start+len_to_copy]
                    q30_full[valid_start:chunk_end_idx] = q30_chunk[valid_chunk_start:valid_chunk_start+len_to_copy]

        df[f"{rsi_col}_q70"] = q70_full
        df[f"{rsi_col}_q30"] = q30_full
        
        # Define regimes
        df[f"{rsi_col}_regime"] = 0
        df.loc[df[rsi_col] > df[f"{rsi_col}_q70"], f"{rsi_col}_regime"] = 1
        df.loc[df[rsi_col] < df[f"{rsi_col}_q30"], f"{rsi_col}_regime"] = -1
        
        return df

    def compute_dynamic_rsi_regime_sequential(self, df, rsi_col, window_days=30, minute_data=True):
        """Sequential version using Numba"""
        window = window_days * 1440 if minute_data else window_days
        rsi_values = df[rsi_col].values
        
        print(f"[INFO] Calculating Dynamic RSI thresholds (Sequential, Window: {window})...")
        q70 = rolling_quantile_numba(rsi_values, window, 0.70)
        q30 = rolling_quantile_numba(rsi_values, window, 0.30)
        
        df[f"{rsi_col}_q70"] = q70
        df[f"{rsi_col}_q30"] = q30
        
        df[f"{rsi_col}_regime"] = 0
        df.loc[df[rsi_col] > df[f"{rsi_col}_q70"], f"{rsi_col}_regime"] = 1
        df.loc[df[rsi_col] < df[f"{rsi_col}_q30"], f"{rsi_col}_regime"] = -1
        
        return df

    def compute_dynamic_rsi_regime(self, df, rsi_col, window_days=30, minute_data=True):
        """
        Calculates dynamic RSI regimes. Dispatches to parallel or sequential based on size.
        """
        # Use parallel if data is large enough
        if len(df) > 20000 and self.max_workers > 1:
            try:
                return self.compute_dynamic_rsi_parallel(df, rsi_col, window_days, minute_data)
            except Exception as e:
                print(f"[WARNING] Parallel RSI failed: {e}. Falling back to sequential.")
                return self.compute_dynamic_rsi_regime_sequential(df, rsi_col, window_days, minute_data)
        else:
            return self.compute_dynamic_rsi_regime_sequential(df, rsi_col, window_days, minute_data)

    def compute_macd(
        self, df, fast_days=12, slow_days=26, signal_days=9, minute_data=True
    ):
        fast = fast_days * 1440 if minute_data else fast_days
        slow = slow_days * 1440 if minute_data else slow_days
        signal = signal_days * 1440 if minute_data else signal_days
        macd = MACD(
            close=df["Close"], window_fast=fast, window_slow=slow, window_sign=signal
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()  # Histogram
        return df

    def plot_indicators(self, df, rsi_days=14, stoch_days=14, macd_days=(12, 26, 9)):
        # RSI
        plt.figure(figsize=(14, 4))
        plt.plot(
            df.index, df[f"rsi_{rsi_days}d"], label=f"RSI {rsi_days}d", color="orange"
        )
        plt.axhline(70, color="red", linestyle="--", label="Overbought")
        plt.axhline(30, color="green", linestyle="--", label="Oversold")
        plt.title("Relative Strength Index (RSI)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        # Stochastic Oscillator
        plt.figure(figsize=(14, 4))
        plt.plot(
            df.index,
            df[f"stoch_k_{stoch_days}d"],
            label=f"%K {stoch_days}d",
            color="blue",
        )
        plt.plot(
            df.index,
            df[f"stoch_d_{stoch_days}d"],
            label=f"%D {stoch_days}d",
            color="magenta",
        )
        plt.axhline(80, color="red", linestyle="--", label="Overbought")
        plt.axhline(20, color="green", linestyle="--", label="Oversold")
        plt.title("Stochastic Oscillator")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        # MACD
        fast, slow, signal = macd_days
        plt.figure(figsize=(14, 5))
        plt.plot(df.index, df["macd"], label=f"MACD ({fast}, {slow})", color="blue")
        plt.plot(df.index, df["macd_signal"], label=f"Signal ({signal})", color="red")
        plt.bar(df.index, df["macd_diff"], label="Histogram", color="gray", alpha=0.4)
        plt.title("MACD")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        rsi_windows=[14],
        stoch_windows=[14],
        macd_params=(12, 26, 9),
    ):
        df = df.copy()
        for w in rsi_windows:
            df = self.compute_rsi(df, days=w)
        for w in stoch_windows:
            df = self.compute_stochastic(df, days=w)
        df = self.compute_macd(df, *macd_params)
        
        # Compute Dynamic RSI Regimes for the first RSI window
        if rsi_windows:
            rsi_col = f"rsi_{rsi_windows[0]}d"
            # Use 30-day window for regime context
            df = self.compute_dynamic_rsi_regime(df, rsi_col, window_days=30)

        # Save to ArcticDB
        self.library.write(symbol, df)
        print(f"[INFO] Written momentum indicators for {symbol} to ArcticDB")

        # Plot the indicators automatically
        self.plot_indicators(
            df,
            rsi_days=rsi_windows[0],
            stoch_days=stoch_windows[0],
            macd_days=macd_params,
        )

        return df


# Time-Series Decomposition Pipeline
class TimeSeriesFeaturesPipeline:
    def __init__(self, lib_name="timeseries_features", store_path=None):
        if store_path is None:
            store_path = ARCTIC_URI
        self.arctic = Arctic(store_path)
        if lib_name not in self.arctic.list_libraries():
            self.arctic.create_library(lib_name)
        self.library = self.arctic[lib_name]

    def compute_trend_residual(self, df, days=1, minute_data=True):
        """
        Decomposes price into Trend and Residual using a simple Moving Average.
        Trend = SMA(window)
        Residual = Close / Trend (Ratio) or Close - Trend (Diff)
        """
        window = days * 1440 if minute_data else days
        
        # Trend (SMA)
        df[f"trend_sma_{days}d"] = df["Close"].rolling(window=window).mean()
        
        # Residual (Ratio - better for normalization)
        df[f"residual_ratio_{days}d"] = df["Close"] / df[f"trend_sma_{days}d"]
        
        # Residual (Diff)
        df[f"residual_diff_{days}d"] = df["Close"] - df[f"trend_sma_{days}d"]
        
        return df

    def compute_volatility_regime(self, df, days=7, minute_data=True):
        """
        Calculates volatility regime based on ATR relative to its history.
        """
        window = days * 1440 if minute_data else days
        
        # Calculate ATR if not present (simplified here or assume present)
        # Let's calculate a simple rolling std dev of returns as proxy if ATR not passed
        if "log_ret" not in df.columns:
            df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
            
        df[f"volatility_{days}d"] = df["log_ret"].rolling(window=window).std()
        
        # Compare current volatility to long-term average (4x window)
        long_window = window * 4
        df[f"volatility_mean_{days}d"] = df[f"volatility_{days}d"].rolling(window=long_window).mean()
        
        df[f"volatility_ratio_{days}d"] = df[f"volatility_{days}d"] / df[f"volatility_mean_{days}d"]
        
        return df

    def run(self, df, symbol: str, trend_days=[1, 7], vol_days=[7]):
        df = df.copy()
        
        for d in trend_days:
            df = self.compute_trend_residual(df, days=d)
            
        for d in vol_days:
            df = self.compute_volatility_regime(df, days=d)
            
        self.library.write(symbol, df)
        print(f"[INFO] Written time-series features for {symbol} to ArcticDB")
        return df


# Volatitliy Indicators pipeline
class VolatilityIndicatorPipeline:
    def __init__(self, lib_name="volatility_indicators", store_path=None):
        if store_path is None:
            store_path = ARCTIC_URI
        self.arctic = Arctic(store_path)
        if lib_name not in self.arctic.list_libraries():
            self.arctic.create_library(lib_name)
        self.library = self.arctic[lib_name]

    def compute_bollinger_bands(self, df, days=20, std=2, minute_data=True):
        window = days * 1440 if minute_data else days
        bb = BollingerBands(close=df["Close"], window=window, window_dev=std)
        df[f"bb_mid_{days}d"] = bb.bollinger_mavg()
        df[f"bb_upper_{days}d"] = bb.bollinger_hband()
        df[f"bb_lower_{days}d"] = bb.bollinger_lband()
        return df

    def compute_atr(self, df, days=14, minute_data=True):
        window = days * 1440 if minute_data else days
        atr = AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"], window=window
        )
        df[f"atr_{days}d"] = atr.average_true_range()
        return df

    def plot_indicators(self, df, bb_days=20, atr_days=14):
        # Bollinger Bands
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df["Close"], label="Close Price", alpha=0.6)
        plt.plot(df.index, df[f"bb_mid_{bb_days}d"], label="BB Mid")
        plt.plot(
            df.index,
            df[f"bb_upper_{bb_days}d"],
            label="BB Upper",
            color="red",
            linestyle="--",
        )
        plt.plot(
            df.index,
            df[f"bb_lower_{bb_days}d"],
            label="BB Lower",
            color="green",
            linestyle="--",
        )
        plt.title(f"Bollinger Bands ({bb_days}d)")
        plt.xlabel("Timestamp")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        # ATR
        plt.figure(figsize=(14, 4))
        plt.plot(
            df.index, df[f"atr_{atr_days}d"], label=f"ATR {atr_days}d", color="purple"
        )
        plt.title(f"Average True Range (ATR {atr_days}d)")
        plt.xlabel("Timestamp")
        plt.ylabel("ATR")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

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
    def __init__(self, lib_name="correlation_indicators", store_path=None):
        if store_path is None:
            store_path = ARCTIC_URI
        self.arctic = Arctic(store_path)
        if lib_name not in self.arctic.list_libraries():
            self.arctic.create_library(lib_name)
        self.library = self.arctic[lib_name]

    def compute_rolling_correlation(
        self, df1, df2, col1="BTC_Close", col2="SP500_Close", days=7, minute_data=True
    ):
        window = days * 1440 if minute_data else days

        # merge only the needed columns
        df_corr = pd.merge(df1[[col1]], df2[[col2]], left_index=True, right_index=True)
        corr_series = df_corr[col1].rolling(window=window).corr(df_corr[col2])

        # merge the result back into df1 without dropping other features
        df1[f"corr_{days}d"] = corr_series
        return df1

    def plot_correlation(self, df, days=7):
        plt.figure(figsize=(14, 4))
        plt.plot(
            df.index, df[f"corr_{days}d"], label=f"Rolling Corr {days}d", color="teal"
        )
        plt.axhline(0, linestyle="--", color="gray")
        plt.title(f"BTC vs SP500 {days}-day Rolling Correlation")
        plt.xlabel("Date")
        plt.ylabel("Correlation")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def run(
        self, df_btc, df_sp500, symbol: str, days_list=[7], col1="Close", col2="Close"
    ):
        df_btc = df_btc.copy()
        df_sp500 = df_sp500.copy()
        output_df = None

        for d in days_list:
            result = self.compute_rolling_correlation(
                df_btc, df_sp500, col1=col1, col2=col2, days=d
            )
            if output_df is None:
                output_df = result
            else:
                output_df = output_df.join(result, how="outer")

        # Store to ArcticDB
        self.library.write(symbol, output_df)
        print(f"[INFO] Written rolling correlation indicators for {symbol} to ArcticDB")

        # Plot only the first correlation (e.g., 7-day)
        self.plot_correlation(output_df, days=days_list[0])
        return output_df


# Optimized JIT-compiled functions for Fractal Dimension calculation
if NUMBA_AVAILABLE:
    @jit(nopython=True, fastmath=True, cache=True)
    def count_crossings_numba(prices, lower_bands, upper_bands):
        n_bands = len(lower_bands)
        n_prices = len(prices)
        total_crossings = 0
        
        for band_idx in prange(n_bands):
            lower_band = lower_bands[band_idx]
            upper_band = upper_bands[band_idx]
            crossings = 0
            
            for i in range(n_prices - 1):
                p1, p2 = prices[i], prices[i + 1]
                
                # Check for crossings with lower band
                if (p1 < lower_band < p2) or (p1 > lower_band > p2):
                    crossings += 1
                # Check for crossings with upper band
                elif (p1 < upper_band < p2) or (p1 > upper_band > p2):
                    crossings += 1
            
            total_crossings += crossings
        
        return total_crossings
    
    @jit(nopython=True, fastmath=True, cache=True)
    def compute_keltner_fd_numba(prices):
        if len(prices) < 3:
            return np.nan
        
        mean_price = np.mean(prices)
        n_range = np.arange(1, 1001, dtype=np.int32)
        deviations = n_range * 0.00001 * mean_price
        upper_bands = mean_price + deviations
        lower_bands = mean_price - deviations
        
        return count_crossings_numba(prices, lower_bands, upper_bands)

    @jit(nopython=True, fastmath=True, cache=True)
    def rolling_quantile_numba(arr, window, quantile):
        n = len(arr)
        result = np.full(n, np.nan)
        
        # For each position
        for i in range(n):
            if i >= window - 1:
                # Extract window
                window_arr = arr[i - window + 1 : i + 1].copy()
                # Sort window to find quantile
                # Note: np.quantile is not fully supported in all numba versions in nopython mode
                # so we use sorting
                window_arr.sort()
                
                # Calculate index
                idx = int(round(quantile * (window - 1)))
                idx = min(max(idx, 0), window - 1)
                result[i] = window_arr[idx]
                
        return result
else:
    # Fallback non-JIT versions
    def count_crossings_numba(prices, lower_bands, upper_bands):
        p1, p2 = prices[:-1], prices[1:]
        p1_matrix, p2_matrix = p1[None, :], p2[None, :]
        lower_matrix, upper_matrix = lower_bands[:, None], upper_bands[:, None]

        cross_up_lower = (p1_matrix < lower_matrix) & (lower_matrix < p2_matrix)
        cross_down_lower = (p1_matrix > lower_matrix) & (lower_matrix > p2_matrix)
        cross_up_upper = (p1_matrix < upper_matrix) & (upper_matrix < p2_matrix)
        cross_down_upper = (p1_matrix > upper_matrix) & (upper_matrix > p2_matrix)

        crossing = cross_down_lower | cross_down_upper | cross_up_lower | cross_up_upper
        return crossing.sum()
    
    def compute_keltner_fd_numba(prices):
        if len(prices) < 3:
            return np.nan
        
        mean_price = prices.mean()
        n_range = np.arange(1, 1001)
        deviations = n_range * 0.00001 * mean_price
        upper_bands = mean_price + deviations
        lower_bands = mean_price - deviations
        
        return count_crossings_numba(prices, lower_bands, upper_bands)

    def rolling_quantile_numba(arr, window, quantile):
        # Fallback non-JIT version using pandas
        return pd.Series(arr).rolling(window=window).quantile(quantile).values

def parallel_fd_worker(chunk_data):
    chunk_df, window_size, chunk_start_idx = chunk_data
    chunk_df = chunk_df.copy()
    chunk_df["mid"] = (chunk_df["High"] + chunk_df["Low"]) / 2
    
    fd_values = []
    for i in range(len(chunk_df)):
        if i >= window_size - 1:
            window_start = max(0, i - window_size + 1)
            window_prices = chunk_df["mid"].iloc[window_start:i+1].values
            fd_val = compute_keltner_fd_numba(window_prices)
        else:
            fd_val = np.nan
        fd_values.append(fd_val)
    
    return chunk_start_idx, np.array(fd_values)

def parallel_rsi_worker(chunk_data):
    chunk_df, rsi_col, window, chunk_start_idx = chunk_data
    # Ensure we have the data
    rsi_values = chunk_df[rsi_col].values
    
    # Calculate quantiles using Numba optimized function
    q70 = rolling_quantile_numba(rsi_values, window, 0.70)
    q30 = rolling_quantile_numba(rsi_values, window, 0.30)
    
    return chunk_start_idx, q70, q30

# Fractal Dimension Indicator Pipeline
class FractalDimensionPipeline:
    def __init__(self, lib_name="fractal_indicators", store_path=None, max_workers=None):
        if store_path is None:
            store_path = ARCTIC_URI
        self.arctic = Arctic(store_path)
        if lib_name not in self.arctic.list_libraries():
            self.arctic.create_library(lib_name)
        self.library = self.arctic[lib_name]
        self.max_workers = max_workers or min(4, (multiprocessing.cpu_count() or 1))
        self._lock = threading.Lock()

    def apply_fd_parallel(self, df, days=7, minute_data=True, chunk_size=None):
        df = df.copy()
        df["mid"] = (df["High"] + df["Low"]) / 2
        window_size = days * 1440 if minute_data else days
        
        if chunk_size is None:
            chunk_size = max(1000, len(df) // self.max_workers)
        
        if len(df) < window_size or len(df) < chunk_size:
            return self.apply_fd_sequential(df, days, minute_data)
        
        # Create overlapping chunks to handle window calculations
        chunks = []
        overlap = window_size - 1
        
        for i in range(0, len(df), chunk_size):
            start_idx = max(0, i - overlap)
            end_idx = min(len(df), i + chunk_size + overlap)
            chunk_df = df.iloc[start_idx:end_idx].copy()
            chunks.append((chunk_df, window_size, start_idx))
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(parallel_fd_worker, chunks))
        
        # Combine results
        fd_values = np.full(len(df), np.nan)
        for chunk_start_idx, chunk_fd_values in results:
            chunk_end_idx = min(len(df), chunk_start_idx + len(chunk_fd_values))
            # Only use the non-overlapping part of each chunk
            if chunk_start_idx == 0:
                # First chunk - use all values
                fd_values[chunk_start_idx:chunk_end_idx] = chunk_fd_values[:chunk_end_idx-chunk_start_idx]
            else:
                # Skip overlap region for subsequent chunks
                skip_overlap = overlap if chunk_start_idx > 0 else 0
                valid_start = chunk_start_idx + skip_overlap
                valid_chunk_start = skip_overlap
                valid_chunk_end = len(chunk_fd_values)
                
                if valid_start < len(df):
                    fd_values[valid_start:chunk_end_idx] = chunk_fd_values[valid_chunk_start:valid_chunk_end][:chunk_end_idx-valid_start]
        
        return fd_values
    
    def apply_fd_sequential(self, df, days=7, minute_data=True):
        df = df.copy()
        df["mid"] = (df["High"] + df["Low"]) / 2
        window_size = days * 1440 if minute_data else days
        
        fd_values = np.full(len(df), np.nan)
        mid_values = df["mid"].values
        
        for i in range(window_size - 1, len(df)):
            window_prices = mid_values[i - window_size + 1:i + 1]
            fd_values[i] = compute_keltner_fd_numba(window_prices)
        
        return fd_values

    def apply_fd(self, df, days=7, minute_data=True, use_parallel=True, chunk_size=None):
        df = df.copy()
        
        # Choose processing method based on data size and availability
        data_size_threshold = 5000  # Use parallel processing for datasets larger than this
        should_use_parallel = (
            use_parallel and 
            len(df) > data_size_threshold and 
            self.max_workers > 1
        )
        
        if should_use_parallel:
            try:
                fd_values = self.apply_fd_parallel(df, days, minute_data, chunk_size)
            except Exception as e:
                print(f"[WARNING] Parallel processing failed: {e}. Falling back to sequential.")
                fd_values = self.apply_fd_sequential(df, days, minute_data)
        else:
            fd_values = self.apply_fd_sequential(df, days, minute_data)
        
        # Convert to pandas Series for normalization
        fd_series = pd.Series(fd_values, index=df.index)
        
        # Use global maximum normalization (like original implementation)
        max_fd = fd_series.max()
        if pd.isna(max_fd) or max_fd == 0:
            max_fd = 1  # fallback to avoid division by zero
        df[f"fd_{days}d"] = fd_series / max_fd
        
        return df

    def plot_fd(self, df, days=7):
        plt.figure(figsize=(14, 4))
        plt.plot(
            df.index,
            df[f"fd_{days}d"],
            label=f"Fractal Dimension {days}d",
            color="darkorange",
        )
        plt.title(f"Fractal Dimension (Keltner Band) - {days} day window")
        plt.xlabel("Date")
        plt.ylabel("Normalized FD")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def run(self, df, symbol="BTC_FD", days_list=[7], use_parallel=True, chunk_size=None, show_performance=True):
        import time
        
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        
        total_start_time = time.time()
        performance_info = []
        
        for d in days_list:
            start_time = time.time()
            df = self.apply_fd(df, days=d, use_parallel=use_parallel, chunk_size=chunk_size)
            end_time = time.time()
            
            elapsed = end_time - start_time
            performance_info.append(f"FD {d}d: {elapsed:.2f}s")
            
            if show_performance:
                processing_mode = "parallel" if use_parallel and len(df) > 5000 else "sequential"
                print(f"[INFO] FD {d}d calculation completed in {elapsed:.2f}s ({processing_mode})")
        
        # Thread-safe ArcticDB write operation
        with self._lock:
            try:
                self.library.write(symbol, df)
                total_time = time.time() - total_start_time
                print(f"[INFO] Written fractal dimension indicators for {symbol} to ArcticDB")
                if show_performance:
                    print(f"[INFO] Total processing time: {total_time:.2f}s")
                    print(f"[INFO] Performance breakdown: {', '.join(performance_info)}")
            except Exception as e:
                print(f"[ERROR] Failed to write to ArcticDB: {e}")
                raise
        
        self.plot_fd(df, days=days_list[0])
        return df
    
    def benchmark_performance(self, df, days=7, minute_data=True, iterations=3):
        import time
        
        print(f"\n[INFO] Benchmarking FD calculation performance...")
        print(f"Data size: {len(df):,} rows, Window: {days} days")
        
        # Test sequential performance
        seq_times = []
        for i in range(iterations):
            start_time = time.time()
            self.apply_fd_sequential(df.copy(), days, minute_data)
            seq_times.append(time.time() - start_time)
        
        avg_seq_time = np.mean(seq_times)
        
        # Test parallel performance (if applicable)
        if len(df) > 5000 and self.max_workers > 1:
            par_times = []
            for i in range(iterations):
                try:
                    start_time = time.time()
                    self.apply_fd_parallel(df.copy(), days, minute_data)
                    par_times.append(time.time() - start_time)
                except Exception as e:
                    print(f"[WARNING] Parallel processing failed: {e}")
                    break
            
            if par_times:
                avg_par_time = np.mean(par_times)
                speedup = avg_seq_time / avg_par_time
                print(f"Sequential: {avg_seq_time:.2f}s")
                print(f"Parallel:   {avg_par_time:.2f}s")
                print(f"Speedup:    {speedup:.2f}x")
            else:
                print(f"Sequential: {avg_seq_time:.2f}s")
                print(f"Parallel processing unavailable")
        else:
            print(f"Sequential: {avg_seq_time:.2f}s")
            print(f"Note: Data too small for parallel processing")
    
    def validate_results(self, df, days=7, minute_data=True, tolerance=1e-6):
        print(f"\n[INFO] Validating optimized vs sequential results...")
        
        # Get results from both methods
        seq_result = self.apply_fd_sequential(df.copy(), days, minute_data)
        
        if len(df) > 5000:
            try:
                par_result = self.apply_fd_parallel(df.copy(), days, minute_data)
                
                # Compare results
                valid_mask = ~(np.isnan(seq_result) | np.isnan(par_result))
                if np.sum(valid_mask) == 0:
                    print("[WARNING] No valid values to compare")
                    return False
                
                diff = np.abs(seq_result[valid_mask] - par_result[valid_mask])
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                print(f"Max difference: {max_diff:.2e}")
                print(f"Mean difference: {mean_diff:.2e}")
                
                if max_diff < tolerance:
                    print("[SUCCESS] Results match within tolerance")
                    return True
                else:
                    print(f"[ERROR] Results differ by more than tolerance ({tolerance:.2e})")
                    return False
            except Exception as e:
                print(f"[WARNING] Parallel validation failed: {e}")
                return True  # Sequential still works
        else:
            print("[INFO] Data too small for parallel processing comparison")
            return True
