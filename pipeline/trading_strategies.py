"""
Trading Strategies Pipeline

This module provides regime-specific trading strategies for different market conditions.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import json
from dataclasses import dataclass


@dataclass
class TradeSignal:
    """Data class for trade signals"""
    timestamp: pd.Timestamp
    action: str  # 'buy' or 'sell'
    price: float
    regime: int
    confidence: float = 1.0
    strategy_name: str = ""


@dataclass
class Trade:
    """Data class for completed trades"""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    entry_regime: int
    exit_regime: int
    pnl: float
    return_pct: float
    duration: int
    strategy_name: str
    exit_reason: str = ""


class BaseStrategy(ABC):
    """Base class for regime-specific trading strategies"""

    def __init__(self, name: str, regime_type: int, **kwargs):
        self.name = name
        self.regime_type = regime_type
        self.parameters = kwargs

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, regime_predictions: np.ndarray) -> List[TradeSignal]:
        """
        Generate trading signals based on regime and technical indicators

        Args:
            df: DataFrame with OHLCV and technical indicators
            regime_predictions: Array of regime predictions

        Returns:
            List of TradeSignal objects
        """
        pass

    def get_parameters(self) -> Dict:
        """Get strategy parameters"""
        return {
            'name': self.name,
            'regime_type': self.regime_type,
            **self.parameters
        }

    def update_parameters(self, **kwargs):
        """Update strategy parameters"""
        self.parameters.update(kwargs)


class BullMarketStrategy(BaseStrategy):
    """Aggressive trend-following strategy for bull markets"""

    def __init__(self,
                 ma_fast: int = 7,
                 ma_slow: int = 20,
                 rsi_threshold: float = 40,
                 stop_loss: float = 0.08,
                 take_profit: float = 0.15,
                 trail_stop: float = 0.05):
        super().__init__("Bull Trend Following", 2,
                        ma_fast=ma_fast, ma_slow=ma_slow, rsi_threshold=rsi_threshold,
                        stop_loss=stop_loss, take_profit=take_profit, trail_stop=trail_stop)

    def generate_signals(self, df: pd.DataFrame, regime_predictions: np.ndarray) -> List[TradeSignal]:
        signals = []
        position = None
        entry_price = 0
        peak_price = 0

        for i in range(len(df)):
            if i >= len(regime_predictions):
                break

            row = df.iloc[i]
            regime = regime_predictions[i]

            # Only trade in bull regime
            if regime != self.regime_type:
                if position == 'long':
                    signals.append(TradeSignal(
                        timestamp=row.name,
                        action='sell',
                        price=row['Close'],
                        regime=regime,
                        strategy_name=self.name
                    ))
                    position = None
                continue

            # Entry conditions for bull market
            if position is None and regime == self.regime_type:
                # Check for bullish momentum
                ema_signal = (f'ema_{self.parameters["ma_fast"]}d' in df.columns and
                            f'ema_{self.parameters["ma_slow"]}d' in df.columns and
                            row[f'ema_{self.parameters["ma_fast"]}d'] > row[f'ema_{self.parameters["ma_slow"]}d'])

                rsi_signal = ('rsi_14d' in df.columns and
                            row['rsi_14d'] > self.parameters['rsi_threshold'])

                macd_signal = ('macd_hist_12_26' in df.columns and
                             row['macd_hist_12_26'] > 0)

                if ema_signal and (rsi_signal or macd_signal):
                    signals.append(TradeSignal(
                        timestamp=row.name,
                        action='buy',
                        price=row['Close'],
                        regime=regime,
                        strategy_name=self.name
                    ))
                    position = 'long'
                    entry_price = row['Close']
                    peak_price = entry_price

            # Exit conditions
            elif position == 'long':
                current_price = row['Close']
                peak_price = max(peak_price, current_price)

                price_change = (current_price - entry_price) / entry_price

                # Stop loss
                if price_change <= -self.parameters['stop_loss']:
                    signals.append(TradeSignal(
                        timestamp=row.name,
                        action='sell',
                        price=current_price,
                        regime=regime,
                        strategy_name=self.name
                    ))
                    position = None

                # Take profit
                elif price_change >= self.parameters['take_profit']:
                    signals.append(TradeSignal(
                        timestamp=row.name,
                        action='sell',
                        price=current_price,
                        regime=regime,
                        strategy_name=self.name
                    ))
                    position = None

                # Trailing stop
                elif peak_price > entry_price * (1 + self.parameters['trail_stop']):
                    trail_threshold = peak_price * (1 - self.parameters['trail_stop'])
                    if current_price <= trail_threshold:
                        signals.append(TradeSignal(
                            timestamp=row.name,
                            action='sell',
                            price=current_price,
                            regime=regime,
                            strategy_name=self.name
                        ))
                        position = None

                # Trend reversal exit
                elif (f'ema_{self.parameters["ma_fast"]}d' in df.columns and
                      f'ema_{self.parameters["ma_slow"]}d' in df.columns and
                      row[f'ema_{self.parameters["ma_fast"]}d'] < row[f'ema_{self.parameters["ma_slow"]}d']):
                    signals.append(TradeSignal(
                        timestamp=row.name,
                        action='sell',
                        price=current_price,
                        regime=regime,
                        strategy_name=self.name
                    ))
                    position = None

        return signals


class BearMarketStrategy(BaseStrategy):
    """Conservative short-term mean reversion for bear markets"""

    def __init__(self,
                 rsi_oversold: float = 25,
                 rsi_overbought: float = 60,
                 stop_loss: float = 0.05,
                 take_profit: float = 0.08,
                 bb_deviation: float = 1.0):
        super().__init__("Bear Mean Reversion", 0,
                        rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought,
                        stop_loss=stop_loss, take_profit=take_profit, bb_deviation=bb_deviation)

    def generate_signals(self, df: pd.DataFrame, regime_predictions: np.ndarray) -> List[TradeSignal]:
        signals = []
        position = None
        entry_price = 0

        for i in range(len(df)):
            if i >= len(regime_predictions):
                break

            row = df.iloc[i]
            regime = regime_predictions[i]

            # Only trade in bear regime
            if regime != self.regime_type:
                if position == 'long':
                    signals.append(TradeSignal(
                        timestamp=row.name,
                        action='sell',
                        price=row['Close'],
                        regime=regime,
                        strategy_name=self.name
                    ))
                    position = None
                continue

            # Entry conditions for bear market (buy oversold)
            if position is None and regime == self.regime_type:
                rsi_oversold = ('rsi_14d' in df.columns and
                              row['rsi_14d'] < self.parameters['rsi_oversold'])

                # Additional confirmation: price below BB lower band
                bb_signal = ('bb_lower_20d' in df.columns and
                           row['Close'] < row['bb_lower_20d'] * (1 + self.parameters['bb_deviation']/100))

                if rsi_oversold and bb_signal:
                    signals.append(TradeSignal(
                        timestamp=row.name,
                        action='buy',
                        price=row['Close'],
                        regime=regime,
                        strategy_name=self.name
                    ))
                    position = 'long'
                    entry_price = row['Close']

            # Exit conditions
            elif position == 'long':
                current_price = row['Close']
                price_change = (current_price - entry_price) / entry_price

                # Stop loss
                if price_change <= -self.parameters['stop_loss']:
                    signals.append(TradeSignal(
                        timestamp=row.name,
                        action='sell',
                        price=current_price,
                        regime=regime,
                        strategy_name=self.name
                    ))
                    position = None

                # Take profit or overbought exit
                elif (price_change >= self.parameters['take_profit'] or
                      ('rsi_14d' in df.columns and row['rsi_14d'] > self.parameters['rsi_overbought'])):
                    signals.append(TradeSignal(
                        timestamp=row.name,
                        action='sell',
                        price=current_price,
                        regime=regime,
                        strategy_name=self.name
                    ))
                    position = None

        return signals


class SidewaysMarketStrategy(BaseStrategy):
    """Range-bound trading for sideways markets"""

    def __init__(self,
                 bb_period: int = 20,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 stop_loss: float = 0.03,
                 take_profit: float = 0.06,
                 bb_threshold: float = 0.01):
        super().__init__("Sideways Range Trading", 1,
                        bb_period=bb_period, rsi_oversold=rsi_oversold, rsi_overbought=rsi_overbought,
                        stop_loss=stop_loss, take_profit=take_profit, bb_threshold=bb_threshold)

    def generate_signals(self, df: pd.DataFrame, regime_predictions: np.ndarray) -> List[TradeSignal]:
        signals = []
        position = None
        entry_price = 0

        for i in range(len(df)):
            if i >= len(regime_predictions):
                break

            row = df.iloc[i]
            regime = regime_predictions[i]

            # Only trade in sideways regime
            if regime != self.regime_type:
                if position == 'long':
                    signals.append(TradeSignal(
                        timestamp=row.name,
                        action='sell',
                        price=row['Close'],
                        regime=regime,
                        strategy_name=self.name
                    ))
                    position = None
                continue

            # Entry conditions for sideways market
            if position is None and regime == self.regime_type:
                # Buy near lower Bollinger Band with oversold RSI
                bb_lower_signal = ('bb_lower_20d' in df.columns and
                                 row['Close'] <= row['bb_lower_20d'] * (1 + self.parameters['bb_threshold']))

                rsi_signal = ('rsi_14d' in df.columns and
                            row['rsi_14d'] < self.parameters['rsi_oversold'])

                if bb_lower_signal and rsi_signal:
                    signals.append(TradeSignal(
                        timestamp=row.name,
                        action='buy',
                        price=row['Close'],
                        regime=regime,
                        strategy_name=self.name
                    ))
                    position = 'long'
                    entry_price = row['Close']

            # Exit conditions
            elif position == 'long':
                current_price = row['Close']
                price_change = (current_price - entry_price) / entry_price

                # Stop loss
                if price_change <= -self.parameters['stop_loss']:
                    signals.append(TradeSignal(
                        timestamp=row.name,
                        action='sell',
                        price=current_price,
                        regime=regime,
                        strategy_name=self.name
                    ))
                    position = None

                # Take profit, upper BB, or overbought exit
                elif (price_change >= self.parameters['take_profit'] or
                      ('bb_upper_20d' in df.columns and current_price >= row['bb_upper_20d'] * (1 - self.parameters['bb_threshold'])) or
                      ('rsi_14d' in df.columns and row['rsi_14d'] > self.parameters['rsi_overbought'])):
                    signals.append(TradeSignal(
                        timestamp=row.name,
                        action='sell',
                        price=current_price,
                        regime=regime,
                        strategy_name=self.name
                    ))
                    position = None

        return signals


class StrategyOptimizer:
    """Optimize strategy parameters using historical data"""

    def __init__(self, strategy_class, regime_predictions: np.ndarray):
        self.strategy_class = strategy_class
        self.regime_predictions = regime_predictions

    def optimize_parameters(self, df: pd.DataFrame, parameter_ranges: Dict) -> Dict:
        """
        Optimize strategy parameters using grid search

        Args:
            df: Historical data
            parameter_ranges: Dictionary of parameter ranges to test

        Returns:
            Best parameters and performance metrics
        """
        from itertools import product

        best_params = None
        best_performance = -np.inf
        results = []

        # Generate parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())

        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))

            # Create strategy with these parameters
            strategy = self.strategy_class(**params)

            # Generate signals and calculate performance
            signals = strategy.generate_signals(df, self.regime_predictions)
            performance = self._calculate_performance(df, signals)

            results.append({
                'parameters': params,
                'performance': performance
            })

            if performance['sharpe_ratio'] > best_performance:
                best_performance = performance['sharpe_ratio']
                best_params = params

        return {
            'best_parameters': best_params,
            'best_performance': best_performance,
            'all_results': results
        }

    def _calculate_performance(self, df: pd.DataFrame, signals: List[TradeSignal],
                             initial_capital: float = 10000, commission: float = 0.001) -> Dict:
        """Calculate strategy performance metrics"""
        # Simple backtest
        portfolio_value = initial_capital
        cash = initial_capital
        position = 0
        trades = []

        for signal in signals:
            price = signal.price

            if signal.action == 'buy' and position == 0:
                position = (cash * (1 - commission)) / price
                cash = 0
                entry_price = price
                entry_time = signal.timestamp

            elif signal.action == 'sell' and position > 0:
                cash = position * price * (1 - commission)
                pnl = cash - initial_capital
                return_pct = (cash - initial_capital) / initial_capital * 100

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': signal.timestamp,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'return_pct': return_pct
                })

                position = 0

        # Calculate metrics
        if not trades:
            return {'sharpe_ratio': -1, 'total_return': 0, 'win_rate': 0}

        returns = [t['return_pct'] for t in trades]
        total_return = sum(returns)
        win_rate = np.mean([r > 0 for r in returns])
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(trades)
        }


class StrategyManager:
    """Manage multiple trading strategies"""

    def __init__(self):
        self.strategies = {}

    def add_strategy(self, name: str, strategy: BaseStrategy):
        """Add a strategy to the manager"""
        self.strategies[name] = strategy

    def remove_strategy(self, name: str):
        """Remove a strategy from the manager"""
        if name in self.strategies:
            del self.strategies[name]

    def generate_all_signals(self, df: pd.DataFrame, regime_predictions: np.ndarray) -> Dict[str, List[TradeSignal]]:
        """Generate signals from all strategies"""
        all_signals = {}

        for name, strategy in self.strategies.items():
            signals = strategy.generate_signals(df, regime_predictions)
            all_signals[name] = signals

        return all_signals

    def save_strategies(self, filepath: str):
        """Save strategy configurations"""
        config = {}
        for name, strategy in self.strategies.items():
            config[name] = strategy.get_parameters()

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_strategies(cls, filepath: str):
        """Load strategy configurations"""
        with open(filepath, 'r') as f:
            config = json.load(f)

        manager = cls()

        # Map strategy classes
        strategy_classes = {
            'Bull Trend Following': BullMarketStrategy,
            'Bear Mean Reversion': BearMarketStrategy,
            'Sideways Range Trading': SidewaysMarketStrategy
        }

        for name, params in config.items():
            strategy_name = params.pop('name')
            regime_type = params.pop('regime_type')

            if strategy_name in strategy_classes:
                strategy_class = strategy_classes[strategy_name]
                strategy = strategy_class(**params)
                manager.add_strategy(name, strategy)

        return manager


def create_default_strategies() -> StrategyManager:
    """Create default set of strategies"""
    manager = StrategyManager()

    # Bull market strategy
    bull_strategy = BullMarketStrategy(
        ma_fast=7, ma_slow=20, rsi_threshold=40,
        stop_loss=0.06, take_profit=0.12, trail_stop=0.04
    )
    manager.add_strategy('bull_default', bull_strategy)

    # Bear market strategy
    bear_strategy = BearMarketStrategy(
        rsi_oversold=25, rsi_overbought=60,
        stop_loss=0.04, take_profit=0.08, bb_deviation=1.0
    )
    manager.add_strategy('bear_default', bear_strategy)

    # Sideways market strategy
    sideways_strategy = SidewaysMarketStrategy(
        bb_period=20, rsi_oversold=30, rsi_overbought=70,
        stop_loss=0.03, take_profit=0.06, bb_threshold=0.01
    )
    manager.add_strategy('sideways_default', sideways_strategy)

    return manager


if __name__ == "__main__":
    # Example usage
    print("Trading Strategies Pipeline")
    print("==========================")

    # Create default strategies
    strategy_manager = create_default_strategies()

    print(f"Created {len(strategy_manager.strategies)} default strategies:")
    for name, strategy in strategy_manager.strategies.items():
        print(f"  - {name}: {strategy.name} (regime {strategy.regime_type})")

    print("\nStrategies ready for use with market regime predictions.")