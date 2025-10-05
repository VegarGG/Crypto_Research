"""
Comprehensive Backtesting Utilities

This module provides advanced backtesting capabilities including:
- Multi-strategy backtesting
- Walk-forward analysis
- Risk metrics calculation
- Performance visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from trading_strategies import TradeSignal, Trade


class BacktestEngine:
    """Advanced backtesting engine with comprehensive analytics"""

    def __init__(self,
                 initial_capital: float = 10000,
                 commission: float = 0.001,
                 slippage: float = 0.0001,
                 margin_requirement: float = 1.0):
        """
        Initialize backtesting engine

        Args:
            initial_capital: Starting portfolio value
            commission: Commission per trade (as fraction of trade value)
            slippage: Price slippage per trade (as fraction)
            margin_requirement: Margin requirement for leveraged trades
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.margin_requirement = margin_requirement

    def run_backtest(self, df: pd.DataFrame, signals: List[TradeSignal],
                    position_sizing: str = 'fixed',
                    max_position_size: float = 1.0) -> Dict:
        """
        Run comprehensive backtest

        Args:
            df: DataFrame with OHLCV data
            signals: List of trading signals
            position_sizing: Position sizing method ('fixed', 'kelly', 'volatility')
            max_position_size: Maximum position size as fraction of capital

        Returns:
            Dictionary with backtest results
        """
        # Initialize tracking variables
        portfolio_values = []
        cash = self.initial_capital
        position = 0
        trades = []
        equity_curve = []

        # Track current trade
        current_trade = None
        trade_count = 0

        # Sort signals by timestamp
        signals = sorted(signals, key=lambda x: x.timestamp)
        signal_idx = 0

        # Process each timestamp
        for i, (timestamp, row) in enumerate(df.iterrows()):
            current_price = row['Close']

            # Process signals for this timestamp
            while (signal_idx < len(signals) and
                   signals[signal_idx].timestamp <= timestamp):

                signal = signals[signal_idx]
                signal_idx += 1

                # Calculate position size
                position_value = self._calculate_position_size(
                    cash, current_price, position_sizing, max_position_size, df.iloc[:i+1]
                )

                # Apply slippage
                adjusted_price = self._apply_slippage(signal.price, signal.action)

                if signal.action == 'buy' and position == 0:
                    # Open long position
                    position = position_value / adjusted_price
                    cash -= position_value
                    trade_cost = position_value * self.commission
                    cash -= trade_cost

                    # Start new trade
                    current_trade = {
                        'trade_id': trade_count,
                        'entry_time': signal.timestamp,
                        'entry_price': adjusted_price,
                        'entry_regime': signal.regime,
                        'strategy_name': signal.strategy_name,
                        'position_size': position,
                        'entry_cost': trade_cost
                    }
                    trade_count += 1

                elif signal.action == 'sell' and position > 0:
                    # Close long position
                    exit_value = position * adjusted_price
                    trade_cost = exit_value * self.commission
                    cash += exit_value - trade_cost

                    # Complete trade record
                    if current_trade:
                        entry_value = current_trade['position_size'] * current_trade['entry_price']
                        gross_pnl = exit_value - entry_value
                        net_pnl = gross_pnl - current_trade['entry_cost'] - trade_cost
                        return_pct = net_pnl / entry_value * 100

                        trade_record = Trade(
                            entry_time=current_trade['entry_time'],
                            entry_price=current_trade['entry_price'],
                            exit_time=signal.timestamp,
                            exit_price=adjusted_price,
                            entry_regime=current_trade['entry_regime'],
                            exit_regime=signal.regime,
                            pnl=net_pnl,
                            return_pct=return_pct,
                            duration=(signal.timestamp - current_trade['entry_time']).days,
                            strategy_name=current_trade['strategy_name']
                        )
                        trades.append(trade_record)

                    position = 0
                    current_trade = None

            # Calculate portfolio value
            if position > 0:
                position_value = position * current_price
                total_value = cash + position_value
            else:
                total_value = cash

            portfolio_values.append(total_value)
            equity_curve.append({
                'timestamp': timestamp,
                'portfolio_value': total_value,
                'cash': cash,
                'position_value': position * current_price if position > 0 else 0,
                'position': position
            })

        # Create results
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
        performance_metrics = self._calculate_metrics(equity_df, trades)

        return {
            'equity_curve': equity_df,
            'trades': trades,
            'performance_metrics': performance_metrics,
            'final_value': portfolio_values[-1] if portfolio_values else self.initial_capital
        }

    def _calculate_position_size(self, cash: float, price: float,
                               method: str, max_size: float,
                               historical_data: pd.DataFrame) -> float:
        """Calculate position size based on specified method"""
        if method == 'fixed':
            return min(cash * max_size, cash * 0.95)  # Leave some cash for fees

        elif method == 'volatility':
            # Volatility-based position sizing
            if len(historical_data) < 20:
                return cash * max_size * 0.5

            returns = historical_data['Close'].pct_change().dropna()
            volatility = returns.std()
            vol_adjustment = max(0.1, min(1.0, 0.02 / volatility))
            return cash * max_size * vol_adjustment

        elif method == 'kelly':
            # Simplified Kelly criterion (requires trade history)
            return cash * max_size * 0.25  # Conservative Kelly

        else:
            return cash * max_size

    def _apply_slippage(self, price: float, action: str) -> float:
        """Apply slippage to trade price"""
        if action == 'buy':
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def _calculate_metrics(self, equity_df: pd.DataFrame, trades: List[Trade]) -> Dict:
        """Calculate comprehensive performance metrics"""
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Returns-based metrics
        returns = equity_df['portfolio_value'].pct_change().dropna()

        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            sortino_ratio = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        # Drawdown metrics
        rolling_max = equity_df['portfolio_value'].cummax()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Trade-based metrics
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]

            win_rate = len(winning_trades) / len(trades)
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if avg_loss != 0 else np.inf

            # Risk-adjusted metrics
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

            # Duration analysis
            avg_trade_duration = np.mean([t.duration for t in trades])
            max_trade_duration = max([t.duration for t in trades])

        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            calmar_ratio = 0
            avg_trade_duration = 0
            max_trade_duration = 0

        return {
            'total_return': total_return * 100,
            'annualized_return': ((final_value / self.initial_capital) ** (252 / len(equity_df)) - 1) * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown * 100,
            'volatility': returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0,
            'num_trades': len(trades),
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'max_trade_duration': max_trade_duration
        }


class WalkForwardAnalysis:
    """Walk-forward analysis for strategy validation"""

    def __init__(self, training_window: int = 252, rebalance_freq: int = 63):
        """
        Initialize walk-forward analysis

        Args:
            training_window: Number of periods for training window
            rebalance_freq: Frequency of rebalancing (in periods)
        """
        self.training_window = training_window
        self.rebalance_freq = rebalance_freq

    def run_analysis(self, df: pd.DataFrame, strategy_generator,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
        """
        Run walk-forward analysis

        Args:
            df: DataFrame with historical data
            strategy_generator: Function that generates strategy given training data
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dictionary with walk-forward results
        """
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        results = []
        current_pos = self.training_window

        while current_pos + self.rebalance_freq < len(df):
            # Define training and testing periods
            train_start = current_pos - self.training_window
            train_end = current_pos
            test_start = current_pos
            test_end = min(current_pos + self.rebalance_freq, len(df))

            train_data = df.iloc[train_start:train_end]
            test_data = df.iloc[test_start:test_end]

            try:
                # Generate strategy for this period
                strategy = strategy_generator(train_data)

                # Test strategy on out-of-sample data
                backtest_engine = BacktestEngine()
                signals = strategy.generate_signals(test_data)
                result = backtest_engine.run_backtest(test_data, signals)

                results.append({
                    'period_start': test_data.index[0],
                    'period_end': test_data.index[-1],
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'performance': result['performance_metrics'],
                    'num_trades': len(result['trades'])
                })

            except Exception as e:
                print(f"Error in walk-forward period {test_data.index[0]}: {e}")

            current_pos += self.rebalance_freq

        return {
            'periods': results,
            'summary': self._summarize_walkforward(results)
        }

    def _summarize_walkforward(self, results: List[Dict]) -> Dict:
        """Summarize walk-forward analysis results"""
        if not results:
            return {}

        returns = [r['performance']['total_return'] for r in results]
        sharpe_ratios = [r['performance']['sharpe_ratio'] for r in results]

        return {
            'avg_return': np.mean(returns),
            'return_std': np.std(returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'sharpe_std': np.std(sharpe_ratios),
            'positive_periods': np.sum([r > 0 for r in returns]),
            'total_periods': len(returns),
            'consistency_ratio': np.sum([r > 0 for r in returns]) / len(returns)
        }


class PerformanceAnalyzer:
    """Comprehensive performance analysis and visualization"""

    @staticmethod
    def plot_equity_curve(equity_df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None):
        """Plot equity curve with optional benchmark"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Equity curve
        ax1 = axes[0]
        ax1.plot(equity_df.index, equity_df['portfolio_value'], label='Strategy', linewidth=2)

        if benchmark_df is not None:
            ax1.plot(benchmark_df.index, benchmark_df['Close'], label='Benchmark', alpha=0.7)

        ax1.set_title('Portfolio Equity Curve')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = axes[1]
        rolling_max = equity_df['portfolio_value'].cummax()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max * 100

        ax2.fill_between(equity_df.index, drawdown, 0, alpha=0.7, color='red')
        ax2.set_title('Portfolio Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_trade_analysis(trades: List[Trade]):
        """Plot comprehensive trade analysis"""
        if not trades:
            print("No trades to analyze")
            return

        trade_df = pd.DataFrame([{
            'pnl': t.pnl,
            'return_pct': t.return_pct,
            'duration': t.duration,
            'entry_regime': t.entry_regime,
            'strategy': t.strategy_name
        } for t in trades])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # PnL distribution
        axes[0, 0].hist(trade_df['pnl'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--')
        axes[0, 0].set_title('Trade PnL Distribution')
        axes[0, 0].set_xlabel('PnL ($)')
        axes[0, 0].set_ylabel('Frequency')

        # Returns by regime
        if 'entry_regime' in trade_df.columns:
            regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
            for regime in trade_df['entry_regime'].unique():
                regime_trades = trade_df[trade_df['entry_regime'] == regime]
                axes[0, 1].scatter([regime] * len(regime_trades), regime_trades['return_pct'],
                                 alpha=0.6, label=regime_names.get(regime, f'Regime {regime}'))

        axes[0, 1].set_title('Returns by Market Regime')
        axes[0, 1].set_xlabel('Market Regime')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].legend()

        # Duration vs Return
        axes[1, 0].scatter(trade_df['duration'], trade_df['return_pct'], alpha=0.6)
        axes[1, 0].set_title('Trade Duration vs Return')
        axes[1, 0].set_xlabel('Duration (days)')
        axes[1, 0].set_ylabel('Return (%)')

        # Cumulative returns
        cumulative_returns = trade_df['pnl'].cumsum()
        axes[1, 1].plot(range(len(cumulative_returns)), cumulative_returns)
        axes[1, 1].set_title('Cumulative Trade PnL')
        axes[1, 1].set_xlabel('Trade Number')
        axes[1, 1].set_ylabel('Cumulative PnL ($)')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def generate_performance_report(backtest_results: Dict) -> str:
        """Generate comprehensive performance report"""
        metrics = backtest_results['performance_metrics']

        report = f"""
        STRATEGY PERFORMANCE REPORT
        ===========================

        RETURN METRICS:
        - Total Return: {metrics['total_return']:.2f}%
        - Annualized Return: {metrics['annualized_return']:.2f}%
        - Volatility: {metrics['volatility']:.2f}%

        RISK METRICS:
        - Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
        - Sortino Ratio: {metrics['sortino_ratio']:.3f}
        - Calmar Ratio: {metrics['calmar_ratio']:.3f}
        - Maximum Drawdown: {metrics['max_drawdown']:.2f}%

        TRADE METRICS:
        - Total Trades: {metrics['num_trades']}
        - Win Rate: {metrics['win_rate']:.2f}%
        - Average Win: ${metrics['avg_win']:.2f}
        - Average Loss: ${metrics['avg_loss']:.2f}
        - Profit Factor: {metrics['profit_factor']:.2f}
        - Average Trade Duration: {metrics['avg_trade_duration']:.1f} days

        STRATEGY ASSESSMENT:
        """

        # Add strategy assessment
        if metrics['sharpe_ratio'] > 1.0:
            report += "- Excellent risk-adjusted returns\n"
        elif metrics['sharpe_ratio'] > 0.5:
            report += "- Good risk-adjusted returns\n"
        else:
            report += "- Poor risk-adjusted returns - consider optimization\n"

        if metrics['win_rate'] > 60:
            report += "- High win rate indicates good signal quality\n"
        elif metrics['win_rate'] < 40:
            report += "- Low win rate - ensure profit/loss ratio is favorable\n"

        if metrics['max_drawdown'] < -20:
            report += "- High drawdown - consider improved risk management\n"

        return report


def compare_strategies(strategy_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple strategy results

    Args:
        strategy_results: Dictionary of strategy name to backtest results

    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []

    for strategy_name, results in strategy_results.items():
        metrics = results['performance_metrics']
        comparison_data.append({
            'Strategy': strategy_name,
            'Total Return (%)': metrics['total_return'],
            'Sharpe Ratio': metrics['sharpe_ratio'],
            'Max Drawdown (%)': metrics['max_drawdown'],
            'Win Rate (%)': metrics['win_rate'],
            'Num Trades': metrics['num_trades'],
            'Profit Factor': metrics['profit_factor']
        })

    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df.round(3)


if __name__ == "__main__":
    print("Backtesting Utilities")
    print("====================")
    print("Advanced backtesting engine with comprehensive analytics ready for use.")