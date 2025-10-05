"""
Hybrid ML Trading Pipeline Demo

This script demonstrates how to use the complete hybrid trading pipeline
combining machine learning regime classification with rule-based strategies.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add pipeline to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_pipeline import HybridTradingPipeline, run_multi_timeframe_analysis
from regime_classifier import train_multiple_models
from trading_strategies import create_default_strategies
from backtesting import BacktestEngine, PerformanceAnalyzer, compare_strategies


def load_sample_data(data_dir: str = "../data", timeframe: str = "1min") -> pd.DataFrame:
    """Load sample data for demonstration"""

    if timeframe == "1min":
        filename = "BTCUSD_2023_1min_cleaned.csv"
    else:
        filename = f"BTCUSD_2023_{timeframe}_cleaned.csv"

    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"Data file not found: {filepath}")
        print("Available files:")
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith('.csv'):
                    print(f"  - {f}")
        return None

    # Load data
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    # Clean unnecessary columns
    cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    print(f"Loaded {timeframe} data: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


def demo_regime_classification(df: pd.DataFrame):
    """Demonstrate regime classification capabilities"""

    print("\n" + "="*60)
    print("DEMO: MARKET REGIME CLASSIFICATION")
    print("="*60)

    # Train multiple models and compare
    print("Training multiple regime classification models...")
    model_results = train_multiple_models(df, test_size=0.3)

    # Show results
    print("\nModel Comparison:")
    print("-" * 40)
    for model_name, result in model_results.items():
        accuracy = result['eval_metrics']['accuracy']
        cv_score = result['train_metrics']['cv_mean']
        print(f"{model_name:>15}: Accuracy {accuracy:.3f} | CV {cv_score:.3f}")

    # Find best model
    best_model_name = max(model_results.keys(),
                         key=lambda k: model_results[k]['eval_metrics']['accuracy'])

    print(f"\nBest model: {best_model_name}")

    return model_results[best_model_name]['pipeline']


def demo_trading_strategies():
    """Demonstrate trading strategy components"""

    print("\n" + "="*60)
    print("DEMO: TRADING STRATEGIES")
    print("="*60)

    # Create strategy manager with default strategies
    strategy_manager = create_default_strategies()

    print("Created default strategies:")
    for name, strategy in strategy_manager.strategies.items():
        params = strategy.get_parameters()
        print(f"  - {name}: {strategy.name}")
        print(f"    Regime: {strategy.regime_type}, Stop Loss: {params.get('stop_loss', 'N/A')}")

    return strategy_manager


def demo_hybrid_pipeline(df: pd.DataFrame):
    """Demonstrate the complete hybrid pipeline"""

    print("\n" + "="*60)
    print("DEMO: HYBRID ML PIPELINE")
    print("="*60)

    # Create pipeline
    pipeline = HybridTradingPipeline()

    # Train regime classifier
    print("Training regime classifier...")
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size]

    train_result = pipeline.train_regime_classifier(train_df, model_type='RandomForest')

    # Run backtest on full dataset
    print("Running backtest...")
    backtest_result = pipeline.backtest(df, initial_capital=10000)

    # Show results
    metrics = backtest_result['performance_metrics']
    print(f"\nBacktest Results:")
    print(f"  - Total Return: {metrics['total_return']:.2f}%")
    print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  - Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"  - Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  - Number of Trades: {metrics['num_trades']}")

    return pipeline, backtest_result


def demo_backtesting_engine(df: pd.DataFrame, pipeline: HybridTradingPipeline):
    """Demonstrate advanced backtesting capabilities"""

    print("\n" + "="*60)
    print("DEMO: ADVANCED BACKTESTING")
    print("="*60)

    # Generate signals
    signals, regime_predictions = pipeline.generate_signals(df)

    print(f"Generated {len(signals)} signals")

    # Test different position sizing methods
    backtest_engines = {
        'Fixed Position': BacktestEngine(commission=0.001),
        'Low Commission': BacktestEngine(commission=0.0005),
        'High Slippage': BacktestEngine(commission=0.001, slippage=0.0005)
    }

    results = {}

    for name, engine in backtest_engines.items():
        print(f"Testing {name}...")
        result = engine.run_backtest(df, signals)
        results[name] = result

    # Compare results
    comparison = compare_strategies(results)
    print("\nBacktest Comparison:")
    print(comparison.to_string(index=False))

    return results


def demo_performance_analysis(backtest_results: dict):
    """Demonstrate performance analysis and visualization"""

    print("\n" + "="*60)
    print("DEMO: PERFORMANCE ANALYSIS")
    print("="*60)

    # Generate performance report for best strategy
    best_strategy = max(backtest_results.keys(),
                       key=lambda k: backtest_results[k]['performance_metrics']['total_return'])

    print(f"Best performing configuration: {best_strategy}")

    # Generate detailed report
    report = PerformanceAnalyzer.generate_performance_report(backtest_results[best_strategy])
    print(report)

    return best_strategy


def demo_multi_timeframe_analysis():
    """Demonstrate multi-timeframe analysis"""

    print("\n" + "="*60)
    print("DEMO: MULTI-TIMEFRAME ANALYSIS")
    print("="*60)

    # Run analysis across multiple timeframes
    data_dir = "../data"
    timeframes = ['15min', '30min', '1day']  # Skip 1min for demo speed

    if os.path.exists(data_dir):
        print("Running multi-timeframe analysis...")
        results = run_multi_timeframe_analysis(data_dir, timeframes, initial_capital=10000)

        if results:
            print(f"\nAnalysis completed for {len(results)} timeframes")

            # Create summary table
            summary_data = []
            for tf, result in results.items():
                metrics = result['backtest_result']['performance_metrics']
                summary_data.append({
                    'Timeframe': tf,
                    'Return (%)': f"{metrics['total_return']:.2f}",
                    'Sharpe': f"{metrics['sharpe_ratio']:.3f}",
                    'Max DD (%)': f"{metrics['max_drawdown']:.2f}",
                    'Trades': metrics['num_trades'],
                    'Win Rate (%)': f"{metrics['win_rate']:.1f}"
                })

            summary_df = pd.DataFrame(summary_data)
            print("\nMulti-Timeframe Results:")
            print(summary_df.to_string(index=False))

            return results
        else:
            print("No results generated - check data availability")
    else:
        print(f"Data directory not found: {data_dir}")

    return None


def save_demo_results(pipeline: HybridTradingPipeline, results: dict):
    """Save demonstration results"""

    print("\n" + "="*60)
    print("SAVING DEMO RESULTS")
    print("="*60)

    # Create demo results directory
    demo_dir = "demo_results"
    os.makedirs(demo_dir, exist_ok=True)

    # Save pipeline
    pipeline_dir = os.path.join(demo_dir, "hybrid_pipeline")
    pipeline.save_pipeline(pipeline_dir)

    # Save backtest results
    if results:
        best_result = max(results.items(), key=lambda x: x[1]['performance_metrics']['total_return'])
        result_name, result_data = best_result

        # Save equity curve
        equity_path = os.path.join(demo_dir, "equity_curve.csv")
        result_data['equity_curve'].to_csv(equity_path)

        # Save trades
        if result_data['trades']:
            trades_data = []
            for trade in result_data['trades']:
                trades_data.append({
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'pnl': trade.pnl,
                    'return_pct': trade.return_pct,
                    'duration': trade.duration,
                    'strategy': trade.strategy_name
                })

            trades_df = pd.DataFrame(trades_data)
            trades_path = os.path.join(demo_dir, "trades.csv")
            trades_df.to_csv(trades_path, index=False)

            print(f"Demo results saved to: {demo_dir}")
            print(f"  - Pipeline: {pipeline_dir}")
            print(f"  - Equity curve: {equity_path}")
            print(f"  - Trades: {trades_path}")


def main():
    """Run complete demonstration"""

    print("HYBRID ML TRADING PIPELINE DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the complete hybrid trading system combining")
    print("machine learning regime classification with rule-based strategies.")
    print("=" * 80)

    try:
        # 1. Load sample data
        print("Loading sample data...")
        df = load_sample_data(timeframe="15min")  # Use 15min for faster demo

        if df is None:
            print("Could not load data. Please ensure cleaned datasets are available.")
            return

        # Use subset for faster demo
        df_sample = df.tail(5000)  # Last 5000 records
        print(f"Using sample data: {df_sample.shape}")

        # 2. Demonstrate regime classification
        best_classifier = demo_regime_classification(df_sample)

        # 3. Demonstrate trading strategies
        strategy_manager = demo_trading_strategies()

        # 4. Demonstrate hybrid pipeline
        pipeline, pipeline_result = demo_hybrid_pipeline(df_sample)

        # 5. Demonstrate advanced backtesting
        backtest_results = demo_backtesting_engine(df_sample, pipeline)

        # 6. Demonstrate performance analysis
        best_strategy = demo_performance_analysis(backtest_results)

        # 7. Save results
        save_demo_results(pipeline, backtest_results)

        # 8. Multi-timeframe analysis (optional)
        print("\nWould you like to run multi-timeframe analysis? (This may take longer)")
        print("Skipping for demo - use run_multi_timeframe_analysis() directly if needed.")

        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("The hybrid ML trading pipeline is ready for use.")
        print("Check the generated demo_results/ directory for saved outputs.")
        print("="*80)

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()