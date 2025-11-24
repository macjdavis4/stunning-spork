"""
Evaluation metrics for FINCON system.

Provides functions to compute:
- Cumulative returns
- Sharpe ratio
- Maximum drawdown
- Other performance metrics
"""

from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_cumulative_return(returns: pd.Series) -> float:
    """
    Compute cumulative return from a series of returns.

    Args:
        returns: Series of period returns

    Returns:
        Cumulative return as a percentage
    """
    if returns.empty:
        return 0.0

    cumulative = (1 + returns).prod() - 1
    return float(cumulative)


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Series of period returns
        risk_free: Risk-free rate (annualized)
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if returns.empty or len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free / periods_per_year)
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0:
        return 0.0

    sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
    return float(sharpe)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute maximum drawdown from an equity curve.

    Args:
        equity_curve: Series of portfolio values over time

    Returns:
        Maximum drawdown as a negative percentage
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0

    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max

    max_dd = drawdown.min()
    return float(max_dd)


def compute_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Series of period returns
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    if returns.empty:
        return 0.0

    cum_return = compute_cumulative_return(returns)
    annualized_return = (1 + cum_return) ** (periods_per_year / len(returns)) - 1

    equity_curve = (1 + returns).cumprod()
    max_dd = abs(compute_max_drawdown(equity_curve))

    if max_dd == 0:
        return 0.0

    return annualized_return / max_dd


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Compute Sortino ratio (uses downside deviation instead of total volatility).

    Args:
        returns: Series of period returns
        risk_free: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino ratio
    """
    if returns.empty or len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free / periods_per_year)
    mean_excess = excess_returns.mean()

    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0

    if downside_std == 0:
        return 0.0

    sortino = mean_excess / downside_std * np.sqrt(periods_per_year)
    return float(sortino)


def compute_win_rate(returns: pd.Series) -> float:
    """
    Compute win rate (percentage of positive return periods).

    Args:
        returns: Series of period returns

    Returns:
        Win rate as a percentage (0-1)
    """
    if returns.empty:
        return 0.0

    wins = (returns > 0).sum()
    total = len(returns)

    return float(wins / total) if total > 0 else 0.0


def compute_profit_factor(returns: pd.Series) -> float:
    """
    Compute profit factor (total gains / total losses).

    Args:
        returns: Series of period returns

    Returns:
        Profit factor
    """
    if returns.empty:
        return 1.0

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return float('inf') if gains > 0 else 1.0

    return float(gains / losses)


def compute_metrics_summary(
    returns: pd.Series,
    equity_curve: pd.Series | None = None,
    risk_free: float = 0.0,
    periods_per_year: int = 252
) -> dict[str, Any]:
    """
    Compute comprehensive metrics summary.

    Args:
        returns: Series of period returns
        equity_curve: Series of portfolio values (optional, will be computed if None)
        risk_free: Risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Dictionary with all metrics
    """
    if equity_curve is None:
        equity_curve = (1 + returns).cumprod()

    metrics = {
        "cumulative_return": compute_cumulative_return(returns),
        "sharpe_ratio": compute_sharpe_ratio(returns, risk_free, periods_per_year),
        "max_drawdown": compute_max_drawdown(equity_curve),
        "calmar_ratio": compute_calmar_ratio(returns, periods_per_year),
        "sortino_ratio": compute_sortino_ratio(returns, risk_free, periods_per_year),
        "win_rate": compute_win_rate(returns),
        "profit_factor": compute_profit_factor(returns),
        "total_trades": len(returns),
        "mean_return": float(returns.mean()),
        "volatility": float(returns.std()),
    }

    return metrics


def plot_equity_curve(
    equity_curve: pd.Series,
    title: str = "Equity Curve",
    save_path: str | None = None
) -> None:
    """
    Plot equity curve.

    Args:
        equity_curve: Series of portfolio values
        title: Plot title
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve.values, linewidth=2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close()


def plot_drawdown(
    equity_curve: pd.Series,
    title: str = "Drawdown",
    save_path: str | None = None
) -> None:
    """
    Plot drawdown over time.

    Args:
        equity_curve: Series of portfolio values
        title: Plot title
        save_path: Optional path to save plot
    """
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max

    plt.figure(figsize=(12, 6))
    plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    plt.plot(drawdown.index, drawdown.values, color='red', linewidth=2)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close()


def plot_returns_distribution(
    returns: pd.Series,
    title: str = "Returns Distribution",
    save_path: str | None = None
) -> None:
    """
    Plot distribution of returns.

    Args:
        returns: Series of period returns
        title: Plot title
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(12, 6))
    plt.hist(returns, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2%}')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Return', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close()


def print_metrics_summary(metrics: dict[str, Any]) -> None:
    """
    Print formatted metrics summary.

    Args:
        metrics: Dictionary of metrics from compute_metrics_summary
    """
    print("\n" + "="*60)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*60)

    print(f"\nReturns:")
    print(f"  Cumulative Return:    {metrics['cumulative_return']:>10.2%}")
    print(f"  Mean Return:          {metrics['mean_return']:>10.4%}")
    print(f"  Volatility:           {metrics['volatility']:>10.4%}")

    print(f"\nRisk-Adjusted:")
    print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:        {metrics['sortino_ratio']:>10.2f}")
    print(f"  Calmar Ratio:         {metrics['calmar_ratio']:>10.2f}")

    print(f"\nRisk:")
    print(f"  Max Drawdown:         {metrics['max_drawdown']:>10.2%}")

    print(f"\nTrading:")
    print(f"  Total Trades:         {metrics['total_trades']:>10.0f}")
    print(f"  Win Rate:             {metrics['win_rate']:>10.2%}")
    print(f"  Profit Factor:        {metrics['profit_factor']:>10.2f}")

    print("\n" + "="*60 + "\n")
