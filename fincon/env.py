"""
Market environment (POMDP) for FINCON system.

Implements a trading environment with:
- Time-series market data
- Portfolio state tracking
- Reward computation based on PnL
- Support for single-asset and multi-asset trading
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class PortfolioState:
    """Current portfolio state."""
    cash: float
    holdings: dict[str, float]  # symbol -> number of shares
    equity_value: float
    total_value: float


class MarketEnvironment:
    """
    POMDP environment for market trading.

    State space:
    - Price history window
    - Returns, volatility, technical indicators
    - Portfolio holdings

    Action space:
    - Single asset: BUY/SELL/HOLD with position size
    - Multi asset: Portfolio weights vector

    Reward:
    - Daily PnL (change in portfolio value)
    """

    def __init__(
        self,
        price_data: dict[str, pd.DataFrame],
        initial_capital: float = 100000.0,
        window_size: int = 20,
        transaction_cost: float = 0.001
    ):
        """
        Initialize market environment.

        Args:
            price_data: Dict mapping symbol -> OHLCV DataFrame
            initial_capital: Starting capital in dollars
            window_size: Size of observation window
            transaction_cost: Transaction cost as fraction of trade value
        """
        self.price_data = price_data
        self.symbols = list(price_data.keys())
        self.initial_capital = initial_capital
        self.window_size = window_size
        self.transaction_cost = transaction_cost

        # Align all dataframes to common date index
        self._align_data()

        # Trading state
        self.current_step = 0
        self.max_steps = len(self.dates) - 1

        # Portfolio state
        self.cash = initial_capital
        self.holdings: dict[str, float] = {symbol: 0.0 for symbol in self.symbols}

        # History tracking
        self.portfolio_values: list[float] = [initial_capital]
        self.actions_taken: list[dict[str, Any]] = []

    def _align_data(self) -> None:
        """Align all price dataframes to common date index."""
        # Get intersection of all dates
        common_dates = None
        for df in self.price_data.values():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates &= set(df.index)

        if not common_dates:
            raise ValueError("No common dates found across all symbols")

        self.dates = sorted(list(common_dates))

        # Validate sufficient data for window size
        min_required_dates = self.window_size + 1
        if len(self.dates) < min_required_dates:
            raise ValueError(
                f"Insufficient data: {len(self.dates)} trading days available, "
                f"but need at least {min_required_dates} days for window_size={self.window_size}. "
                f"Please either:\n"
                f"  1. Increase your date range to include more trading days\n"
                f"  2. Reduce window_size to {len(self.dates) - 1} or less"
            )

        # Reindex all dataframes
        for symbol in self.symbols:
            self.price_data[symbol] = self.price_data[symbol].loc[self.dates]

    def reset(self) -> dict[str, Any]:
        """
        Reset environment to initial state.

        Returns:
            Initial observation
        """
        self.current_step = self.window_size
        self.cash = self.initial_capital
        self.holdings = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_values = [self.initial_capital]
        self.actions_taken = []

        return self._get_observation()

    def _get_observation(self) -> dict[str, Any]:
        """
        Get current observation.

        Returns:
            Dictionary with market state and portfolio state
        """
        obs = {
            "date": self.dates[self.current_step],
            "step": self.current_step,
            "symbols": {}
        }

        for symbol in self.symbols:
            df = self.price_data[symbol]

            # Get price window
            start_idx = max(0, self.current_step - self.window_size)
            end_idx = self.current_step + 1
            window = df.iloc[start_idx:end_idx]

            # Compute features
            close_prices = window["Close"].values
            returns = np.diff(close_prices) / close_prices[:-1] if len(close_prices) > 1 else np.array([0.0])

            obs["symbols"][symbol] = {
                "close": float(close_prices[-1]),
                "open": float(window["Open"].iloc[-1]),
                "high": float(window["High"].iloc[-1]),
                "low": float(window["Low"].iloc[-1]),
                "volume": float(window["Volume"].iloc[-1]),
                "returns": returns.tolist(),
                "volatility": float(np.std(returns)) if len(returns) > 1 else 0.0,
                "mean_return": float(np.mean(returns)) if len(returns) > 0 else 0.0,
                "sma_5": float(close_prices[-5:].mean()) if len(close_prices) >= 5 else float(close_prices[-1]),
                "sma_20": float(close_prices.mean()),
                "price_change": float((close_prices[-1] - close_prices[0]) / close_prices[0]) if len(close_prices) > 1 else 0.0,
            }

        # Portfolio state
        obs["portfolio"] = self._get_portfolio_state()

        return obs

    def _get_portfolio_state(self) -> dict[str, Any]:
        """
        Get current portfolio state.

        Returns:
            Dictionary with portfolio information
        """
        equity_value = 0.0

        for symbol, shares in self.holdings.items():
            current_price = self.price_data[symbol].iloc[self.current_step]["Close"]
            equity_value += shares * current_price

        total_value = self.cash + equity_value

        return {
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "equity_value": equity_value,
            "total_value": total_value,
            "returns": (total_value - self.initial_capital) / self.initial_capital
        }

    def step(
        self,
        action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Execute action and advance environment.

        Args:
            action: Action dictionary with:
                - Single asset: {"action": "BUY"|"SELL"|"HOLD", "position_size": float, "symbol": str}
                - Multi asset: {"weights": dict[str, float]}

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0.0, True, {"error": "Episode finished"}

        # Record portfolio value before action
        prev_value = self._get_portfolio_value()

        # Execute action
        if "weights" in action:
            self._execute_portfolio_action(action["weights"])
        else:
            self._execute_single_action(action)

        # Advance time
        self.current_step += 1

        # Record portfolio value after action
        new_value = self._get_portfolio_value()
        self.portfolio_values.append(new_value)

        # Compute reward (daily PnL)
        reward = (new_value - prev_value) / prev_value

        # Check if done
        done = self.current_step >= self.max_steps

        # Get new observation
        obs = self._get_observation()

        # Info
        info = {
            "portfolio_value": new_value,
            "daily_return": reward,
            "cash": self.cash,
            "equity_value": new_value - self.cash
        }

        self.actions_taken.append({
            "step": self.current_step,
            "action": action,
            "reward": reward,
            "portfolio_value": new_value
        })

        return obs, reward, done, info

    def _execute_single_action(self, action: dict[str, Any]) -> None:
        """
        Execute single-asset action.

        Args:
            action: Dictionary with action, position_size, and symbol
        """
        action_type = action.get("action", "HOLD")
        position_size = action.get("position_size", 0.0)
        symbol = action.get("symbol", self.symbols[0])

        if symbol not in self.symbols:
            return

        current_price = self.price_data[symbol].iloc[self.current_step]["Close"]
        portfolio_value = self._get_portfolio_value()

        if action_type == "BUY" and position_size > 0:
            # Buy shares
            target_value = portfolio_value * abs(position_size)
            shares_to_buy = target_value / current_price
            cost = shares_to_buy * current_price * (1 + self.transaction_cost)

            if cost <= self.cash:
                self.holdings[symbol] += shares_to_buy
                self.cash -= cost

        elif action_type == "SELL" and position_size > 0:
            # Sell shares
            shares_to_sell = min(
                self.holdings[symbol],
                portfolio_value * abs(position_size) / current_price
            )

            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.holdings[symbol] -= shares_to_sell
                self.cash += proceeds

    def _execute_portfolio_action(self, weights: dict[str, float]) -> None:
        """
        Execute portfolio rebalancing action.

        Args:
            weights: Dictionary mapping symbol -> target weight
        """
        portfolio_value = self._get_portfolio_value()

        # Liquidate all current holdings
        for symbol in self.symbols:
            if self.holdings[symbol] > 0:
                current_price = self.price_data[symbol].iloc[self.current_step]["Close"]
                proceeds = self.holdings[symbol] * current_price * (1 - self.transaction_cost)
                self.cash += proceeds
                self.holdings[symbol] = 0.0

        # Allocate according to target weights
        for symbol, weight in weights.items():
            if symbol not in self.symbols or abs(weight) < 0.01:
                continue

            current_price = self.price_data[symbol].iloc[self.current_step]["Close"]
            target_value = portfolio_value * abs(weight)

            if weight > 0:  # Long position
                shares = target_value / current_price
                cost = shares * current_price * (1 + self.transaction_cost)

                if cost <= self.cash:
                    self.holdings[symbol] = shares
                    self.cash -= cost

    def _get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value.

        Returns:
            Total portfolio value in dollars
        """
        equity_value = 0.0

        for symbol, shares in self.holdings.items():
            current_price = self.price_data[symbol].iloc[self.current_step]["Close"]
            equity_value += shares * current_price

        return self.cash + equity_value

    def get_portfolio_value(self) -> float:
        """
        Get current portfolio value (public method).

        Returns:
            Total portfolio value
        """
        return self._get_portfolio_value()

    def get_daily_pnl(self) -> list[float]:
        """
        Get daily PnL series.

        Returns:
            List of daily returns
        """
        if len(self.portfolio_values) < 2:
            return []

        pnl = []
        for i in range(1, len(self.portfolio_values)):
            daily_return = (
                (self.portfolio_values[i] - self.portfolio_values[i-1])
                / self.portfolio_values[i-1]
            )
            pnl.append(daily_return)

        return pnl

    def get_equity_curve(self) -> pd.Series:
        """
        Get equity curve as pandas Series.

        Returns:
            Series indexed by date with portfolio values
        """
        dates = self.dates[self.window_size:self.window_size + len(self.portfolio_values)]
        return pd.Series(self.portfolio_values, index=dates)
