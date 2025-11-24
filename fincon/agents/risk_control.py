"""
Risk control for FINCON system.

Implements:
- Within-episode CVaR-based risk management
- Cross-episode conceptual verbal reinforcement
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import numpy as np

from fincon.llm_backend import LLMBackend
from fincon.memory import EpisodicMemory, EpisodeRecord


@dataclass
class RiskState:
    """Current risk state."""
    cvar: float
    risk_averse: bool
    advisory: str
    max_drawdown: float
    current_drawdown: float


class RiskControl:
    """
    Risk control system with CVaR monitoring and verbal reinforcement.

    Within-episode: Monitors CVaR and issues risk advisories
    Cross-episode: Uses LLM to generate belief updates from performance
    """

    def __init__(
        self,
        llm: LLMBackend,
        episodic_memory: EpisodicMemory,
        cvar_alpha: float = 0.05,
        cvar_threshold: float = -0.03,
        drawdown_threshold: float = 0.1
    ):
        """
        Initialize risk control.

        Args:
            llm: LLM backend for verbal reinforcement
            episodic_memory: Episodic memory to store beliefs
            cvar_alpha: Confidence level for CVaR (default 5%)
            cvar_threshold: CVaR threshold triggering risk aversion
            drawdown_threshold: Drawdown threshold for risk aversion
        """
        self.llm = llm
        self.episodic_memory = episodic_memory
        self.cvar_alpha = cvar_alpha
        self.cvar_threshold = cvar_threshold
        self.drawdown_threshold = drawdown_threshold

        # Episode state
        self.daily_pnls: list[float] = []
        self.equity_curve: list[float] = []
        self.risk_averse = False

    def reset_episode(self) -> None:
        """Reset for new episode."""
        self.daily_pnls = []
        self.equity_curve = []
        self.risk_averse = False

    def update(self, daily_pnl: float, portfolio_value: float) -> RiskState:
        """
        Update risk state with new PnL.

        Args:
            daily_pnl: Daily profit/loss as percentage
            portfolio_value: Current portfolio value

        Returns:
            Updated risk state
        """
        self.daily_pnls.append(daily_pnl)
        self.equity_curve.append(portfolio_value)

        # Compute CVaR
        cvar = self._compute_cvar()

        # Compute drawdown
        max_drawdown, current_drawdown = self._compute_drawdown()

        # Determine risk aversion
        self.risk_averse = (
            cvar < self.cvar_threshold or
            current_drawdown > self.drawdown_threshold
        )

        # Generate advisory
        if self.risk_averse:
            if cvar < self.cvar_threshold:
                advisory = f"CVaR ({cvar:.2%}) below threshold. Reduce risk exposure."
            else:
                advisory = f"Drawdown ({current_drawdown:.2%}) exceeds threshold. Reduce risk exposure."
        else:
            advisory = "Risk levels normal."

        return RiskState(
            cvar=cvar,
            risk_averse=self.risk_averse,
            advisory=advisory,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown
        )

    def _compute_cvar(self) -> float:
        """
        Compute Conditional Value at Risk (CVaR).

        Returns:
            CVaR at alpha confidence level
        """
        if len(self.daily_pnls) < 2:
            return 0.0

        pnls = np.array(self.daily_pnls)

        # Compute VaR (alpha-quantile)
        var = np.quantile(pnls, self.cvar_alpha)

        # CVaR is mean of values below VaR
        cvar_values = pnls[pnls <= var]

        if len(cvar_values) > 0:
            return float(np.mean(cvar_values))
        else:
            return var

    def _compute_drawdown(self) -> tuple[float, float]:
        """
        Compute maximum and current drawdown.

        Returns:
            Tuple of (max_drawdown, current_drawdown)
        """
        if len(self.equity_curve) < 2:
            return 0.0, 0.0

        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max

        max_drawdown = float(np.min(drawdown))
        current_drawdown = float(drawdown[-1])

        return max_drawdown, current_drawdown

    def get_risk_state(self) -> dict[str, Any]:
        """
        Get current risk state as dictionary.

        Returns:
            Risk state dictionary
        """
        if not self.daily_pnls:
            return {
                "cvar": 0.0,
                "risk_averse": False,
                "advisory": "No data yet",
                "max_drawdown": 0.0,
                "current_drawdown": 0.0
            }

        state = self.update(self.daily_pnls[-1], self.equity_curve[-1])

        return {
            "cvar": state.cvar,
            "risk_averse": state.risk_averse,
            "advisory": state.advisory,
            "max_drawdown": state.max_drawdown,
            "current_drawdown": state.current_drawdown
        }

    def perform_cross_episode_reinforcement(
        self,
        current_episode: EpisodeRecord,
        previous_episode: EpisodeRecord | None = None
    ) -> dict[str, Any]:
        """
        Perform cross-episode verbal reinforcement.

        Compares current episode to previous and generates:
        - Natural language summary of what worked/failed
        - Updated belief weights for analysts
        - Conceptual beliefs for future episodes

        Args:
            current_episode: Current episode record
            previous_episode: Previous episode record (if available)

        Returns:
            Dictionary with updated beliefs and summary
        """
        # Build comparison context
        if previous_episode:
            comparison_text = self._build_episode_comparison(current_episode, previous_episode)
        else:
            comparison_text = self._build_single_episode_summary(current_episode)

        # Get LLM verbal reinforcement
        response = self._call_llm_for_reinforcement(comparison_text)

        # Parse response
        try:
            reinforcement = self._parse_reinforcement_response(response)
        except Exception as e:
            print(f"Reinforcement parsing error: {e}")
            reinforcement = {
                "summary": "Analysis failed",
                "belief_updates": {},
                "conceptual_beliefs": []
            }

        # Update episodic memory
        if reinforcement.get("belief_updates"):
            self.episodic_memory.update_beliefs(reinforcement["belief_updates"])

        for belief in reinforcement.get("conceptual_beliefs", []):
            self.episodic_memory.add_conceptual_belief(belief)

        return reinforcement

    def _build_episode_comparison(
        self,
        current: EpisodeRecord,
        previous: EpisodeRecord
    ) -> str:
        """Build comparison text between episodes."""
        text = "EPISODE COMPARISON:\n\n"

        text += f"Previous Episode (#{previous.episode_id}):\n"
        text += f"  Total Return: {previous.total_return:.2%}\n"
        text += f"  Sharpe Ratio: {previous.sharpe_ratio:.2f}\n"
        text += f"  Max Drawdown: {previous.max_drawdown:.2%}\n"
        text += f"  Number of Trades: {previous.num_trades}\n\n"

        text += f"Current Episode (#{current.episode_id}):\n"
        text += f"  Total Return: {current.total_return:.2%}\n"
        text += f"  Sharpe Ratio: {current.sharpe_ratio:.2f}\n"
        text += f"  Max Drawdown: {current.max_drawdown:.2%}\n"
        text += f"  Number of Trades: {current.num_trades}\n\n"

        # Performance change
        return_change = current.total_return - previous.total_return
        sharpe_change = current.sharpe_ratio - previous.sharpe_ratio

        text += f"Performance Change:\n"
        text += f"  Return: {return_change:+.2%}\n"
        text += f"  Sharpe: {sharpe_change:+.2f}\n\n"

        return text

    def _build_single_episode_summary(self, episode: EpisodeRecord) -> str:
        """Build summary for single episode."""
        text = f"EPISODE SUMMARY (#{episode.episode_id}):\n\n"

        text += f"Total Return: {episode.total_return:.2%}\n"
        text += f"Sharpe Ratio: {episode.sharpe_ratio:.2f}\n"
        text += f"Max Drawdown: {episode.max_drawdown:.2%}\n"
        text += f"Number of Trades: {episode.num_trades}\n\n"

        return text

    def _call_llm_for_reinforcement(self, context: str) -> str:
        """Call LLM to generate verbal reinforcement."""
        system_prompt = """You are an expert trading coach analyzing episode performance.

Your task is to:
1. Identify what strategies worked and what failed
2. Suggest how to adjust reliance on different analysts
3. Provide conceptual beliefs for future episodes

Output JSON:
{
  "summary": <what worked and what failed>,
  "belief_updates": {
    "NewsAnalyst": <new weight 0-2>,
    "FundamentalsAnalyst": <new weight 0-2>,
    "DataAnalyst": <new weight 0-2>
  },
  "conceptual_beliefs": [<list of belief statements>]
}

Weights interpretation:
- 0.0-0.5: Low confidence, reduce reliance
- 0.5-1.0: Moderate confidence
- 1.0-1.5: High confidence, standard reliance
- 1.5-2.0: Very high confidence, increase reliance

Be specific and actionable."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]

        return self.llm.chat(messages, temperature=0.5)

    def _parse_reinforcement_response(self, response: str) -> dict[str, Any]:
        """Parse LLM reinforcement response."""
        import json

        # Try to extract JSON
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()

        # Parse JSON
        try:
            data = json.loads(response)
            return data
        except json.JSONDecodeError:
            # Try to find JSON object
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                data = json.loads(response[start_idx:end_idx])
                return data

            raise ValueError(f"Could not parse reinforcement response: {response[:200]}")
