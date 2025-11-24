"""
Manager agent for FINCON system.

The manager synthesizes inputs from analysts and risk control
to make final trading decisions.
"""

from typing import Any
import numpy as np
from scipy.optimize import minimize

from fincon.agents.base import BaseAgent
from fincon.llm_backend import LLMBackend
from fincon.memory import ProceduralMemory, EpisodicMemory


class ManagerAgent(BaseAgent):
    """
    Manager agent that synthesizes analyst outputs into trading decisions.

    For single-asset: Returns BUY/SELL/HOLD with position size
    For portfolio: Returns portfolio weights using Markowitz optimization
    """

    def __init__(
        self,
        llm: LLMBackend,
        episodic_memory: EpisodicMemory,
        procedural_memory: ProceduralMemory | None = None,
        max_position_size: float = 0.3
    ):
        """
        Initialize manager agent.

        Args:
            llm: LLM backend
            episodic_memory: Episodic memory with beliefs
            procedural_memory: Optional procedural memory
            max_position_size: Maximum position size (0-1)
        """
        super().__init__(
            name="Manager",
            role_description="Synthesizes analyst inputs to make trading decisions",
            llm=llm,
            memory=procedural_memory
        )
        self.episodic_memory = episodic_memory
        self.max_position_size = max_position_size

    def build_system_prompt(self) -> str:
        """Build system prompt for manager."""
        return """You are an expert portfolio manager making trading decisions.

Your task is to synthesize inputs from multiple analysts and make a final trading decision.

You receive:
- News analyst sentiment and catalysts
- Fundamentals analyst view
- Data/technical analyst signals
- Risk control advisories
- Analyst belief weights (learned from past performance)

For SINGLE ASSET decisions, output JSON:
{
  "action": <"BUY"|"SELL"|"HOLD">,
  "position_size": <float 0-1 representing fraction of portfolio>,
  "reasoning": <brief explanation>
}

For PORTFOLIO decisions, output JSON:
{
  "decisions": {
    "<symbol>": <"BUY"|"SELL"|"HOLD">
  },
  "reasoning": <brief explanation>
}

Key principles:
- Weight analyst inputs by their belief weights
- Respect risk control advisories
- Be decisive but risk-aware
- Justify your decisions clearly"""

    def build_messages(
        self,
        analyst_outputs: dict[str, Any],
        belief_weights: dict[str, float],
        risk_state: dict[str, Any],
        mode: str = "single",
        **kwargs
    ) -> list[dict[str, str]]:
        """
        Build messages for manager decision.

        Args:
            analyst_outputs: Dictionary of analyst outputs
            belief_weights: Dictionary of analyst belief weights
            risk_state: Risk control state
            mode: "single" or "portfolio"
            **kwargs: Additional context

        Returns:
            Message list for LLM
        """
        context_text = "ANALYST INPUTS:\n\n"

        # Add analyst outputs with weights
        for analyst_name, output in analyst_outputs.items():
            weight = belief_weights.get(analyst_name, 1.0)
            context_text += f"{analyst_name} (weight: {weight:.2f}):\n"
            context_text += f"  {output}\n\n"

        # Add risk state
        context_text += "RISK CONTROL:\n"
        context_text += f"  CVaR: {risk_state.get('cvar', 'N/A')}\n"
        context_text += f"  Risk Averse: {risk_state.get('risk_averse', False)}\n"
        context_text += f"  Advisory: {risk_state.get('advisory', 'None')}\n\n"

        # Add conceptual beliefs from episodic memory
        conceptual_beliefs = self.episodic_memory.get_conceptual_beliefs()
        if conceptual_beliefs:
            context_text += "LEARNED BELIEFS (from past episodes):\n"
            for belief in conceptual_beliefs[-3:]:
                context_text += f"  - {belief}\n"
            context_text += "\n"

        if mode == "single":
            user_message = f"{context_text}\nMake your trading decision for this asset."
        else:
            user_message = f"{context_text}\nMake your trading decisions for the portfolio."

        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_message}
        ]

    def decide_single_asset(
        self,
        symbol: str,
        analyst_outputs: dict[str, Any],
        risk_state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Make single-asset trading decision.

        Args:
            symbol: Stock symbol
            analyst_outputs: Dictionary of analyst outputs
            risk_state: Risk control state

        Returns:
            Decision dictionary with action, position_size, reasoning
        """
        # Get belief weights
        belief_weights = self.episodic_memory.get_belief_weights()

        # If no weights set, use equal weighting
        if not belief_weights:
            belief_weights = {name: 1.0 for name in analyst_outputs.keys()}

        # Get LLM decision
        response = self.call_llm(
            analyst_outputs=analyst_outputs,
            belief_weights=belief_weights,
            risk_state=risk_state,
            mode="single"
        )

        # Parse response
        try:
            decision = self.parse_json_response(response)
        except ValueError as e:
            print(f"Manager parsing error: {e}")
            decision = {
                "action": "HOLD",
                "position_size": 0.0,
                "reasoning": "Decision parsing failed, defaulting to HOLD"
            }

        # Apply risk constraints
        if risk_state.get('risk_averse', False):
            # Reduce position size if risk averse
            if 'position_size' in decision:
                decision['position_size'] *= 0.5

        # Cap position size
        if 'position_size' in decision:
            decision['position_size'] = min(
                decision['position_size'],
                self.max_position_size
            )

        # Add symbol to decision
        decision['symbol'] = symbol

        # Log to memory
        self.log_to_memory({
            "symbol": symbol,
            "decision": decision,
            "analyst_outputs": analyst_outputs,
            "risk_state": risk_state
        }, tags=["manager_decision", "single_asset"])

        return decision

    def decide_portfolio(
        self,
        symbols: list[str],
        analyst_outputs: dict[str, dict[str, Any]],
        risk_state: dict[str, Any],
        expected_returns: dict[str, float] | None = None,
        covariance_matrix: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        Make portfolio allocation decision.

        Args:
            symbols: List of symbols
            analyst_outputs: Dictionary mapping symbol -> analyst outputs
            risk_state: Risk control state
            expected_returns: Optional expected returns for optimization
            covariance_matrix: Optional covariance matrix

        Returns:
            Decision dictionary with weights and reasoning
        """
        # Get belief weights
        belief_weights = self.episodic_memory.get_belief_weights()

        if not belief_weights:
            belief_weights = {name: 1.0 for name in ["NewsAnalyst", "FundamentalsAnalyst", "DataAnalyst"]}

        # Get LLM directional views
        response = self.call_llm(
            analyst_outputs=analyst_outputs,
            belief_weights=belief_weights,
            risk_state=risk_state,
            mode="portfolio"
        )

        # Parse response
        try:
            decision = self.parse_json_response(response)
            directional_decisions = decision.get("decisions", {})
        except ValueError as e:
            print(f"Manager parsing error: {e}")
            directional_decisions = {symbol: "HOLD" for symbol in symbols}

        # Convert to weights using Markowitz optimization if data available
        if expected_returns and covariance_matrix is not None:
            weights = self._optimize_portfolio(
                symbols,
                directional_decisions,
                expected_returns,
                covariance_matrix,
                risk_state
            )
        else:
            # Simple equal weighting
            weights = self._simple_weights(symbols, directional_decisions)

        result = {
            "weights": weights,
            "reasoning": decision.get("reasoning", "Portfolio allocation based on analyst consensus")
        }

        # Log to memory
        self.log_to_memory({
            "symbols": symbols,
            "decision": result,
            "directional_decisions": directional_decisions
        }, tags=["manager_decision", "portfolio"])

        return result

    def _simple_weights(
        self,
        symbols: list[str],
        directional_decisions: dict[str, str]
    ) -> dict[str, float]:
        """
        Generate simple equal weights respecting directional decisions.

        Args:
            symbols: List of symbols
            directional_decisions: Dictionary of symbol -> BUY/SELL/HOLD

        Returns:
            Dictionary of symbol -> weight
        """
        weights = {}

        # Count BUY decisions
        buy_symbols = [s for s in symbols if directional_decisions.get(s) == "BUY"]

        if buy_symbols:
            weight_per_symbol = 1.0 / len(buy_symbols)
            for symbol in symbols:
                if symbol in buy_symbols:
                    weights[symbol] = weight_per_symbol
                else:
                    weights[symbol] = 0.0
        else:
            # All HOLD or SELL, stay in cash
            weights = {symbol: 0.0 for symbol in symbols}

        return weights

    def _optimize_portfolio(
        self,
        symbols: list[str],
        directional_decisions: dict[str, str],
        expected_returns: dict[str, float],
        covariance_matrix: np.ndarray,
        risk_state: dict[str, Any]
    ) -> dict[str, float]:
        """
        Optimize portfolio using Markowitz mean-variance optimization.

        Args:
            symbols: List of symbols
            directional_decisions: Dictionary of symbol -> BUY/SELL/HOLD
            expected_returns: Expected returns for each symbol
            covariance_matrix: Covariance matrix of returns
            risk_state: Risk control state

        Returns:
            Dictionary of symbol -> optimal weight
        """
        n = len(symbols)

        # Expected returns vector
        mu = np.array([expected_returns.get(s, 0.0) for s in symbols])

        # Risk aversion parameter
        risk_aversion = 2.0
        if risk_state.get('risk_averse', False):
            risk_aversion = 4.0

        # Objective: maximize return - risk_aversion * variance
        def objective(w):
            portfolio_return = w @ mu
            portfolio_variance = w @ covariance_matrix @ w
            return -(portfolio_return - risk_aversion * portfolio_variance)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1.0}  # Weights sum to 1
        ]

        # Bounds based on directional decisions
        bounds = []
        for symbol in symbols:
            decision = directional_decisions.get(symbol, "HOLD")
            if decision == "BUY":
                bounds.append((0.0, self.max_position_size))
            elif decision == "SELL":
                bounds.append((-self.max_position_size, 0.0))
            else:  # HOLD
                bounds.append((0.0, 0.0))

        # Initial guess: equal weights for BUY symbols
        w0 = np.zeros(n)
        buy_indices = [i for i, s in enumerate(symbols) if directional_decisions.get(s) == "BUY"]
        if buy_indices:
            w0[buy_indices] = 1.0 / len(buy_indices)

        # Optimize
        try:
            result = minimize(
                objective,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                optimal_weights = result.x
            else:
                optimal_weights = w0
        except Exception as e:
            print(f"Optimization failed: {e}, using simple weights")
            optimal_weights = w0

        # Return as dictionary
        return {symbol: float(w) for symbol, w in zip(symbols, optimal_weights)}
