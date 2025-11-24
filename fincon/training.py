"""
Training loop for FINCON system.

Orchestrates the multi-agent system across episodes with:
- Environment interaction
- Analyst execution
- Manager decisions
- Risk control and verbal reinforcement
"""

from datetime import datetime
from typing import Any
import pandas as pd

from fincon.config import FinconConfig
from fincon.data import MarketDataClient, EdgarClient, FinlightNewsClient
from fincon.env import MarketEnvironment
from fincon.llm_backend import LLMBackend, create_llm_backend
from fincon.memory import WorkingMemory, ProceduralMemory, EpisodicMemory
from fincon.agents.analyst import (
    NewsAnalystAgent,
    FundamentalsAnalystAgent,
    DataAnalystAgent,
    AudioECCAnalystAgent,
    StockSelectionAnalystAgent,
)
from fincon.agents.manager import ManagerAgent
from fincon.agents.risk_control import RiskControl
from fincon.evaluation import compute_metrics_summary, print_metrics_summary


class FinconTrainer:
    """
    Main trainer for FINCON multi-agent system.

    Handles:
    - Multi-episode training
    - Agent orchestration
    - Memory management
    - Cross-episode learning
    """

    def __init__(
        self,
        config: FinconConfig,
        market_data_client: MarketDataClient,
        edgar_client: EdgarClient,
        finlight_client: FinlightNewsClient,
        llm_backend: LLMBackend
    ):
        """
        Initialize FINCON trainer.

        Args:
            config: System configuration
            market_data_client: Market data client
            edgar_client: SEC EDGAR client
            finlight_client: News client
            llm_backend: LLM backend
        """
        self.config = config
        self.market_data_client = market_data_client
        self.edgar_client = edgar_client
        self.finlight_client = finlight_client
        self.llm_backend = llm_backend

        # Initialize memory systems
        self.episodic_memory = EpisodicMemory()
        self.procedural_memory = ProceduralMemory()
        self.working_memory = WorkingMemory()

        # Initialize agents
        self._initialize_agents()

        # Training state
        self.episode_results: list[dict[str, Any]] = []

    def _initialize_agents(self) -> None:
        """Initialize all agents."""
        # Analysts
        self.news_analyst = NewsAnalystAgent(
            llm=self.llm_backend,
            news_client=self.finlight_client,
            memory=self.procedural_memory
        ) if self.config.enable_news_analyst else None

        self.fundamentals_analyst = FundamentalsAnalystAgent(
            llm=self.llm_backend,
            edgar_client=self.edgar_client,
            memory=self.procedural_memory
        ) if self.config.enable_fundamentals_analyst else None

        self.data_analyst = DataAnalystAgent(
            llm=self.llm_backend,
            memory=self.procedural_memory
        ) if self.config.enable_data_analyst else None

        self.ecc_analyst = AudioECCAnalystAgent(
            llm=self.llm_backend,
            memory=self.procedural_memory
        ) if self.config.enable_ecc_analyst else None

        self.stock_selection_analyst = StockSelectionAnalystAgent(
            llm=self.llm_backend,
            memory=self.procedural_memory
        ) if self.config.enable_stock_selection_analyst else None

        # Manager
        self.manager = ManagerAgent(
            llm=self.llm_backend,
            episodic_memory=self.episodic_memory,
            procedural_memory=self.procedural_memory,
            max_position_size=self.config.portfolio.max_position_size
        )

        # Risk control
        self.risk_control = RiskControl(
            llm=self.llm_backend,
            episodic_memory=self.episodic_memory,
            cvar_alpha=self.config.cvar_alpha
        )

    def train(self) -> list[dict[str, Any]]:
        """
        Run multi-episode training.

        Returns:
            List of episode results
        """
        print(f"\n{'='*60}")
        print(f"FINCON TRAINING")
        print(f"{'='*60}")
        print(f"Symbols: {self.config.symbols}")
        print(f"Period: {self.config.start_date} to {self.config.end_date}")
        print(f"Episodes: {self.config.episodes}")
        print(f"{'='*60}\n")

        # Fetch market data once
        price_data = self.market_data_client.get_multi_price_history(
            symbols=self.config.symbols,
            start=self.config.start_date,
            end=self.config.end_date
        )

        if not price_data:
            raise ValueError("No market data available for specified symbols and dates")

        # Run episodes
        for episode_id in range(self.config.episodes):
            print(f"\n{'='*60}")
            print(f"EPISODE {episode_id + 1}/{self.config.episodes}")
            print(f"{'='*60}\n")

            episode_result = self._run_episode(episode_id, price_data)
            self.episode_results.append(episode_result)

            # Cross-episode reinforcement
            if episode_id > 0:
                print("\nPerforming cross-episode reinforcement...")
                current_record = self.episodic_memory.get_recent_episodes(1)[0]
                previous_record = self.episodic_memory.get_recent_episodes(2)[0]

                reinforcement = self.risk_control.perform_cross_episode_reinforcement(
                    current_record,
                    previous_record
                )

                print(f"Belief Updates: {reinforcement.get('belief_updates', {})}")
                print(f"Summary: {reinforcement.get('summary', '')}")

        return self.episode_results

    def _run_episode(
        self,
        episode_id: int,
        price_data: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        """
        Run a single episode.

        Args:
            episode_id: Episode number
            price_data: Market price data

        Returns:
            Episode results dictionary
        """
        start_time = datetime.now()

        # Initialize environment
        env = MarketEnvironment(
            price_data=price_data,
            initial_capital=self.config.initial_capital,
            window_size=20
        )

        # Reset
        observation = env.reset()
        self.working_memory.clear()
        self.risk_control.reset_episode()

        done = False
        step_count = 0

        # Determine mode
        mode = "portfolio" if len(self.config.symbols) > 1 else "single"

        # Episode loop
        while not done:
            step_count += 1

            if self.config.verbose:
                print(f"Step {step_count}/{env.max_steps - env.window_size}")

            # Run analysts
            analyst_outputs = self._run_analysts(observation)

            # Update working memory
            self.working_memory.update_state(observation)
            for name, output in analyst_outputs.items():
                self.working_memory.update_analyst_output(name, output)

            # Get risk state
            portfolio_value = env.get_portfolio_value()
            daily_pnl = env.get_daily_pnl()
            if daily_pnl:
                risk_state_obj = self.risk_control.update(daily_pnl[-1], portfolio_value)
                risk_state = {
                    "cvar": risk_state_obj.cvar,
                    "risk_averse": risk_state_obj.risk_averse,
                    "advisory": risk_state_obj.advisory,
                    "max_drawdown": risk_state_obj.max_drawdown
                }
            else:
                risk_state = {
                    "cvar": 0.0,
                    "risk_averse": False,
                    "advisory": "No data yet",
                    "max_drawdown": 0.0
                }

            self.working_memory.update_risk_state(risk_state)

            # Manager decision
            if mode == "single":
                symbol = self.config.symbols[0]
                decision = self.manager.decide_single_asset(
                    symbol=symbol,
                    analyst_outputs=analyst_outputs,
                    risk_state=risk_state
                )
            else:
                decision = self.manager.decide_portfolio(
                    symbols=self.config.symbols,
                    analyst_outputs=analyst_outputs,
                    risk_state=risk_state
                )

            if self.config.verbose:
                print(f"Decision: {decision}")

            # Execute action
            observation, reward, done, info = env.step(decision)

            if self.config.verbose:
                print(f"Reward: {reward:.4f}, Portfolio Value: ${info['portfolio_value']:,.2f}")

        # Episode complete
        end_time = datetime.now()

        # Compute metrics
        equity_curve = env.get_equity_curve()
        returns = pd.Series(env.get_daily_pnl())

        metrics = compute_metrics_summary(returns, equity_curve)

        # Store in episodic memory
        self.episodic_memory.add_episode(
            episode_id=episode_id,
            start_time=start_time,
            end_time=end_time,
            total_return=metrics['cumulative_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            num_trades=len(env.actions_taken),
            actions=env.actions_taken,
            summary=f"Episode {episode_id} completed"
        )

        # Print metrics
        print(f"\nEpisode {episode_id + 1} Results:")
        print(f"  Total Return: {metrics['cumulative_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Total Trades: {metrics['total_trades']}")

        return {
            "episode_id": episode_id,
            "metrics": metrics,
            "equity_curve": equity_curve,
            "returns": returns,
            "actions": env.actions_taken
        }

    def _run_analysts(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Run all enabled analysts.

        Args:
            observation: Environment observation

        Returns:
            Dictionary of analyst name -> output
        """
        outputs = {}

        # Single symbol or first symbol for analysis
        primary_symbol = self.config.symbols[0]

        # News analyst
        if self.news_analyst:
            try:
                print(f"Running NewsAnalyst for {primary_symbol}...")
                news_output = self.news_analyst.analyze(primary_symbol)
                outputs["NewsAnalyst"] = news_output
                print(f"NewsAnalyst complete")
            except Exception as e:
                print(f"News analyst error: {e}")
                outputs["NewsAnalyst"] = {
                    "sentiment_score": 0.0,
                    "key_risks": ["Analysis failed"],
                    "key_opportunities": [],
                    "catalysts": [],
                    "confidence": 0.0
                }

        # Fundamentals analyst
        if self.fundamentals_analyst:
            try:
                print(f"Running FundamentalsAnalyst for {primary_symbol}...")
                fund_output = self.fundamentals_analyst.analyze(primary_symbol)
                outputs["FundamentalsAnalyst"] = fund_output
                print(f"FundamentalsAnalyst complete")
            except Exception as e:
                print(f"Fundamentals analyst error: {e}")
                outputs["FundamentalsAnalyst"] = {
                    "fundamental_view": "neutral",
                    "rationale": "Analysis failed",
                    "fundamental_score": 0.0
                }

        # Data analyst
        if self.data_analyst:
            try:
                print(f"Running DataAnalyst for {primary_symbol}...")
                data_output = self.data_analyst.analyze(primary_symbol, observation)
                outputs["DataAnalyst"] = data_output
                print(f"DataAnalyst complete")
            except Exception as e:
                print(f"Data analyst error: {e}")
                outputs["DataAnalyst"] = {
                    "momentum_score": 0.0,
                    "volatility_regime": "medium",
                    "risk_assessment": "Analysis failed",
                    "data_signal_score": 0.0
                }

        # ECC analyst (if enabled and transcript provided)
        if self.ecc_analyst:
            try:
                ecc_output = self.ecc_analyst.analyze(primary_symbol, transcript="")
                outputs["AudioECCAnalyst"] = ecc_output
            except Exception as e:
                print(f"ECC analyst error: {e}")

        # Stock selection analyst (for portfolio mode)
        if self.stock_selection_analyst and len(self.config.symbols) > 1:
            try:
                selection_output = self.stock_selection_analyst.analyze(
                    self.config.symbols,
                    observation
                )
                outputs["StockSelectionAnalyst"] = selection_output
            except Exception as e:
                print(f"Stock selection analyst error: {e}")

        return outputs

    def get_final_metrics(self) -> dict[str, Any]:
        """
        Get aggregated metrics across all episodes.

        Returns:
            Dictionary with aggregated metrics
        """
        if not self.episode_results:
            return {}

        all_returns = pd.concat([r['returns'] for r in self.episode_results])
        all_equity = pd.concat([r['equity_curve'] for r in self.episode_results])

        final_metrics = compute_metrics_summary(all_returns, all_equity)
        final_metrics['num_episodes'] = len(self.episode_results)

        return final_metrics

    def print_summary(self) -> None:
        """Print summary of all episodes."""
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*60}\n")

        for i, result in enumerate(self.episode_results):
            metrics = result['metrics']
            print(f"Episode {i+1}:")
            print(f"  Return: {metrics['cumulative_return']:>8.2%}")
            print(f"  Sharpe: {metrics['sharpe_ratio']:>8.2f}")
            print(f"  Max DD: {metrics['max_drawdown']:>8.2%}")

        print(f"\n{'='*60}")
        print("AGGREGATED METRICS")
        print(f"{'='*60}")

        final_metrics = self.get_final_metrics()
        print_metrics_summary(final_metrics)


def create_trainer_from_config(config: FinconConfig) -> FinconTrainer:
    """
    Create trainer from configuration.

    Args:
        config: System configuration

    Returns:
        FinconTrainer instance
    """
    # Create clients
    market_data_client = MarketDataClient()

    edgar_client = EdgarClient(
        user_agent=config.edgar.user_agent,
        base_url=config.edgar.base_url
    )

    finlight_client = FinlightNewsClient(
        api_key=config.finlight.api_key,
        base_url=config.finlight.base_url
    )

    # Create LLM backend
    llm_backend = create_llm_backend({
        "model": config.llm.model,
        "api_key": config.llm.api_key,
        "base_url": config.llm.base_url,
        "temperature": config.llm.temperature,
        "max_tokens": config.llm.max_tokens
    })

    # Create trainer
    trainer = FinconTrainer(
        config=config,
        market_data_client=market_data_client,
        edgar_client=edgar_client,
        finlight_client=finlight_client,
        llm_backend=llm_backend
    )

    return trainer
