"""
FINCON: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement
for Enhanced Financial Decision Making

This package implements a multi-agent reinforcement learning system for financial
decision-making combining:
- Real market data (yfinance)
- Fundamental data (SEC EDGAR API)
- News data (finlight.me)
- LLM-based analyst and manager agents
- CVaR-based risk control
- Cross-episode conceptual verbal reinforcement
"""

__version__ = "0.1.0"

from fincon.config import FinconConfig, load_config
from fincon.data import MarketDataClient, EdgarClient, FinlightNewsClient
from fincon.env import MarketEnvironment
from fincon.llm_backend import LLMBackend, OpenAILLMBackend
from fincon.memory import WorkingMemory, ProceduralMemory, EpisodicMemory
from fincon.training import FinconTrainer
from fincon.evaluation import (
    compute_cumulative_return,
    compute_sharpe_ratio,
    compute_max_drawdown,
)

__all__ = [
    "FinconConfig",
    "load_config",
    "MarketDataClient",
    "EdgarClient",
    "FinlightNewsClient",
    "MarketEnvironment",
    "LLMBackend",
    "OpenAILLMBackend",
    "WorkingMemory",
    "ProceduralMemory",
    "EpisodicMemory",
    "FinconTrainer",
    "compute_cumulative_return",
    "compute_sharpe_ratio",
    "compute_max_drawdown",
]
