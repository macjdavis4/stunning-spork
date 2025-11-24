"""
Agent implementations for the FINCON system.

Includes:
- BaseAgent: Abstract base class for all agents
- Analyst agents: NewsAnalyst, FundamentalsAnalyst, DataAnalyst, AudioECCAnalyst, StockSelectionAnalyst
- ManagerAgent: Synthesizes analyst outputs into trading decisions
- RiskControl: CVaR-based risk management and verbal reinforcement
"""

from fincon.agents.base import BaseAgent
from fincon.agents.analyst import (
    NewsAnalystAgent,
    FundamentalsAnalystAgent,
    DataAnalystAgent,
    AudioECCAnalystAgent,
    StockSelectionAnalystAgent,
)
from fincon.agents.manager import ManagerAgent
from fincon.agents.risk_control import RiskControl, RiskState

__all__ = [
    "BaseAgent",
    "NewsAnalystAgent",
    "FundamentalsAnalystAgent",
    "DataAnalystAgent",
    "AudioECCAnalystAgent",
    "StockSelectionAnalystAgent",
    "ManagerAgent",
    "RiskControl",
    "RiskState",
]
