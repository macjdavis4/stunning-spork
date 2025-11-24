"""
Memory systems for FINCON agents.

Implements:
- WorkingMemory: Short-term context for current decision
- ProceduralMemory: Step-by-step action history with recency weighting
- EpisodicMemory: Episode summaries and conceptual beliefs
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import numpy as np


@dataclass
class MemoryEntry:
    """Single memory entry."""
    timestamp: datetime
    content: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    importance: float = 1.0


class WorkingMemory:
    """
    Short-term working memory for current decision-making context.

    Stores:
    - Current market state
    - Recent analyst outputs
    - Active risk warnings
    """

    def __init__(self):
        """Initialize working memory."""
        self.current_state: dict[str, Any] = {}
        self.analyst_outputs: dict[str, Any] = {}
        self.risk_state: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}

    def update_state(self, state: dict[str, Any]) -> None:
        """
        Update current market state.

        Args:
            state: Market state dictionary
        """
        self.current_state = state

    def update_analyst_output(self, analyst_name: str, output: dict[str, Any]) -> None:
        """
        Update output from an analyst.

        Args:
            analyst_name: Name of the analyst
            output: Analyst output dictionary
        """
        self.analyst_outputs[analyst_name] = output

    def update_risk_state(self, risk_state: dict[str, Any]) -> None:
        """
        Update risk management state.

        Args:
            risk_state: Risk state dictionary
        """
        self.risk_state = risk_state

    def get_context(self) -> dict[str, Any]:
        """
        Get complete working memory context.

        Returns:
            Dictionary with all working memory contents
        """
        return {
            "state": self.current_state,
            "analyst_outputs": self.analyst_outputs,
            "risk_state": self.risk_state,
            "metadata": self.metadata
        }

    def clear(self) -> None:
        """Clear all working memory."""
        self.current_state = {}
        self.analyst_outputs = {}
        self.risk_state = {}
        self.metadata = {}


class ProceduralMemory:
    """
    Procedural memory storing step-by-step action history.

    Implements recency weighting for retrieval.
    """

    def __init__(self, max_size: int = 1000, decay_rate: float = 0.95):
        """
        Initialize procedural memory.

        Args:
            max_size: Maximum number of entries to store
            decay_rate: Decay rate for recency weighting (0-1)
        """
        self.max_size = max_size
        self.decay_rate = decay_rate
        self.entries: list[MemoryEntry] = []

    def add(
        self,
        content: dict[str, Any],
        tags: list[str] | None = None,
        importance: float = 1.0
    ) -> None:
        """
        Add a new memory entry.

        Args:
            content: Memory content
            tags: Optional tags for categorization
            importance: Importance weight (0-1)
        """
        if tags is None:
            tags = []

        entry = MemoryEntry(
            timestamp=datetime.now(),
            content=content,
            tags=tags,
            importance=importance
        )

        self.entries.append(entry)

        # Trim if exceeds max size
        if len(self.entries) > self.max_size:
            self.entries = self.entries[-self.max_size:]

    def retrieve_recent(self, k: int = 10) -> list[dict[str, Any]]:
        """
        Retrieve k most recent entries.

        Args:
            k: Number of entries to retrieve

        Returns:
            List of memory contents
        """
        recent_entries = self.entries[-k:]
        return [entry.content for entry in recent_entries]

    def retrieve_by_tag(self, tag: str, k: int = 10) -> list[dict[str, Any]]:
        """
        Retrieve entries with specific tag.

        Args:
            tag: Tag to filter by
            k: Maximum number of entries to retrieve

        Returns:
            List of memory contents
        """
        tagged_entries = [
            entry for entry in self.entries
            if tag in entry.tags
        ]

        # Sort by timestamp (most recent first)
        tagged_entries.sort(key=lambda x: x.timestamp, reverse=True)

        return [entry.content for entry in tagged_entries[:k]]

    def retrieve_weighted(self, k: int = 10) -> list[tuple[dict[str, Any], float]]:
        """
        Retrieve entries with recency-based weights.

        Args:
            k: Number of entries to retrieve

        Returns:
            List of (content, weight) tuples
        """
        if not self.entries:
            return []

        # Calculate recency weights
        n = len(self.entries)
        weights = np.array([
            self.decay_rate ** (n - i - 1) * entry.importance
            for i, entry in enumerate(self.entries)
        ])

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()

        # Get top k by weight
        top_indices = np.argsort(weights)[-k:][::-1]

        return [
            (self.entries[i].content, weights[i])
            for i in top_indices
        ]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about procedural memory.

        Returns:
            Dictionary with memory statistics
        """
        if not self.entries:
            return {
                "count": 0,
                "tags": {},
                "avg_importance": 0.0
            }

        tag_counts: dict[str, int] = {}
        for entry in self.entries:
            for tag in entry.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "count": len(self.entries),
            "tags": tag_counts,
            "avg_importance": np.mean([e.importance for e in self.entries])
        }

    def clear(self) -> None:
        """Clear all procedural memory."""
        self.entries = []


@dataclass
class EpisodeRecord:
    """Record of a single episode."""
    episode_id: int
    start_time: datetime
    end_time: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    actions: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    belief_updates: dict[str, float] = field(default_factory=dict)


class EpisodicMemory:
    """
    Episodic memory storing episode-level summaries and beliefs.

    Implements conceptual verbal reinforcement across episodes.
    """

    def __init__(self):
        """Initialize episodic memory."""
        self.episodes: list[EpisodeRecord] = []
        self.analyst_belief_weights: dict[str, float] = {}
        self.conceptual_beliefs: list[str] = []

    def add_episode(
        self,
        episode_id: int,
        start_time: datetime,
        end_time: datetime,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        num_trades: int,
        actions: list[dict[str, Any]] | None = None,
        summary: str = "",
        belief_updates: dict[str, float] | None = None
    ) -> None:
        """
        Add a completed episode record.

        Args:
            episode_id: Episode identifier
            start_time: Episode start timestamp
            end_time: Episode end timestamp
            total_return: Total return for episode
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown
            num_trades: Number of trades executed
            actions: List of actions taken
            summary: Natural language summary
            belief_updates: Updated belief weights
        """
        if actions is None:
            actions = []
        if belief_updates is None:
            belief_updates = {}

        record = EpisodeRecord(
            episode_id=episode_id,
            start_time=start_time,
            end_time=end_time,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            num_trades=num_trades,
            actions=actions,
            summary=summary,
            belief_updates=belief_updates
        )

        self.episodes.append(record)

        # Update analyst belief weights
        if belief_updates:
            self.analyst_belief_weights.update(belief_updates)

    def update_beliefs(self, belief_updates: dict[str, float]) -> None:
        """
        Update analyst belief weights.

        Args:
            belief_updates: Dictionary of analyst -> weight updates
        """
        self.analyst_belief_weights.update(belief_updates)

    def add_conceptual_belief(self, belief: str) -> None:
        """
        Add a conceptual belief statement.

        Args:
            belief: Natural language belief statement
        """
        self.conceptual_beliefs.append(belief)

    def get_recent_episodes(self, k: int = 3) -> list[EpisodeRecord]:
        """
        Get k most recent episode records.

        Args:
            k: Number of episodes to retrieve

        Returns:
            List of episode records
        """
        return self.episodes[-k:]

    def get_performance_trend(self) -> dict[str, Any]:
        """
        Analyze performance trend across episodes.

        Returns:
            Dictionary with trend analysis
        """
        if not self.episodes:
            return {
                "trend": "no_data",
                "avg_return": 0.0,
                "avg_sharpe": 0.0,
                "improving": False
            }

        returns = [ep.total_return for ep in self.episodes]
        sharpes = [ep.sharpe_ratio for ep in self.episodes]

        # Check if recent performance is improving
        if len(returns) >= 2:
            recent_return = np.mean(returns[-2:])
            earlier_return = np.mean(returns[:-2]) if len(returns) > 2 else returns[0]
            improving = recent_return > earlier_return
        else:
            improving = False

        return {
            "trend": "improving" if improving else "declining",
            "avg_return": np.mean(returns),
            "avg_sharpe": np.mean(sharpes),
            "improving": improving,
            "num_episodes": len(self.episodes)
        }

    def get_belief_weights(self) -> dict[str, float]:
        """
        Get current analyst belief weights.

        Returns:
            Dictionary of analyst -> weight
        """
        return self.analyst_belief_weights.copy()

    def get_conceptual_beliefs(self) -> list[str]:
        """
        Get all conceptual belief statements.

        Returns:
            List of belief statements
        """
        return self.conceptual_beliefs.copy()

    def clear(self) -> None:
        """Clear all episodic memory."""
        self.episodes = []
        self.analyst_belief_weights = {}
        self.conceptual_beliefs = []
