"""
Configuration management for FINCON system using Pydantic.

Handles loading from JSON files with environment variable fallbacks.
"""

import json
import os
from typing import Any
from pydantic import BaseModel, Field


class MarketDataConfig(BaseModel):
    """Configuration for market data provider."""
    provider: str = "yfinance"


class EdgarConfig(BaseModel):
    """Configuration for SEC EDGAR API access."""
    user_agent: str = Field(
        default_factory=lambda: os.getenv(
            "EDGAR_USER_AGENT",
            "FinconSystem research@example.com"
        )
    )
    base_url: str = "https://data.sec.gov"


class FinlightConfig(BaseModel):
    """Configuration for Finlight.me news API."""
    api_key: str = Field(
        default_factory=lambda: os.getenv("FINLIGHT_API_KEY", "")
    )
    base_url: str = "https://api.finlight.me/v1"


class LLMConfig(BaseModel):
    """Configuration for LLM backend."""
    model: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4")
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv(
            "OPENAI_BASE_URL",
            "https://api.openai.com/v1"
        )
    )
    temperature: float = 0.3
    max_tokens: int = 2048


class PortfolioConfig(BaseModel):
    """Configuration for portfolio optimization."""
    max_position_size: float = 0.3
    min_position_size: float = 0.05
    risk_free_rate: float = 0.02


class FinconConfig(BaseModel):
    """Main configuration for FINCON system."""

    # Trading universe
    symbols: list[str] = Field(default_factory=lambda: ["AAPL"])
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"

    # Episode and learning configuration
    episodes: int = 3
    initial_capital: float = 100000.0

    # Risk management
    cvar_alpha: float = 0.05
    discount_factor: float = 1.0

    # Sub-configurations
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    edgar: EdgarConfig = Field(default_factory=EdgarConfig)
    finlight: FinlightConfig = Field(default_factory=FinlightConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)

    # Agent configuration
    enable_news_analyst: bool = True
    enable_fundamentals_analyst: bool = True
    enable_data_analyst: bool = True
    enable_ecc_analyst: bool = False
    enable_stock_selection_analyst: bool = False

    # Verbosity
    verbose: bool = False


def load_config(path: str) -> FinconConfig:
    """
    Load configuration from a JSON file.

    Environment variables are used as fallbacks for sensitive data:
    - EDGAR_USER_AGENT: User agent for SEC EDGAR API
    - FINLIGHT_API_KEY: API key for Finlight.me
    - OPENAI_API_KEY: API key for OpenAI
    - OPENAI_BASE_URL: Base URL for OpenAI API (for custom endpoints)
    - LLM_MODEL: LLM model to use

    Args:
        path: Path to JSON configuration file

    Returns:
        FinconConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    with open(path, "r") as f:
        config_dict = json.load(f)

    return FinconConfig(**config_dict)


def create_default_config(path: str) -> None:
    """
    Create a default configuration file at the specified path.

    Args:
        path: Path where to save the default config
    """
    default_config = FinconConfig()

    with open(path, "w") as f:
        json.dump(default_config.model_dump(), f, indent=2)
