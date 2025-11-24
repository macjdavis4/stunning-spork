# FINCON: Multi-Agent Financial Decision System

A fully functional implementation of the FINCON system described in the paper "FINCON: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making".

## Features

- **Multi-Agent Architecture**: Specialized LLM-based analyst agents for news, fundamentals, technical analysis, earnings calls, and stock selection
- **Real Data Integration**:
  - Market data from **yfinance**
  - Fundamental data from **SEC EDGAR API**
  - News data from **finlight.me**
- **Risk Management**: CVaR-based risk control with adaptive position sizing
- **Cross-Episode Learning**: Conceptual verbal reinforcement to improve performance across episodes
- **Portfolio Support**: Both single-stock and multi-stock portfolio optimization
- **Complete Implementation**: No TODOs or placeholders - fully runnable end-to-end

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     FINCON System                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ News Analyst │  │ Fundamentals │  │ Data Analyst │      │
│  │              │  │   Analyst    │  │              │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           │                                │
│                    ┌──────▼───────┐                        │
│                    │   Manager    │                        │
│                    │    Agent     │                        │
│                    └──────┬───────┘                        │
│                           │                                │
│                    ┌──────▼───────┐                        │
│                    │ Risk Control │                        │
│                    │   (CVaR)     │                        │
│                    └──────────────┘                        │
│                                                            │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Episodic Memory (Conceptual Reinforcement)        │     │
│  └───────────────────────────────────────────────────┘     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10 or higher
- API keys:
  - OpenAI API key (or compatible LLM endpoint)
  - Finlight.me API key (optional, falls back to mock data)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### Generate Default Config

```bash
python -m fincon create-config --output config.json
```

### Edit Configuration

Edit `config.json` and set the following:

**Note on Episodes**: The system runs multiple episodes over the same time period to enable cross-episode learning. Each episode consists of `episode_days` trading days (default: 5 days), and the system runs `episodes` iterations (default: 4) over this same period. The agents learn and improve their decision-making across episodes through conceptual verbal reinforcement.

```json
{
  "symbols": ["AAPL"],
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "episodes": 4,
  "episode_days": 5,
  "initial_capital": 100000.0,
  "cvar_alpha": 0.05,
  "edgar": {
    "user_agent": "YourName your.email@example.com"
  },
  "finlight": {
    "api_key": "your-finlight-api-key"
  },
  "llm": {
    "model": "gpt-4",
    "api_key": "your-openai-api-key"
  }
}
```

### Environment Variables

You can also use environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export FINLIGHT_API_KEY="your-finlight-key"
export EDGAR_USER_AGENT="YourName your.email@example.com"
```

## Usage

### Command-Line Interface

#### Single-Stock Trading

```bash
python -m fincon single-stock AAPL --config config.json
```

Optional arguments:
- `--start-date YYYY-MM-DD`: Override start date
- `--end-date YYYY-MM-DD`: Override end date
- `--episodes N`: Override number of episodes
- `--verbose`: Enable detailed output

#### Portfolio Trading

```bash
python -m fincon portfolio AAPL MSFT GOOGL --config config.json
```

### Python API

#### Single-Stock Example

```python
from fincon.config import FinconConfig
from fincon.training import create_trainer_from_config

# Create configuration
config = FinconConfig(
    symbols=["AAPL"],
    start_date="2023-01-01",
    end_date="2023-06-30",
    episodes=4,
    episode_days=5
)

# Create and run trainer
trainer = create_trainer_from_config(config)
trainer.train()
trainer.print_summary()
```

#### Portfolio Example

```python
from fincon.config import FinconConfig
from fincon.training import create_trainer_from_config

# Create configuration
config = FinconConfig(
    symbols=["AAPL", "MSFT", "GOOGL"],
    start_date="2023-01-01",
    end_date="2023-06-30",
    episodes=4,
    episode_days=5,
    enable_stock_selection_analyst=True
)

# Create and run trainer
trainer = create_trainer_from_config(config)
trainer.train()
trainer.print_summary()
```

### Example Scripts

Run the included example scripts:

```bash
# Single-stock demo
python examples/single_stock_demo.py

# Portfolio demo
python examples/portfolio_demo.py
```

## System Components

### Data Layer (`fincon/data.py`)

- **MarketDataClient**: Fetches OHLCV data from yfinance
- **EdgarClient**: Retrieves fundamental data from SEC EDGAR API
- **FinlightNewsClient**: Fetches news articles from finlight.me

### Environment (`fincon/env.py`)

- **MarketEnvironment**: POMDP environment for trading simulation
- Tracks portfolio state, computes rewards, manages positions

### Agents

#### Analyst Agents (`fincon/agents/analyst.py`)

1. **NewsAnalystAgent**: Analyzes news sentiment and identifies catalysts
2. **FundamentalsAnalystAgent**: Evaluates financial metrics and company health
3. **DataAnalystAgent**: Performs technical/quantitative analysis
4. **AudioECCAnalystAgent**: Processes earnings call transcripts
5. **StockSelectionAnalystAgent**: Selects stocks for portfolio construction

#### Manager Agent (`fincon/agents/manager.py`)

- Synthesizes analyst outputs using weighted beliefs
- Makes final trading decisions (BUY/SELL/HOLD)
- Performs portfolio optimization using Markowitz mean-variance

#### Risk Control (`fincon/agents/risk_control.py`)

- **Within-Episode**: Monitors CVaR and triggers risk aversion
- **Cross-Episode**: Generates verbal reinforcement to update beliefs

### Memory Systems (`fincon/memory.py`)

- **WorkingMemory**: Short-term context for current decision
- **ProceduralMemory**: Step-by-step action history with recency weighting
- **EpisodicMemory**: Episode summaries and conceptual beliefs

### Training (`fincon/training.py`)

- **FinconTrainer**: Orchestrates multi-episode training
- Manages agent execution and memory updates
- Computes performance metrics

### Evaluation (`fincon/evaluation.py`)

Performance metrics:
- Cumulative returns
- Sharpe ratio
- Maximum drawdown
- Sortino ratio
- Calmar ratio
- Win rate
- Profit factor

## Output

The system provides comprehensive performance metrics:

```
============================================================
PERFORMANCE METRICS SUMMARY
============================================================

Returns:
  Cumulative Return:         15.23%
  Mean Return:              0.0012%
  Volatility:               0.0234%

Risk-Adjusted:
  Sharpe Ratio:               1.85
  Sortino Ratio:              2.34
  Calmar Ratio:               3.12

Risk:
  Max Drawdown:              -5.67%

Trading:
  Total Trades:                120
  Win Rate:                  58.33%
  Profit Factor:              1.45

============================================================
```

## Advanced Configuration

### Custom LLM Backend

```python
from fincon.llm_backend import ClaudeLLMBackend

# Use Claude instead of OpenAI
llm = ClaudeLLMBackend(
    model="claude-3-opus-20240229",
    api_key="your-anthropic-key"
)
```

### Adjust Risk Parameters

```python
config = FinconConfig(
    cvar_alpha=0.05,  # CVaR confidence level
    portfolio=PortfolioConfig(
        max_position_size=0.25,  # Max 25% per position
        min_position_size=0.05,  # Min 5% per position
    )
)
```

### Enable/Disable Analysts

```python
config = FinconConfig(
    enable_news_analyst=True,
    enable_fundamentals_analyst=True,
    enable_data_analyst=True,
    enable_ecc_analyst=False,  # Disable if no transcripts
    enable_stock_selection_analyst=True  # For portfolio mode
)
```

## Project Structure

```
fincon/
├── __init__.py
├── __main__.py              # CLI entry point
├── config.py                # Configuration management
├── data.py                  # Data clients (yfinance, EDGAR, finlight)
├── env.py                   # Trading environment (POMDP)
├── llm_backend.py           # LLM abstraction layer
├── memory.py                # Memory systems
├── training.py              # Training orchestration
├── evaluation.py            # Performance metrics
├── cli.py                   # Command-line interface
└── agents/
    ├── __init__.py
    ├── base.py              # Base agent class
    ├── analyst.py           # Analyst agents
    ├── manager.py           # Manager agent
    └── risk_control.py      # Risk control & reinforcement

examples/
├── single_stock_demo.py     # Single-stock example
└── portfolio_demo.py        # Portfolio example
```

## Testing Without API Keys

For testing without API keys, the system includes fallbacks:

- **Finlight News**: Falls back to mock news data
- **LLM**: Set `llm.model = "mock"` to use mock LLM responses

```python
config = FinconConfig(
    llm=LLMConfig(model="mock"),  # Use mock LLM
    finlight=FinlightConfig(api_key="")  # Will use mock news
)
```

## Troubleshooting

### No CIK mapping for ticker

Add the ticker to `TICKER_TO_CIK` in `fincon/data.py`:

```python
TICKER_TO_CIK = {
    "YOUR_SYMBOL": "0000123456",  # Add your CIK
    ...
}
```

Find CIKs at: https://www.sec.gov/edgar/searchedgar/companysearch

### API rate limits

The system includes retry logic and delays for SEC EDGAR API compliance. Adjust delays in `fincon/data.py` if needed.

## Citation

If you use this implementation, please cite the original FINCON paper:

```
"FINCON: A Synthesized LLM Multi-Agent System with Conceptual Verbal
Reinforcement for Enhanced Financial Decision Making"
```

## License

MIT License - See LICENSE file for details.

## Disclaimer

This software is for research and educational purposes only. Not financial advice. Use at your own risk.
