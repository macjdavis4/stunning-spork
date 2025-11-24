"""
Analyst agents for FINCON system.

Implements specialized analysts:
- NewsAnalystAgent: Analyzes news sentiment and catalysts
- FundamentalsAnalystAgent: Evaluates fundamental metrics
- DataAnalystAgent: Analyzes technical/quantitative data
- AudioECCAnalystAgent: Processes earnings call transcripts
- StockSelectionAnalystAgent: Selects stocks for portfolio
"""

from typing import Any
import numpy as np
import pandas as pd

from fincon.agents.base import BaseAgent
from fincon.llm_backend import LLMBackend
from fincon.memory import ProceduralMemory
from fincon.data import FinlightNewsClient, EdgarClient


class NewsAnalystAgent(BaseAgent):
    """
    News analyst that processes news articles and sentiment.
    """

    def __init__(
        self,
        llm: LLMBackend,
        news_client: FinlightNewsClient,
        memory: ProceduralMemory | None = None
    ):
        """
        Initialize news analyst.

        Args:
            llm: LLM backend
            news_client: Finlight news client
            memory: Optional procedural memory
        """
        super().__init__(
            name="NewsAnalyst",
            role_description="Analyzes news sentiment and identifies catalysts",
            llm=llm,
            memory=memory
        )
        self.news_client = news_client

    def build_system_prompt(self) -> str:
        """Build system prompt for news analyst."""
        return """You are an expert financial news analyst specializing in sentiment analysis and catalyst identification.

Your task is to analyze news articles about a company and provide:
1. Overall sentiment score from -1 (very negative) to +1 (very positive)
2. Key risks mentioned in the news
3. Key opportunities identified
4. Potential catalysts (events that could move the stock)
5. Your confidence in this analysis (0 to 1)

Output your analysis as a JSON object with this structure:
{
  "sentiment_score": <float between -1 and 1>,
  "key_risks": [<list of risk strings>],
  "key_opportunities": [<list of opportunity strings>],
  "catalysts": [<list of catalyst strings>],
  "confidence": <float between 0 and 1>
}

Be objective and evidence-based. Focus on material information that could impact stock price."""

    def build_messages(self, symbol: str, news_articles: list[dict[str, Any]], **kwargs) -> list[dict[str, str]]:
        """
        Build messages for news analysis.

        Args:
            symbol: Stock symbol
            news_articles: List of news article dictionaries
            **kwargs: Additional context

        Returns:
            Message list for LLM
        """
        # Format news articles
        news_text = f"News articles for {symbol}:\n\n"

        for i, article in enumerate(news_articles[:20], 1):
            news_text += f"{i}. {article.get('title', 'No title')}\n"
            news_text += f"   Source: {article.get('source', 'Unknown')}\n"
            news_text += f"   Date: {article.get('published_at', 'Unknown')}\n"

            summary = article.get('summary', '')
            if summary:
                news_text += f"   Summary: {summary}\n"

            news_text += "\n"

        user_message = f"Analyze the following news articles for {symbol} and provide your assessment:\n\n{news_text}"

        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_message}
        ]

    def analyze(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        """
        Analyze news for a symbol.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of news articles to fetch

        Returns:
            Analysis results dictionary
        """
        # Fetch news
        news_articles = self.news_client.fetch_news(symbol, limit=limit)

        if not news_articles:
            return {
                "sentiment_score": 0.0,
                "key_risks": ["No news data available"],
                "key_opportunities": [],
                "catalysts": [],
                "confidence": 0.0
            }

        # Get LLM analysis
        response = self.call_llm(symbol=symbol, news_articles=news_articles)

        # Parse response
        try:
            analysis = self.parse_json_response(response)
        except ValueError as e:
            print(f"News analyst parsing error: {e}")
            analysis = {
                "sentiment_score": 0.0,
                "key_risks": ["Analysis parsing failed"],
                "key_opportunities": [],
                "catalysts": [],
                "confidence": 0.0
            }

        # Log to memory
        self.log_to_memory({
            "symbol": symbol,
            "analysis": analysis,
            "num_articles": len(news_articles)
        }, tags=["news_analysis"])

        return analysis


class FundamentalsAnalystAgent(BaseAgent):
    """
    Fundamentals analyst that evaluates company financials.
    """

    def __init__(
        self,
        llm: LLMBackend,
        edgar_client: EdgarClient,
        memory: ProceduralMemory | None = None
    ):
        """
        Initialize fundamentals analyst.

        Args:
            llm: LLM backend
            edgar_client: SEC EDGAR client
            memory: Optional procedural memory
        """
        super().__init__(
            name="FundamentalsAnalyst",
            role_description="Evaluates fundamental financial metrics",
            llm=llm,
            memory=memory
        )
        self.edgar_client = edgar_client

    def build_system_prompt(self) -> str:
        """Build system prompt for fundamentals analyst."""
        return """You are an expert fundamental analyst specializing in company financial analysis.

Your task is to analyze fundamental financial metrics and provide:
1. Fundamental view: "strong_long", "weak_long", "neutral", or "short_bias"
2. Detailed rationale for your view
3. Fundamental score from -1 (very bearish) to +1 (very bullish)

Consider:
- Revenue growth trends
- Profitability metrics
- Balance sheet health
- Valuation relative to fundamentals

Output your analysis as a JSON object:
{
  "fundamental_view": <"strong_long"|"weak_long"|"neutral"|"short_bias">,
  "rationale": <detailed explanation>,
  "fundamental_score": <float between -1 and 1>
}

Be thorough but concise. Focus on the most material fundamental factors."""

    def build_messages(self, symbol: str, fundamentals: dict[str, Any], **kwargs) -> list[dict[str, str]]:
        """
        Build messages for fundamental analysis.

        Args:
            symbol: Stock symbol
            fundamentals: Fundamental data dictionary
            **kwargs: Additional context

        Returns:
            Message list for LLM
        """
        # Format fundamentals
        fundamentals_text = f"Fundamental data for {symbol}:\n\n"

        for key, value in fundamentals.items():
            if isinstance(value, dict):
                latest = value.get('latest', 'N/A')
                date = value.get('date', 'N/A')
                fundamentals_text += f"{key}: {latest} (as of {date})\n"

                if 'history' in value and value['history']:
                    history_values = [h['value'] for h in value['history'][:3]]
                    fundamentals_text += f"  Recent history: {history_values}\n"
            else:
                fundamentals_text += f"{key}: {value}\n"

        user_message = f"Analyze the fundamental metrics for {symbol}:\n\n{fundamentals_text}"

        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_message}
        ]

    def analyze(self, symbol: str) -> dict[str, Any]:
        """
        Analyze fundamentals for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Analysis results dictionary
        """
        # Get CIK for symbol
        cik = self.edgar_client.get_cik_from_ticker(symbol)

        if not cik:
            return {
                "fundamental_view": "neutral",
                "rationale": f"No CIK mapping available for {symbol}",
                "fundamental_score": 0.0
            }

        # Fetch fundamentals
        fundamentals = self.edgar_client.get_key_fundamentals(cik)

        if not fundamentals:
            return {
                "fundamental_view": "neutral",
                "rationale": f"No fundamental data available for {symbol}",
                "fundamental_score": 0.0
            }

        # Get LLM analysis
        response = self.call_llm(symbol=symbol, fundamentals=fundamentals)

        # Parse response
        try:
            analysis = self.parse_json_response(response)
        except ValueError as e:
            print(f"Fundamentals analyst parsing error: {e}")
            analysis = {
                "fundamental_view": "neutral",
                "rationale": "Analysis parsing failed",
                "fundamental_score": 0.0
            }

        # Log to memory
        self.log_to_memory({
            "symbol": symbol,
            "analysis": analysis,
            "fundamentals_summary": {k: v.get('latest') if isinstance(v, dict) else v for k, v in fundamentals.items()}
        }, tags=["fundamentals_analysis"])

        return analysis


class DataAnalystAgent(BaseAgent):
    """
    Data analyst that performs technical/quantitative analysis.
    """

    def __init__(
        self,
        llm: LLMBackend,
        memory: ProceduralMemory | None = None
    ):
        """
        Initialize data analyst.

        Args:
            llm: LLM backend
            memory: Optional procedural memory
        """
        super().__init__(
            name="DataAnalyst",
            role_description="Performs technical and quantitative analysis",
            llm=llm,
            memory=memory
        )

    def build_system_prompt(self) -> str:
        """Build system prompt for data analyst."""
        return """You are an expert quantitative analyst specializing in technical analysis and market data.

Your task is to analyze price and volume data to provide:
1. Momentum score from -1 (strong bearish momentum) to +1 (strong bullish momentum)
2. Volatility regime: "low", "medium", or "high"
3. Risk assessment narrative
4. Overall data signal score from -1 to +1

Consider:
- Price trends and momentum
- Volatility patterns
- Volume analysis
- Technical indicators (moving averages, etc.)

Output your analysis as a JSON object:
{
  "momentum_score": <float between -1 and 1>,
  "volatility_regime": <"low"|"medium"|"high">,
  "risk_assessment": <brief assessment>,
  "data_signal_score": <float between -1 and 1>
}

Be precise and data-driven."""

    def build_messages(self, symbol: str, market_data: dict[str, Any], **kwargs) -> list[dict[str, str]]:
        """
        Build messages for data analysis.

        Args:
            symbol: Stock symbol
            market_data: Market data dictionary
            **kwargs: Additional context

        Returns:
            Message list for LLM
        """
        data_text = f"Market data analysis for {symbol}:\n\n"

        # Extract key metrics
        data_text += f"Current price: ${market_data.get('close', 'N/A')}\n"
        data_text += f"Price change: {market_data.get('price_change', 0):.2%}\n"
        data_text += f"Volatility (std of returns): {market_data.get('volatility', 0):.4f}\n"
        data_text += f"Mean return: {market_data.get('mean_return', 0):.4f}\n"
        data_text += f"5-day SMA: ${market_data.get('sma_5', 'N/A')}\n"
        data_text += f"20-day SMA: ${market_data.get('sma_20', 'N/A')}\n"
        data_text += f"Volume: {market_data.get('volume', 'N/A'):,.0f}\n"

        returns = market_data.get('returns', [])
        if returns:
            data_text += f"\nRecent returns (last {len(returns)} periods): {[f'{r:.4f}' for r in returns[-10:]]}\n"

        user_message = f"Analyze the technical/quantitative data for {symbol}:\n\n{data_text}"

        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_message}
        ]

    def analyze(self, symbol: str, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze market data for a symbol.

        Args:
            symbol: Stock ticker symbol
            observation: Environment observation with market data

        Returns:
            Analysis results dictionary
        """
        if symbol not in observation.get('symbols', {}):
            return {
                "momentum_score": 0.0,
                "volatility_regime": "medium",
                "risk_assessment": f"No data available for {symbol}",
                "data_signal_score": 0.0
            }

        market_data = observation['symbols'][symbol]

        # Get LLM analysis
        response = self.call_llm(symbol=symbol, market_data=market_data)

        # Parse response
        try:
            analysis = self.parse_json_response(response)
        except ValueError as e:
            print(f"Data analyst parsing error: {e}")
            analysis = {
                "momentum_score": 0.0,
                "volatility_regime": "medium",
                "risk_assessment": "Analysis parsing failed",
                "data_signal_score": 0.0
            }

        # Log to memory
        self.log_to_memory({
            "symbol": symbol,
            "analysis": analysis,
            "market_data_summary": {
                "price": market_data.get('close'),
                "volatility": market_data.get('volatility')
            }
        }, tags=["data_analysis"])

        return analysis


class AudioECCAnalystAgent(BaseAgent):
    """
    Earnings call analyst that processes transcripts.
    """

    def __init__(
        self,
        llm: LLMBackend,
        memory: ProceduralMemory | None = None
    ):
        """
        Initialize earnings call analyst.

        Args:
            llm: LLM backend
            memory: Optional procedural memory
        """
        super().__init__(
            name="AudioECCAnalyst",
            role_description="Analyzes earnings call transcripts",
            llm=llm,
            memory=memory
        )

    def build_system_prompt(self) -> str:
        """Build system prompt for earnings call analyst."""
        return """You are an expert analyst specializing in earnings call transcript analysis.

Your task is to analyze management commentary and provide:
1. ECC score from -1 (very negative tone) to +1 (very positive tone)
2. Guidance bias: "bullish", "neutral", or "bearish"
3. Key quotes that are most impactful

Consider:
- Management tone and confidence
- Forward guidance and outlook
- Key strategic initiatives
- Response to analyst questions

Output your analysis as a JSON object:
{
  "ecc_score": <float between -1 and 1>,
  "guidance_bias": <"bullish"|"neutral"|"bearish">,
  "key_quotes": [<list of important quote strings>]
}

Focus on forward-looking statements and strategic direction."""

    def build_messages(self, symbol: str, transcript: str, **kwargs) -> list[dict[str, str]]:
        """
        Build messages for earnings call analysis.

        Args:
            symbol: Stock symbol
            transcript: Earnings call transcript
            **kwargs: Additional context

        Returns:
            Message list for LLM
        """
        user_message = f"Analyze the following earnings call transcript for {symbol}:\n\n{transcript}"

        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_message}
        ]

    def analyze(self, symbol: str, transcript: str = "") -> dict[str, Any]:
        """
        Analyze earnings call transcript.

        Args:
            symbol: Stock ticker symbol
            transcript: Earnings call transcript text

        Returns:
            Analysis results dictionary
        """
        if not transcript:
            return {
                "ecc_score": 0.0,
                "guidance_bias": "neutral",
                "key_quotes": ["No transcript available"]
            }

        # Get LLM analysis
        response = self.call_llm(symbol=symbol, transcript=transcript)

        # Parse response
        try:
            analysis = self.parse_json_response(response)
        except ValueError as e:
            print(f"ECC analyst parsing error: {e}")
            analysis = {
                "ecc_score": 0.0,
                "guidance_bias": "neutral",
                "key_quotes": ["Analysis parsing failed"]
            }

        # Log to memory
        self.log_to_memory({
            "symbol": symbol,
            "analysis": analysis
        }, tags=["ecc_analysis"])

        return analysis


class StockSelectionAnalystAgent(BaseAgent):
    """
    Stock selection analyst for portfolio construction.
    """

    def __init__(
        self,
        llm: LLMBackend,
        memory: ProceduralMemory | None = None
    ):
        """
        Initialize stock selection analyst.

        Args:
            llm: LLM backend
            memory: Optional procedural memory
        """
        super().__init__(
            name="StockSelectionAnalyst",
            role_description="Selects stocks for portfolio construction",
            llm=llm,
            memory=memory
        )

    def build_system_prompt(self) -> str:
        """Build system prompt for stock selection analyst."""
        return """You are an expert portfolio analyst specializing in stock selection.

Your task is to analyze multiple stocks and select the best candidates for a portfolio.

Provide:
1. List of selected symbols (subset of candidates)
2. Brief rationale for each selection

Output your analysis as a JSON object:
{
  "selected_symbols": [<list of ticker strings>],
  "rationale": <explanation of selection criteria>
}

Consider diversification, risk-return profile, and current market conditions."""

    def build_messages(self, symbols: list[str], market_data: dict[str, Any], **kwargs) -> list[dict[str, str]]:
        """
        Build messages for stock selection.

        Args:
            symbols: List of candidate symbols
            market_data: Market data for all symbols
            **kwargs: Additional context

        Returns:
            Message list for LLM
        """
        data_text = "Stock candidates:\n\n"

        for symbol in symbols:
            if symbol in market_data:
                data = market_data[symbol]
                data_text += f"{symbol}:\n"
                data_text += f"  Price: ${data.get('close', 'N/A')}\n"
                data_text += f"  Volatility: {data.get('volatility', 0):.4f}\n"
                data_text += f"  Mean return: {data.get('mean_return', 0):.4f}\n"
                data_text += "\n"

        user_message = f"Analyze and select stocks for portfolio construction:\n\n{data_text}"

        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": user_message}
        ]

    def analyze(
        self,
        symbols: list[str],
        observation: dict[str, Any],
        max_selections: int = 5
    ) -> dict[str, Any]:
        """
        Perform stock selection analysis.

        Args:
            symbols: List of candidate symbols
            observation: Environment observation
            max_selections: Maximum number of stocks to select

        Returns:
            Analysis results with selected symbols and metrics
        """
        market_data = observation.get('symbols', {})

        # Calculate correlation and covariance matrices
        returns_data = {}
        for symbol in symbols:
            if symbol in market_data:
                returns = market_data[symbol].get('returns', [])
                if returns:
                    returns_data[symbol] = returns

        if not returns_data:
            return {
                "selected_symbols": symbols[:max_selections],
                "expected_return_vector": {},
                "covariance_matrix": {},
                "rationale": "Insufficient data for correlation analysis"
            }

        # Compute simple metrics
        expected_returns = {}
        volatilities = {}

        for symbol, returns in returns_data.items():
            expected_returns[symbol] = float(np.mean(returns))
            volatilities[symbol] = float(np.std(returns))

        # Simple selection: top performers by Sharpe-like ratio
        sharpe_scores = {
            symbol: expected_returns[symbol] / (volatilities[symbol] + 1e-8)
            for symbol in returns_data.keys()
        }

        selected = sorted(sharpe_scores.keys(), key=lambda s: sharpe_scores[s], reverse=True)[:max_selections]

        # Log to memory
        self.log_to_memory({
            "selected_symbols": selected,
            "sharpe_scores": {s: sharpe_scores[s] for s in selected}
        }, tags=["stock_selection"])

        return {
            "selected_symbols": selected,
            "expected_return_vector": {s: expected_returns[s] for s in selected},
            "covariance_matrix": {},  # Simplified for now
            "rationale": f"Selected top {len(selected)} stocks by risk-adjusted returns"
        }
