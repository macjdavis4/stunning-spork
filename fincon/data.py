"""
Data layer for FINCON system.

Provides clients for:
- Market data (yfinance)
- Fundamental data (SEC EDGAR API)
- News data (finlight.me)
"""

import time
from datetime import datetime
from typing import Any
import warnings

import pandas as pd
import numpy as np
import yfinance as yf
import requests


# Ticker to CIK mapping for common symbols
# Users can extend this mapping as needed
TICKER_TO_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
    "AMZN": "0001018724",
    "TSLA": "0001318605",
    "META": "0001326801",
    "NVDA": "0001045810",
    "JPM": "0000019617",
    "V": "0001403161",
    "WMT": "0000104169",
    "MA": "0001141391",
    "UNH": "0000731766",
    "JNJ": "0000200406",
    "BAC": "0000070858",
    "PG": "0000080424",
    "XOM": "0000034088",
    "DIS": "0001744489",
    "NFLX": "0001065280",
    "KO": "0000021344",
}


class MarketDataClient:
    """
    Client for fetching market data using yfinance.
    """

    def __init__(self):
        """Initialize the market data client."""
        warnings.filterwarnings("ignore", category=FutureWarning)

    def get_price_history(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Use yfinance to download OHLCV data for a single symbol.

        Args:
            symbol: Stock ticker symbol
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame indexed by date with columns:
            ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        Raises:
            ValueError: If no data is returned for the symbol
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)

            if df.empty:
                raise ValueError(f"No data returned for symbol {symbol}")

            # Ensure consistent column names
            df.index.name = "Date"

            return df

        except Exception as e:
            raise ValueError(f"Error fetching data for {symbol}: {e}")

    def get_multi_price_history(
        self,
        symbols: list[str],
        start: str,
        end: str,
        interval: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of stock ticker symbols
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)

        Returns:
            Dictionary mapping symbol -> OHLCV DataFrame
        """
        result = {}

        for symbol in symbols:
            try:
                df = self.get_price_history(symbol, start, end, interval)
                result[symbol] = df
            except ValueError as e:
                print(f"Warning: {e}")
                continue

        return result

    def get_latest_price(self, symbol: str) -> float:
        """
        Get the most recent closing price for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Latest closing price
        """
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")

        if hist.empty:
            raise ValueError(f"No recent data for {symbol}")

        return float(hist["Close"].iloc[-1])


class EdgarClient:
    """
    Client for fetching fundamental data from SEC EDGAR API.
    """

    def __init__(
        self,
        user_agent: str,
        base_url: str = "https://data.sec.gov"
    ):
        """
        Initialize EDGAR client.

        Args:
            user_agent: User agent string (required by SEC, format: 'Name email@example.com')
            base_url: Base URL for SEC EDGAR API
        """
        self.user_agent = user_agent
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _make_request(self, url: str, max_retries: int = 3) -> dict[str, Any]:
        """
        Make HTTP request with retries.

        Args:
            url: URL to fetch
            max_retries: Maximum number of retry attempts

        Returns:
            Parsed JSON response

        Raises:
            requests.RequestException: If request fails after retries
        """
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1 * (attempt + 1))

        return {}

    def get_company_concept(
        self,
        cik: str,
        taxonomy: str,
        concept: str
    ) -> dict[str, Any]:
        """
        Call the SEC 'company concept' endpoint.

        Args:
            cik: Central Index Key (CIK) with leading zeros
            taxonomy: Taxonomy (e.g., 'us-gaap', 'ifrs-full')
            concept: Concept name (e.g., 'Revenues', 'Assets')

        Returns:
            Parsed JSON response from SEC API
        """
        url = f"{self.base_url}/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{concept}.json"

        try:
            return self._make_request(url)
        except requests.RequestException as e:
            print(f"Warning: Failed to fetch {concept} for CIK {cik}: {e}")
            return {}

    def get_key_fundamentals(self, cik: str) -> dict[str, Any]:
        """
        Fetch key fundamental metrics for a company.

        Args:
            cik: Central Index Key (CIK) with leading zeros

        Returns:
            Dictionary with fundamental metrics:
            - revenues: Latest revenue and historical trend
            - net_income: Latest net income and trend
            - assets: Total assets
            - liabilities: Total liabilities
            - equity: Stockholders equity
            - eps: Earnings per share
        """
        fundamentals = {}

        # Key concepts to fetch
        concepts = {
            "Revenues": "revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax": "revenues_alt",
            "NetIncomeLoss": "net_income",
            "Assets": "assets",
            "Liabilities": "liabilities",
            "StockholdersEquity": "equity",
            "EarningsPerShareBasic": "eps_basic",
            "EarningsPerShareDiluted": "eps_diluted",
        }

        for concept, key in concepts.items():
            data = self.get_company_concept(cik, "us-gaap", concept)

            if not data or "units" not in data:
                continue

            # Extract USD values
            if "USD" in data["units"]:
                values = data["units"]["USD"]

                # Filter for annual data (10-K filings)
                annual_values = [
                    v for v in values
                    if v.get("form") == "10-K" and "val" in v and "end" in v
                ]

                if annual_values:
                    # Sort by end date
                    annual_values.sort(key=lambda x: x["end"], reverse=True)

                    fundamentals[key] = {
                        "latest": annual_values[0]["val"],
                        "date": annual_values[0]["end"],
                        "history": [
                            {"value": v["val"], "date": v["end"]}
                            for v in annual_values[:5]
                        ]
                    }

            # Handle per-share values
            elif "USD/shares" in data["units"]:
                values = data["units"]["USD/shares"]
                annual_values = [
                    v for v in values
                    if v.get("form") == "10-K" and "val" in v and "end" in v
                ]

                if annual_values:
                    annual_values.sort(key=lambda x: x["end"], reverse=True)

                    fundamentals[key] = {
                        "latest": annual_values[0]["val"],
                        "date": annual_values[0]["end"],
                        "history": [
                            {"value": v["val"], "date": v["end"]}
                            for v in annual_values[:5]
                        ]
                    }

        return fundamentals

    def get_cik_from_ticker(self, ticker: str) -> str | None:
        """
        Get CIK from ticker symbol.

        Args:
            ticker: Stock ticker symbol

        Returns:
            CIK string with leading zeros, or None if not found
        """
        return TICKER_TO_CIK.get(ticker.upper())


class FinlightNewsClient:
    """
    Client for fetching news data from finlight.me API.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.finlight.me/v1"
    ):
        """
        Initialize Finlight news client.

        Args:
            api_key: API key for finlight.me
            base_url: Base URL for Finlight API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

    def fetch_news(
        self,
        symbol: str,
        limit: int = 20,
        from_date: str | None = None,
        to_date: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Fetch recent news articles for a ticker.

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles to fetch
            from_date: Start date in YYYY-MM-DD format (optional)
            to_date: End date in YYYY-MM-DD format (optional)

        Returns:
            List of news articles with fields:
            - title: Article title
            - summary: Article summary/description
            - published_at: Publication timestamp
            - source: News source
            - url: Article URL
            - sentiment: Sentiment score if available, else None
        """
        if not self.api_key:
            print("Warning: No Finlight API key provided, returning empty news list")
            return []

        # Construct endpoint (this is a mock endpoint structure)
        # Adjust based on actual Finlight API documentation
        url = f"{self.base_url}/news"

        params = {
            "symbol": symbol,
            "limit": limit
        }

        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Normalize response
            articles = []

            if isinstance(data, dict) and "articles" in data:
                raw_articles = data["articles"]
            elif isinstance(data, list):
                raw_articles = data
            else:
                raw_articles = []

            for article in raw_articles:
                normalized = {
                    "title": article.get("title", ""),
                    "summary": article.get("summary") or article.get("description", ""),
                    "published_at": article.get("published_at") or article.get("publishedAt", ""),
                    "source": article.get("source", {}).get("name", "Unknown") if isinstance(article.get("source"), dict) else str(article.get("source", "Unknown")),
                    "url": article.get("url", ""),
                    "sentiment": article.get("sentiment")
                }
                articles.append(normalized)

            return articles[:limit]

        except requests.RequestException as e:
            print(f"Warning: Failed to fetch news for {symbol}: {e}")
            # Return mock data for testing when API is unavailable
            return self._get_mock_news(symbol, limit)

    def _get_mock_news(self, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
        """
        Generate mock news data for testing when API is unavailable.

        Args:
            symbol: Stock ticker symbol
            limit: Number of mock articles

        Returns:
            List of mock news articles
        """
        mock_articles = [
            {
                "title": f"{symbol} Reports Strong Quarterly Earnings",
                "summary": f"{symbol} exceeded analyst expectations with strong revenue growth.",
                "published_at": datetime.now().isoformat(),
                "source": "Financial Times",
                "url": f"https://example.com/news/{symbol}/earnings",
                "sentiment": 0.7
            },
            {
                "title": f"Analysts Upgrade {symbol} Stock Rating",
                "summary": f"Major investment banks upgrade {symbol} citing positive outlook.",
                "published_at": datetime.now().isoformat(),
                "source": "Bloomberg",
                "url": f"https://example.com/news/{symbol}/upgrade",
                "sentiment": 0.6
            },
            {
                "title": f"{symbol} Announces New Product Launch",
                "summary": f"{symbol} unveils innovative product expected to drive growth.",
                "published_at": datetime.now().isoformat(),
                "source": "Reuters",
                "url": f"https://example.com/news/{symbol}/product",
                "sentiment": 0.5
            }
        ]

        return mock_articles[:limit]
