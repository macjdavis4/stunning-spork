"""
LLM backend abstraction for FINCON system.

Provides a protocol for LLM backends and implementations for:
- OpenAI-compatible APIs (OpenAI, Azure OpenAI, local models via vLLM, etc.)
- Anthropic Claude (via Messages API)
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Protocol

import requests


class LLMBackend(Protocol):
    """Protocol for LLM backends."""

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> str:
        """
        Send chat messages to LLM and get response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Response text from LLM
        """
        ...


class OpenAILLMBackend:
    """
    LLM backend for OpenAI-compatible APIs.

    Compatible with:
    - OpenAI API
    - Azure OpenAI
    - Local models via vLLM, llama.cpp, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_temperature: float = 0.3,
        default_max_tokens: int = 2048
    ):
        """
        Initialize OpenAI-compatible LLM backend.

        Args:
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: API key
            base_url: Base URL for API endpoint
            default_temperature: Default sampling temperature
            default_max_tokens: Default max tokens
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> str:
        """
        Send chat messages to OpenAI-compatible API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (uses default if None)
            max_tokens: Maximum tokens (uses default if None)

        Returns:
            Response text from LLM

        Raises:
            requests.RequestException: If API request fails
            ValueError: If response format is unexpected
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            print(f"Calling LLM API with model {self.model}...")
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "choices" not in data or len(data["choices"]) == 0:
                raise ValueError("No choices in API response")

            print(f"LLM API call successful")
            return data["choices"][0]["message"]["content"]

        except requests.RequestException as e:
            print(f"LLM API call failed: {e}")
            raise requests.RequestException(f"LLM API request failed: {e}")


class ClaudeLLMBackend:
    """
    LLM backend for Anthropic Claude via Messages API.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        default_temperature: float = 0.3,
        default_max_tokens: int = 2048
    ):
        """
        Initialize Claude LLM backend.

        Args:
            model: Model name (e.g., 'claude-3-opus-20240229')
            api_key: Anthropic API key
            base_url: Base URL for API endpoint
            default_temperature: Default sampling temperature
            default_max_tokens: Default max tokens
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        })

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> str:
        """
        Send chat messages to Claude API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (uses default if None)
            max_tokens: Maximum tokens (uses default if None)

        Returns:
            Response text from Claude

        Raises:
            requests.RequestException: If API request fails
            ValueError: If response format is unexpected
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        url = f"{self.base_url}/messages"

        # Extract system message if present
        system_message = None
        filtered_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                filtered_messages.append(msg)

        payload = {
            "model": self.model,
            "messages": filtered_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if system_message:
            payload["system"] = system_message

        try:
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "content" not in data or len(data["content"]) == 0:
                raise ValueError("No content in API response")

            return data["content"][0]["text"]

        except requests.RequestException as e:
            raise requests.RequestException(f"Claude API request failed: {e}")


class MockLLMBackend:
    """
    Mock LLM backend for testing without API calls.
    Returns structured responses based on message content.
    """

    def __init__(self):
        """Initialize mock LLM backend."""
        pass

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> str:
        """
        Return mock responses based on message content.

        Args:
            messages: List of message dicts
            temperature: Ignored
            max_tokens: Ignored

        Returns:
            Mock response string
        """
        # Extract last user message
        last_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_message = msg["content"].lower()
                break

        # Return appropriate mock response based on content
        if "news" in last_message or "sentiment" in last_message:
            return json.dumps({
                "sentiment_score": 0.6,
                "key_risks": ["Market volatility", "Regulatory changes"],
                "key_opportunities": ["Product launch", "Market expansion"],
                "catalysts": ["Earnings report", "Partnership announcement"],
                "confidence": 0.75
            })

        elif "fundamental" in last_message or "earnings" in last_message:
            return json.dumps({
                "fundamental_view": "strong_long",
                "rationale": "Strong revenue growth and improving margins",
                "fundamental_score": 0.7
            })

        elif "technical" in last_message or "data" in last_message:
            return json.dumps({
                "momentum_score": 0.5,
                "volatility_regime": "medium",
                "risk_assessment": "Moderate volatility with positive trend",
                "data_signal_score": 0.55
            })

        elif "decision" in last_message or "action" in last_message:
            return json.dumps({
                "action": "BUY",
                "position_size": 0.15,
                "reasoning": "Positive sentiment, strong fundamentals, and favorable technicals"
            })

        else:
            return json.dumps({
                "response": "Analysis complete",
                "confidence": 0.7
            })


def create_llm_backend(config: dict[str, Any]) -> LLMBackend:
    """
    Factory function to create LLM backend from configuration.

    Args:
        config: Configuration dict with keys:
            - model: Model name
            - api_key: API key
            - base_url: Base URL (optional)
            - temperature: Default temperature (optional)
            - max_tokens: Default max tokens (optional)

    Returns:
        LLM backend instance

    Raises:
        ValueError: If configuration is invalid
    """
    model = config.get("model", "")
    api_key = config.get("api_key", "")

    # Detect backend type from model name or base URL
    if "claude" in model.lower():
        return ClaudeLLMBackend(
            model=model,
            api_key=api_key,
            base_url=config.get("base_url", "https://api.anthropic.com/v1"),
            default_temperature=config.get("temperature", 0.3),
            default_max_tokens=config.get("max_tokens", 2048)
        )
    elif model == "mock" or not api_key:
        return MockLLMBackend()
    else:
        return OpenAILLMBackend(
            model=model,
            api_key=api_key,
            base_url=config.get("base_url", "https://api.openai.com/v1"),
            default_temperature=config.get("temperature", 0.3),
            default_max_tokens=config.get("max_tokens", 2048)
        )
