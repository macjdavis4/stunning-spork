"""
Base agent class for FINCON system.

Defines the abstract interface for all agents (analysts and manager).
"""

import json
from abc import ABC, abstractmethod
from typing import Any

from fincon.llm_backend import LLMBackend
from fincon.memory import ProceduralMemory


class BaseAgent(ABC):
    """
    Abstract base class for all FINCON agents.

    All agents (analysts and manager) inherit from this class and implement:
    - System prompt construction
    - Message building for LLM calls
    - Output parsing
    """

    def __init__(
        self,
        name: str,
        role_description: str,
        llm: LLMBackend,
        memory: ProceduralMemory | None = None
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name
            role_description: Description of agent's role
            llm: LLM backend instance
            memory: Optional procedural memory
        """
        self.name = name
        self.role_description = role_description
        self.llm = llm
        self.memory = memory

    @abstractmethod
    def build_system_prompt(self) -> str:
        """
        Build system prompt for this agent.

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def build_messages(self, **kwargs) -> list[dict[str, str]]:
        """
        Build message list for LLM call.

        Args:
            **kwargs: Context-specific arguments

        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        pass

    def call_llm(
        self,
        messages: list[dict[str, str]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs
    ) -> str:
        """
        Call LLM with messages.

        Args:
            messages: Message list (if None, will call build_messages)
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Arguments passed to build_messages if messages is None

        Returns:
            LLM response text
        """
        if messages is None:
            messages = self.build_messages(**kwargs)

        response = self.llm.chat(
            messages=messages,
            temperature=temperature or 0.3,
            max_tokens=max_tokens or 2048
        )

        return response

    def parse_json_response(self, response: str) -> dict[str, Any]:
        """
        Parse JSON from LLM response.

        Handles cases where response contains markdown code blocks or extra text.

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON cannot be parsed
        """
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()

        # Remove leading/trailing whitespace
        response = response.strip()

        # Try to parse JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Try to find JSON object in response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                try:
                    return json.loads(response[start_idx:end_idx])
                except json.JSONDecodeError:
                    pass

            raise ValueError(f"Failed to parse JSON from response: {e}\nResponse: {response[:200]}")

    def log_to_memory(
        self,
        content: dict[str, Any],
        tags: list[str] | None = None,
        importance: float = 1.0
    ) -> None:
        """
        Log event to procedural memory.

        Args:
            content: Content to log
            tags: Optional tags
            importance: Importance weight
        """
        if self.memory:
            if tags is None:
                tags = [self.name]
            else:
                tags = [self.name] + tags

            self.memory.add(content, tags=tags, importance=importance)

    def get_memory_context(self, k: int = 5) -> list[dict[str, Any]]:
        """
        Get recent memory context.

        Args:
            k: Number of recent memories to retrieve

        Returns:
            List of memory contents
        """
        if self.memory:
            return self.memory.retrieve_recent(k)
        return []
