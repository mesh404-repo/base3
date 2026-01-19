"""LLM Client for SuperAgent - uses direct HTTP requests to OpenRouter."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional

import httpx

from src.config.models import AgentConfig, Provider
from src.api.retry import RetryHandler


# Custom exceptions to replace term_sdk exceptions
class LLMError(Exception):
    """Base exception for LLM errors."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class CostLimitExceeded(LLMError):
    """Raised when cost limit is exceeded."""
    def __init__(self, used: float, limit: float):
        self.used = used
        self.limit = limit
        super().__init__("cost_limit_exceeded", f"Cost ${used:.4f} exceeded limit ${limit:.4f}")


@dataclass
class FunctionCall:
    """Represents a function/tool call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]
    
    @classmethod
    def from_openai(cls, call: dict[str, Any]) -> "FunctionCall":
        """Parse from OpenAI tool_calls format."""
        func = call.get("function", {})
        args_str = func.get("arguments", "{}")
        
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {"raw": args_str}
        
        return cls(
            id=call.get("id", ""),
            name=func.get("name", ""),
            arguments=args,
        )
    
    @classmethod
    def from_anthropic(cls, content: dict[str, Any]) -> "FunctionCall":
        """Parse from Anthropic tool_use format."""
        return cls(
            id=content.get("id", ""),
            name=content.get("name", ""),
            arguments=content.get("input", {}),
        )


@dataclass
class LLMResponse:
    """Response from the LLM."""
    text: str = ""
    function_calls: list[FunctionCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    raw: Optional[dict[str, Any]] = None
    cost: float = 0.0
    
    @property
    def has_function_calls(self) -> bool:
        """Check if response contains function calls."""
        return len(self.function_calls) > 0
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens
    
    @property
    def tokens(self) -> dict[str, int]:
        """Token usage as dict (for compatibility)."""
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "cached": self.cached_tokens,
        }


class LLMClient:
    """LLM Client using direct HTTP requests to OpenRouter."""
    
    # OpenRouter pricing per 1M tokens (approximate, for cost estimation)
    # Format: model_pattern -> (input_cost, output_cost)
    MODEL_PRICING = {
        "claude-opus-4": (15.0, 75.0),
        "claude-sonnet-4": (3.0, 15.0),
        "claude-3.5-sonnet": (3.0, 15.0),
        "claude-3-opus": (15.0, 75.0),
        "gpt-4o": (2.5, 10.0),
        "gpt-4-turbo": (10.0, 30.0),
        "gpt-4": (30.0, 60.0),
        "gpt-3.5": (0.5, 1.5),
        "deepseek": (0.14, 0.28),
    }
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.retry_handler = RetryHandler(config.retry)
        self._http_client: Optional[httpx.Client] = None
        self._seen_content_hashes: set[str] = set()
        
        # Initialize HTTP client
        self._init_http_client()
    
    def _init_http_client(self) -> None:
        """Initialize HTTP client for OpenRouter API."""
        api_key = self.config.get_api_key()
        
        # Check for LLM proxy URL (used in evaluation)
        # base_url = os.environ.get("LLM_PROXY_URL")
        # if not base_url:
        base_url = self.config.get_base_url()
        
        self._base_url = base_url
        
        self._http_client = httpx.Client(
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/PlatformNetwork/term-challenge",
                "X-Title": "Term Challenge Agent",
            },
        )
    
    def _get_model_name(self) -> str:
        """Get the model name for the API."""
        model = self.config.model
        
        # For OpenRouter, model names may already include provider prefix
        if self.config.provider == Provider.OPENROUTER:
            return model
        
        # For direct providers, strip provider prefix if present
        if "/" in model:
            return model.split("/", 1)[1]
        
        return model
    
    def _supports_temperature(self, model: str) -> bool:
        """Check if the model supports the temperature parameter.
        
        Reasoning models like o1, o3, deepseek-r1 don't support temperature.
        """
        model_lower = model.lower()
        # OpenAI reasoning models
        if model_lower.startswith("o1") or model_lower.startswith("o3"):
            return False
        if "/o1" in model_lower or "/o3" in model_lower:
            return False
        # DeepSeek reasoning models
        if "deepseek-r1" in model_lower or "deepseek/deepseek-r1" in model_lower:
            return False
        return True
    
    def _build_tools_json(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build tools JSON for the API."""
        result = []
        for tool in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        return result
    
    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on token usage."""
        model_lower = model.lower()
        
        # Find matching pricing
        input_cost_per_m = 3.0  # default
        output_cost_per_m = 15.0  # default
        
        for pattern, (inp, out) in self.MODEL_PRICING.items():
            if pattern in model_lower:
                input_cost_per_m = inp
                output_cost_per_m = out
                break
        
        cost = (input_tokens / 1_000_000) * input_cost_per_m
        cost += (output_tokens / 1_000_000) * output_cost_per_m
        return cost
    
    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse API response to LLMResponse."""
        response = LLMResponse(raw=data)
        
        # Parse usage
        usage = data.get("usage", {})
        response.input_tokens = usage.get("prompt_tokens", 0)
        response.output_tokens = usage.get("completion_tokens", 0)
        
        # Check for cached tokens
        prompt_details = usage.get("prompt_tokens_details", {})
        response.cached_tokens = prompt_details.get("cached_tokens", 0)
        
        # Parse model
        response.model = data.get("model", self.config.model)
        
        # Parse choices
        choices = data.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            response.finish_reason = choice.get("finish_reason", "")
            
            # Text content
            response.text = message.get("content") or ""
            
            # Function calls
            tool_calls = message.get("tool_calls", [])
            for call in tool_calls:
                if call.get("type") == "function":
                    response.function_calls.append(FunctionCall.from_openai(call))
        
        # Get actual cost from OpenRouter (when "usage": {"include": True} is set)
        # OpenRouter returns cost in credits (USD) in the usage object
        if "cost" in usage:
            response.cost = float(usage["cost"])
        else:
            # Fallback to estimation if cost not provided
            response.cost = self._estimate_cost(
                response.model,
                response.input_tokens,
                response.output_tokens
            )
        
        return response
    
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[dict[str, Any]] = None,
        on_retry: Optional[Callable[[Any], None]] = None,
    ) -> LLMResponse:
        """Send a chat request to the LLM.
        
        Args:
            messages: List of messages in OpenAI format
            tools: Optional list of tools/functions
            max_tokens: Optional max tokens override
            extra_body: Optional extra body parameters (e.g., reasoning)
            on_retry: Optional callback for retry events
            
        Returns:
            LLMResponse with text and/or function calls
        """
        if self._http_client is None:
            raise RuntimeError("HTTP client not initialized")
        
        # Build request body
        model_name = self._get_model_name()
        body: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            # Request usage info including actual cost from OpenRouter
            "usage": {"include": True},
        }
        
        # Only add temperature if the model supports it
        if self._supports_temperature(model_name):
            body["temperature"] = self.config.temperature
        
        if tools:
            body["tools"] = self._build_tools_json(tools)
            body["tool_choice"] = "auto"
        
        # Add extra body params (e.g., reasoning effort)
        if extra_body:
            body.update(extra_body)
        
        # Make request with retry
        def do_request() -> dict[str, Any]:
            url = f"{self._base_url}/chat/completions"
            resp = self._http_client.post(url, json=body)  # type: ignore
            
            # Handle errors
            if resp.status_code != 200:
                try:
                    error_data = resp.json()
                    error_msg = error_data.get("error", {}).get("message", resp.text)
                except Exception:
                    error_msg = resp.text
                
                raise LLMError(
                    code=f"http_{resp.status_code}",
                    message=f"API error: {error_msg}"
                )
            
            return resp.json()
        
        data = self.retry_handler.execute(do_request, on_retry=on_retry)
        return self._parse_response(data)
    
    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> Generator[str, None, LLMResponse]:
        """Stream a chat response.
        
        Args:
            messages: List of messages
            tools: Optional list of tools
            on_chunk: Optional callback for each text chunk
            
        Yields:
            Text chunks
            
        Returns:
            Final LLMResponse with complete text and usage
        """
        # For now, use non-streaming and yield the full response
        # TODO: Implement proper SSE streaming
        response = self.chat(messages, tools)
        
        if response.text:
            if on_chunk:
                on_chunk(response.text)
            yield response.text
        
        return response
    
    def close(self) -> None:
        """Close the client and release resources."""
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
