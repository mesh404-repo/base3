"""Configuration models for SuperAgent using standard dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for the model."""
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class OutputMode(str, Enum):
    """Output mode for the agent."""
    HUMAN = "human"
    JSON = "json"


class Provider(str, Enum):
    """LLM provider."""
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ReasoningConfig:
    """Configuration for model reasoning."""
    effort: ReasoningEffort = ReasoningEffort.HIGH


@dataclass
class CacheConfig:
    """Configuration for prompt caching."""
    enabled: bool = True


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    retry_on_status: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])


@dataclass
class ToolsConfig:
    """Configuration for available tools."""
    shell_enabled: bool = True
    shell_timeout: int = 30
    file_ops_enabled: bool = True
    max_file_size: int = 1048576
    grep_enabled: bool = True
    max_grep_results: int = 100


@dataclass
class OutputConfig:
    """Configuration for output formatting."""
    mode: OutputMode = OutputMode.HUMAN
    streaming: bool = True
    colors: bool = True


@dataclass
class PathsConfig:
    """Configuration for file paths."""
    cwd: str = ""
    readable_roots: List[str] = field(default_factory=list)
    writable_roots: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Resolve empty cwd to current directory."""
        if not self.cwd:
            self.cwd = os.getcwd()
        else:
            self.cwd = str(Path(self.cwd).resolve())


@dataclass
class AgentConfig:
    """Main configuration for the SuperAgent."""
    
    # Model settings
    model: str = "anthropic/claude-opus-4.5"
    provider: Provider = Provider.OPENROUTER
    max_iterations: int = 200
    timeout: int = 600
    temperature: float = 0.0
    max_tokens: int = 16384
    
    # Sub-configurations
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    @property
    def working_directory(self) -> Path:
        """Get the working directory as a Path object."""
        return Path(self.paths.cwd or os.getcwd())
    
    def get_api_key(self) -> str:
        """Get the API key for the configured provider."""
        env_vars = {
            Provider.OPENROUTER: ["OPENROUTER_API_KEY"],
            Provider.OPENAI: ["OPENAI_API_KEY"],
            Provider.ANTHROPIC: ["ANTHROPIC_API_KEY"],
        }
        
        for var in env_vars.get(self.provider, []):
            key = os.environ.get(var)
            if key:
                return key
        
        raise ValueError(f"No API key found for provider {self.provider}. "
                        f"Set one of: {env_vars.get(self.provider, [])}")
    
    def get_base_url(self) -> str:
        """Get the base URL for the configured provider."""
        urls = {
            Provider.OPENROUTER: "https://openrouter.ai/api/v1",
            Provider.OPENAI: "https://api.openai.com/v1",
            Provider.ANTHROPIC: "https://api.anthropic.com/v1",
        }
        return urls[self.provider]
