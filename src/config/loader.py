"""Configuration loader for SuperAgent."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from src.config.models import (
    AgentConfig,
    ReasoningConfig,
    CacheConfig,
    RetryConfig,
    ToolsConfig,
    OutputConfig,
    PathsConfig,
    Provider,
    ReasoningEffort,
    OutputMode,
)


def _create_config_from_dict(data: dict[str, Any]) -> AgentConfig:
    """Create AgentConfig from a dictionary."""
    # Build sub-configs
    reasoning_data = data.get("reasoning", {})
    if isinstance(reasoning_data, dict):
        effort = reasoning_data.get("effort", "high")
        if isinstance(effort, str):
            effort = ReasoningEffort(effort)
        reasoning = ReasoningConfig(effort=effort)
    else:
        reasoning = ReasoningConfig()
    
    cache_data = data.get("cache", {})
    if isinstance(cache_data, dict):
        cache = CacheConfig(enabled=cache_data.get("enabled", True))
    else:
        cache = CacheConfig()
    
    retry_data = data.get("retry", {})
    if isinstance(retry_data, dict):
        retry = RetryConfig(
            max_attempts=retry_data.get("max_attempts", 5),
            base_delay=retry_data.get("base_delay", 1.0),
            max_delay=retry_data.get("max_delay", 60.0),
            retry_on_status=retry_data.get("retry_on_status", [429, 500, 502, 503, 504]),
        )
    else:
        retry = RetryConfig()
    
    tools_data = data.get("tools", {})
    if isinstance(tools_data, dict):
        tools = ToolsConfig(
            shell_enabled=tools_data.get("shell_enabled", True),
            shell_timeout=tools_data.get("shell_timeout", 30),
            file_ops_enabled=tools_data.get("file_ops_enabled", True),
            max_file_size=tools_data.get("max_file_size", 1048576),
            grep_enabled=tools_data.get("grep_enabled", True),
            max_grep_results=tools_data.get("max_grep_results", 100),
        )
    else:
        tools = ToolsConfig()
    
    output_data = data.get("output", {})
    if isinstance(output_data, dict):
        mode = output_data.get("mode", "human")
        if isinstance(mode, str):
            mode = OutputMode(mode)
        output = OutputConfig(
            mode=mode,
            streaming=output_data.get("streaming", True),
            colors=output_data.get("colors", True),
        )
    else:
        output = OutputConfig()
    
    paths_data = data.get("paths", {})
    if isinstance(paths_data, dict):
        paths = PathsConfig(
            cwd=paths_data.get("cwd", ""),
            readable_roots=paths_data.get("readable_roots", []),
            writable_roots=paths_data.get("writable_roots", []),
        )
    else:
        paths = PathsConfig()
    
    # Get provider
    provider = data.get("provider", "openrouter")
    if isinstance(provider, str):
        provider = Provider(provider)
    
    return AgentConfig(
        model=data.get("model", "anthropic/claude-opus-4.5"),
        provider=provider,
        max_iterations=data.get("max_iterations", 50),
        timeout=data.get("timeout", 120),
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 16384),
        reasoning=reasoning,
        cache=cache,
        retry=retry,
        tools=tools,
        output=output,
        paths=paths,
    )


def load_config_from_file(path: Path) -> AgentConfig:
    """Load configuration from a TOML file.
    
    Args:
        path: Path to the TOML configuration file.
        
    Returns:
        AgentConfig instance with loaded configuration.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config file is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "rb") as f:
        raw_config = tomllib.load(f)
    
    # Merge agent section with top level
    config_dict: dict[str, Any] = {}
    
    if "agent" in raw_config:
        config_dict.update(raw_config["agent"])
    
    # Add other sections
    for section in ["cache", "retry", "tools", "output", "paths", "reasoning"]:
        if section in raw_config:
            config_dict[section] = raw_config[section]
    
    return _create_config_from_dict(config_dict)


def load_config(
    config_path: Optional[Path] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> AgentConfig:
    """Load configuration with optional overrides.
    
    Args:
        config_path: Optional path to a TOML config file.
        overrides: Optional dictionary of configuration overrides.
        
    Returns:
        AgentConfig instance.
    """
    # Start with defaults
    config_dict: dict[str, Any] = {}
    
    # Load from file if provided
    if config_path and config_path.exists():
        with open(config_path, "rb") as f:
            raw_config = tomllib.load(f)
        
        # Transform TOML structure
        if "agent" in raw_config:
            for key, value in raw_config["agent"].items():
                config_dict[key] = value
        
        for section in ["cache", "retry", "tools", "output", "paths", "reasoning"]:
            if section in raw_config:
                config_dict[section] = raw_config[section]
    
    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            if "." in key:
                # Handle nested keys like "cache.enabled"
                parts = key.split(".")
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
    
    return _create_config_from_dict(config_dict)


def find_config_file() -> Optional[Path]:
    """Find the configuration file in standard locations.
    
    Searches in order:
    1. ./config.toml
    2. ./superagent.toml
    3. ~/.config/superagent/config.toml
    4. ~/.superagent/config.toml
    
    Returns:
        Path to the config file if found, None otherwise.
    """
    search_paths = [
        Path.cwd() / "config.toml",
        Path.cwd() / "superagent.toml",
        Path.home() / ".config" / "superagent" / "config.toml",
        Path.home() / ".superagent" / "config.toml",
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None
