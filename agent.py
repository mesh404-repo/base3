#!/usr/bin/env python3
"""
SuperAgent for Term Challenge - Entry Point.

This agent simulates the behavior of:
codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check \
  --enable unified_exec -c model_reasoning_effort=xhigh --model gpt-5.2 --json

All settings are HARDCODED for benchmark mode.
No CLI arguments needed - instruction comes from term_sdk context.

Usage:
    python agent.py
    # or
    from term_sdk import run
    run(SuperAgent())
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from term_sdk import Agent, AgentContext, run

from src.config.defaults import CONFIG
from src.config.models import AgentConfig, Provider
from src.api.client import LLMClient, LLMError, CostLimitExceeded
from src.core.loop import run_agent_loop
from src.tools.registry import ToolRegistry
from src.output.jsonl import emit, ErrorEvent


class SuperAgent(Agent):
    """
    Main agent class for Term Challenge benchmark.
    
    Simulates Codex exec with:
    - Model: gpt-5.2
    - Reasoning effort: xhigh
    - All safety bypasses enabled
    - JSONL output format
    """
    
    def setup(self):
        """Initialize LLM client and tools (called once at startup)."""
        # Build AgentConfig from CONFIG dictionary
        agent_config = AgentConfig(
            model=CONFIG["model"],
            provider=Provider(CONFIG.get("provider", "openrouter")),
            max_iterations=CONFIG.get("max_iterations", 200),
            timeout=CONFIG.get("shell_timeout", 60),
            temperature=CONFIG.get("temperature", 0.0),
            max_tokens=CONFIG["max_tokens"],
        )
        
        # Initialize LLMClient with litellm
        self.llm = LLMClient(agent_config)
        self.tools = ToolRegistry()
        # self._start_time = time.time()
        
        self._log(f"SuperAgent initialized")
        self._log(f"Model: {CONFIG['model']}")
        self._log(f"Reasoning effort: {CONFIG['reasoning_effort']}")
        self._log(f"Using litellm for OpenRouter")
    
    def run(self, ctx: AgentContext):
        """
        Execute the task (called for each task).
        
        The instruction comes from ctx.instruction - no CLI arguments needed.
        """
        self._log(f"Task received: {ctx.instruction[:100]}...")
        
        try:
            run_agent_loop(
                llm=self.llm,
                tools=self.tools,
                ctx=ctx,
                config=CONFIG,
            )
        except CostLimitExceeded as e:
            self._log(f"Cost limit exceeded: ${e.used:.4f} / ${e.limit:.4f}")
            emit(ErrorEvent(message=f"Cost limit exceeded: {e}"))
            ctx.done()
        except Exception as e:
            self._log(f"Fatal error: {e}")
            emit(ErrorEvent(message=str(e)))
            ctx.done()
            raise
    
    def cleanup(self):
        """Print stats and cleanup (called at shutdown)."""
        # elapsed = time.time() - self._start_time
        
        # self._log(f"Elapsed: {elapsed:.1f}s")
        
        try:
            self.llm.close()
        except Exception:
            pass
    
    def _log(self, msg: str):
        """Log to stderr (not mixed with JSONL on stdout)."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [superagent] {msg}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    run(SuperAgent())
