from dataclasses import dataclass  # configuration containers
from pathlib import Path  # filesystem paths for transcripts
from typing import Dict, List  # type hints for prompt pieces

import json  # serialisation for saved prompt sessions

"""
Python & AI for Algorithmic Trading
Chapter 20 -- AI-Enhanced Research and Automation Workflows

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Prompt and transcript helpers for AI-assisted research workflows.

This module does not depend on any particular model provider. Instead,
it focuses on structuring prompts and recording interactions so that
they can be inspected, reproduced, and audited alongside code and
data. You can plug your preferred model client into the small hooks
provided here.
"""


@dataclass
class PromptTemplate:
    """Simple prompt template with placeholders."""

    role: str
    goal: str
    instructions: str
    examples: List[str]

    def render(self, context: Dict[str, str]) -> str:
        """Render the template with context variables."""
        parts: List[str]=[]
        parts.append(f"Role: {self.role}")
        parts.append(f"Goal: {self.goal}")
        parts.append("")
        parts.append("Instructions:")
        parts.append(self.instructions.format(**context))
        if self.examples:
            parts.append("")
            parts.append("Examples:")
            for ex in self.examples:
                parts.append(f"- {ex}")
        return "\n".join(parts)


@dataclass
class PromptSession:
    """Record of prompts and model-style responses for auditing."""

    name: str
    items: List[Dict[str, str]]

    def add(self, prompt: str, response: str) -> None:
        """Append a new prompt-response pair to the session."""
        self.items.append(
            {"prompt": prompt, "response": response},
        )

    def to_json(self) -> str:
        """Serialise the session to a JSON string."""
        return json.dumps(
            {"name": self.name, "items": self.items},
            indent=2,
        )

    def save(self, path: Path) -> Path:
        """Save the session transcript to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf8")
        return path


def build_backtest_review_template() -> PromptTemplate:
    """Create a default template for backtest review tasks."""
    return PromptTemplate(
        role="Quantitative research assistant",
        goal=(
            "Review a backtest report and suggest diagnostics, "
            "stress tests, and next steps."
        ),
        instructions=(
            "You are given summary metrics and configuration for a "
            "trading strategy.\n"
            "Metrics:\n"
            "  - annualised return: {ann_return}\n"
            "  - annualised volatility: {ann_vol}\n"
            "  - max drawdown: {max_dd}\n"
            "  - Sharpe ratio: {sharpe}\n"
            "Strategy notes: {notes}\n\n"
            "Write a numbered list of suggestions that focus on "
            "improving the quality of evidence rather than on "
            "tweaking parameters blindly."
        ),
        examples=[
            "Compare results to a simple buy-and-hold benchmark.",
            "Check sensitivity to transaction costs and slippage.",
            "Investigate performance by sub-period and regime.",
        ],
    )


def demo_local_session() -> PromptSession:
    """Create a small example prompt session without model calls."""
    template = build_backtest_review_template()
    context = {
        "ann_return": "0.18",
        "ann_vol": "0.24",
        "max_dd": "-0.32",
        "sharpe": "0.75",
        "notes": (
            "Time-series momentum strategy on liquid futures with "
            "daily rebalancing and modest leverage."
        ),
    }
    prompt = template.render(context)

    response = (
        "1. Compare the strategy to a futures buy-and-hold or "
        "simple moving-average crossover benchmark using the same "
        "cost assumptions.\n"
        "2. Break down performance by contract, sector, and decade "
        "to see where most risk and return originate.\n"
        "3. Re-run the backtest with higher transaction-cost and "
        "slippage settings to check how sensitive the Sharpe ratio "
        "is to implementation details.\n"
        "4. Conduct a simple parameter stability check by varying "
        "the lookback window around its current value.\n"
        "5. Review trade-level logs for periods around the maximum "
        "drawdown to understand whether behaviour matches design."
    )

    session = PromptSession(name="demo_backtest_review", items=[])
    session.add(prompt=prompt, response=response)
    return session


def main() -> None:
    """Create and persist a small demo prompt session."""
    session = demo_local_session()
    out_path = Path("reports") / "demo_backtest_review_prompt.json"
    path = session.save(out_path)
    print(f"Saved demo prompt session to {path}")


if __name__ == "__main__":
    main()

