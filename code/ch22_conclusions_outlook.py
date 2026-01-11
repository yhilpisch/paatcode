from dataclasses import dataclass  # configuration container
from pathlib import Path  # filesystem paths for plan output
from typing import List  # type hints for task lists

import json  # serialisation of learning plans

"""
Python & AI for Algorithmic Trading
Chapter 22 -- Conclusions and Outlook

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Simple learning-plan helper aligned with the concluding chapter.

The idea is to capture concrete next steps in a small JSON file that
can live next to notebooks and configuration. This does not replace a
full project-management tool but provides a light-weight structure you
can adapt for personal study plans or research roadmaps.
"""


@dataclass
class LearningTask:
    """Single task in a learning or research plan."""

    area: str
    description: str
    estimated_hours: float
    priority: int


@dataclass
class LearningPlan:
    """Container for multiple tasks and simple summaries."""

    name: str
    tasks: List[LearningTask]

    def to_json(self) -> str:
        """Serialise the plan to JSON."""
        payload = {
            "name": self.name,
            "tasks": [
                {
                    "area": t.area,
                    "description": t.description,
                    "estimated_hours": t.estimated_hours,
                    "priority": t.priority,
                }
                for t in self.tasks
            ],
        }
        return json.dumps(payload, indent=2)

    def save(self, path: Path) -> Path:
        """Write the plan to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf8")
        return path


def demo_plan() -> LearningPlan:
    """Create a small example learning plan."""
    tasks = [
        LearningTask(
            area="data",
            description="Rebuild EOD and intraday panels for two "
            "liquid instruments and document all filters.",
            estimated_hours=6.0,
            priority=1,
        ),
        LearningTask(
            area="backtesting",
            description="Implement an additional strategy in the "
            "Chapter 7 framework and compare it with existing "
            "benchmarks.",
            estimated_hours=5.0,
            priority=2,
        ),
        LearningTask(
            area="risk",
            description="Extend post-trade analytics from Chapter 19 "
            "to include instrument-level concentration limits.",
            estimated_hours=4.0,
            priority=2,
        ),
        LearningTask(
            area="ai",
            description="Design one AI-assisted research workflow "
            "using the prompt session helpers from Chapter 20 and "
            "record at least two experiments.",
            estimated_hours=3.0,
            priority=3,
        ),
    ]
    return LearningPlan(name="pyaialgo_demo_plan", tasks=tasks)


def main() -> None:
    """Generate and persist the example learning plan."""
    plan = demo_plan()
    out_path = Path("reports") / "learning_plan_pyaialgo.json"
    path = plan.save(out_path)
    print(f"Saved example learning plan to {path}")


if __name__ == "__main__":
    main()

