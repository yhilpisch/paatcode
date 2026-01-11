from dataclasses import dataclass  # configuration containers
from pathlib import Path  # filesystem paths for config output
from typing import Dict, List  # type hints for service maps

import json  # serialisation for schedule previews

"""
Python & AI for Algorithmic Trading
Chapter 17 -- Containers, Local Clouds, and Deployment Patterns

(c) Dr. Yves J. Hilpisch | The Python Quants GmbH
AI-powered by GPT 5.1 | https://linktr.ee/dyjh

Lightweight helpers for container-style deployment planning.

The goal of this script is not to manage real containers but to keep
deployment thinking concrete at the level of configuration files. The
helpers below model small services as data classes and translate them
into a minimal ``docker-compose``-style configuration that you can
adapt to your own environment.

You can treat this module as a bridge between the conceptual material
of Chapter 17 and practical experiments with single-node deployments.
"""


@dataclass
class ServiceSpec:
    """Description of a small service or job container."""

    name: str  # short identifier, for example "data_collector"
    image: str  # base image name, for example "python:3.12-slim"
    command: str  # shell command executed inside the container
    schedule: str  # human-readable schedule, such as "daily 22:15"
    env: Dict[str, str]  # small set of environment variables
    restart: str="unless-stopped"  # restart policy for the service


def service_to_compose(service: ServiceSpec) -> Dict[str, object]:
    """Translate a ServiceSpec into a docker-compose service entry."""
    return {
        "image": service.image,
        "command": service.command,
        "environment": service.env,
        "restart": service.restart,
    }


def build_compose_config(services: List[ServiceSpec]) -> Dict[str, object]:
    """Build a minimal compose configuration from service specs."""
    return {
        "version": "3.9",
        "services": {
            spec.name: service_to_compose(spec) for spec in services
        },
    }


def write_compose_yaml(
    config: Dict[str, object],
    path: Path,
) -> Path:
    """Write a very small subset of YAML for the compose config.

    The implementation uses manual string formatting instead of
    depending on a third-party YAML library. The resulting file is
    intentionally compact and easy to inspect.
    """
    lines: List[str]=[]

    lines.append("version: '3.9'\n")
    lines.append("services:\n")

    services = config.get("services", {})
    for name, spec in services.items():
        lines.append(f"  {name}:\n")
        image = spec.get("image", "")
        command = spec.get("command", "")
        restart = spec.get("restart", "unless-stopped")
        env = spec.get("environment", {})

        lines.append(f"    image: {image}\n")
        if command:
            lines.append(f"    command: {command}\n")
        lines.append(f"    restart: {restart}\n")

        if env:
            lines.append("    environment:\n")
            for key, value in env.items():
                lines.append(f"      {key}: \"{value}\"\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf8")
    return path


def preview_schedule(services: List[ServiceSpec]) -> str:
    """Return a small JSON preview of service schedules."""
    schedule_info = [
        {
            "name": s.name,
            "schedule": s.schedule,
            "command": s.command,
        }
        for s in services
    ]
    return json.dumps(schedule_info, indent=2)


def main() -> None:
    """Create an example compose file and print schedule preview."""
    services = [
        ServiceSpec(
            name="data_collector",
            image="python:3.12-slim",
            command="python code/ch05_eod_engineering.py",
            schedule="daily 22:15",
            env={"PROFILE": "prod"},
        ),
        ServiceSpec(
            name="strategy_loop",
            image="python:3.12-slim",
            command="python code/ch07_baseline_strategies.py",
            schedule="market-hours",
            env={"PROFILE": "prod"},
        ),
        ServiceSpec(
            name="report_runner",
            image="python:3.12-slim",
            command="python code/ch16_reporting_monitoring.py",
            schedule="daily 22:30",
            env={"PROFILE": "prod"},
        ),
    ]

    config = build_compose_config(services)
    out_path = Path("deploy") / "docker-compose.example.yml"
    path = write_compose_yaml(config, out_path)

    print(f"Wrote example compose file to {path}")
    print("\nSchedule preview:")
    print(preview_schedule(services))


if __name__ == "__main__":
    main()

