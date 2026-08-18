"""Validation result value object."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Result:
    name: str
    evidence: str
    method: str
    metric: str
    value: float
    expected_value: float
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["difference"] = self.value - self.expected_value
        return payload
