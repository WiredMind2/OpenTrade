"""
Walk-forward splitter for time-series validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass
class WalkForwardSplit:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


class WalkForwardSplitter:
    def __init__(
        self,
        train_mode: str = "expanding",
        train_size: int | None = None,
        min_train_size: int = 500,
        test_size: int = 20,
        step_size: int = 20,
        gap: int = 5,
    ):
        self.train_mode = train_mode
        self.train_size = train_size
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.gap = gap

    def split(self, n_samples: int) -> Iterator[WalkForwardSplit]:
        cursor = self.min_train_size
        while cursor + self.gap + self.test_size <= n_samples:
            if self.train_mode == "rolling" and self.train_size:
                train_start = max(0, cursor - self.train_size)
            else:
                train_start = 0
            train_end = cursor
            test_start = cursor + self.gap
            test_end = test_start + self.test_size
            yield WalkForwardSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            cursor += self.step_size
