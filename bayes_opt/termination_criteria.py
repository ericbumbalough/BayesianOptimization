"""Termination criteria."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from bayes_opt import BayesianOptimization


class TerminationCriteria:
    """Termination criteria.

    If any criterion is met the iteration will stop.

    Parameters
    ----------
    termination_value: float, optional(default=0)
        Terminate if the maximum is greater than this value.
    termination_wall_time: datetime.timedelta, optional(default=timedelta(0))
        Terminate if the elapsed wall time is greater than this value.
    termination_improvement_value: float, optional(default=0)
        Terminate if improvements in the termination are less then or equal to this value for
        `termination_improvement_iter` consecutive iterations.
    termination_improvement_iter: int, optional(default=0)
        See `termination_improvement_value`.
    """

    def __init__(
        self,
        termination_value: float = 0,
        termination_wall_time: timedelta = timedelta(0),
        termination_improvement_value: float = 0,
        termination_improvement_iter: int = 0,
    ):
        self.termination_value = termination_value
        self.termination_wall_time = termination_wall_time
        self.termination_improvement_value = termination_improvement_value
        self.termination_improvement_iter = termination_improvement_iter

    def _state_to_dict(self) -> dict[str, float | timedelta]:
        return {
            "termination_value": self.termination_value,
            "termination_wall_time": str(self.termination_wall_time),
            "termination_improvement_value": self.termination_improvement_value,
            "termination_improvement_iter": self.termination_improvement_iter,
        }

    def met(self, opt: BayesianOptimization) -> bool:
        """Return true if a termination criterion has been met."""
        if opt._iterations >= opt.n_iter:
            return True

        if self.termination_value and opt.max is not None and opt.max["target"] >= self.termination_value:
            return True

        if (
            self.termination_wall_time
            and datetime.now(timezone.utc) - opt._start_time > self.termination_wall_time
        ):
            return True

        if (
            self.termination_improvement_iter
            and len(opt._space.target) >= self.termination_improvement_iter + 1
        ):
            # Determine improvements that have occurred each iteration
            improvements = np.diff(np.maximum.accumulate(opt._space.target))
            # Check if there are improvements in the specified number of iterations
            relevant_improvements = improvements[-self.termination_improvement_iter :]
            if relevant_improvements.max() <= self.termination_improvement_value:
                # There has been no large enough improvement within the iterations specified
                return True
        return False
