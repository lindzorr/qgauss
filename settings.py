from __future__ import annotations
import numbers
from typing import Any, Dict

class Settings:
    """
    Global settings and options.

    ---- Parameters ----
    atol : float
        Absolute tolerance used in numerical comparisons.
    rtol : float
        Relative tolerance used in numerical comparisons.
    auto_tidyup : bool
        If True, elements smaller in magnitude than auto_tidyup_atol are 
        removed when creating QGstates, QGopers, and QGsupers.
    tidyup_atol : float
        Defauly lower limit magnitude for array elements, below which they are 
        considered zero in tidyup operations.

    """
    _defaults: Dict[str, Any] = {"atol": 1e-12,
                                 "rtol": 1e-12,
                                 "auto_tidyup": True,
                                 "tidyup_atol": 1e-12}

    def __init__(self, **kwargs: Any) -> None:
        self.reset()
        self.update(**kwargs)

    @classmethod
    def _validate(cls, key: str, value: Any) -> None:
        if key not in cls._defaults:
            raise ValueError(f"Unknown setting: {key}")
        if key == "auto_tidyup":
            if not isinstance(value, bool):
                raise TypeError("auto_tidyup must be a bool.")
        elif (isinstance(value, bool) or not isinstance(value, numbers.Real)
              or not value >= 0):                      # also rejects NaN
            raise ValueError(f"{key} must be a non-negative real number.")

    def __setattr__(self, key: str, value: Any) -> None:
        self._validate(key, value)
        super().__setattr__(key, value)

    def update(self, **kwargs: Any) -> None:
        """ Update settings, e.g. settings.update(atol=1e-10, auto_tidyup=False).
        Nothing is changed if any key or value is invalid. """
        for key, value in kwargs.items():
            self._validate(key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def reset(self) -> None:
        """ Replace settings with their default values. """
        self.update(**self._defaults)

    def as_dict(self) -> Dict[str, Any]:
        """ Return all settings as a dictionary. """
        return {key: getattr(self, key) for key in self._defaults}

    def solver_options(self) -> Dict[str, float]:
        """ Tolerances only; safe to pass to ODE solvers. """
        return {"atol": self.atol, "rtol": self.rtol}