from __future__ import annotations
from typing import Any, Dict

class Settings:
    """
    Settings class similar to hold global settings and options.

    ---- Parameters ----
    atol : float
        Absolute tolerance used in numerical comparisons.
    rtol : float
        Relative tolerance used in numerical comparisons.
    auto_tidyup : bool
        If True, elements smaller in magnitude than auto_tidyup_atol are removed when creating 
        QGstates, QGopers, and QGsupers.
    auto_tidyup_atol : float
        Defauly lower limit magnitude for array elements, below which they are considered zero in tidyup operations.

    """
    def __init__(self,
                 atol: float = 1e-12,
                 rtol: float = 1e-12,
                 auto_tidyup: bool = True,
                 auto_tidyup_atol: float = 1e-12,
                 ):
        
        self.atol = atol
        self.rtol = rtol
        self.auto_tidyup = auto_tidyup
        self.auto_tidyup_atol = auto_tidyup_atol

    def update(self, **kwargs: Any) -> None:
        """ Update settings from keyword arguments, for example: settings.update(atol=1e-10, auto_tidyup=False) """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown setting: {key}")

    def reset(self) -> None:
        """ Replace settings with their default values. """
        self.atol = 1e-12
        self.rtol = 1e-12
        self.auto_tidyup = True
        self.auto_tidyup_atol = 1e-12

    def as_dict(self) -> Dict[str, Any]:
        """ Return settings as a dictionary; to be passed to ODE solvers. """
        return {"atol": self.atol,
                "rtol": self.rtol,
                "auto_tidyup": self.auto_tidyup,
                "auto_tidyup_atol": self.auto_tidyup_atol
                }