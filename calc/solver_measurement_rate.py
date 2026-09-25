import warnings
from dataclasses import dataclass, field
import numpy.typing as npt
import qgauss
import numpy as np

from ..core.qgstate import QGstate
from ..core.qgoper import QGoper
from ..dev.qghle import QGhle
from ..calc.utilities import *

__all__ = ['measurement_rate','MeasRateResult','MeasRateResultArray']


@dataclass(frozen=True)
class MeasRateResult:
    """
    Measurement rate result for a single pair of pointer states.
 
    ---- Attributes ----
    signal_A : complex
        Projection of pointer state A onto the measurement quadrature.
    signal_B : complex
        Projection of pointer state B onto the measurement quadrature.
    noise_A : float
        Measurement noise contribution from pointer state A along the
        measurement quadrature
    noise_B : float
        Measurement noise contribution from pointer state B along the
        measurement quadrature
    meas_oper : QGoper
        Linear measurement operator used (or found to be optimal) for this
        pair of pointer states.
    pointers : tuple[int, int]
        The pair of pointer-state indices.
    """
    signal_A: complex
    signal_B: complex
    noise_A: float
    noise_B: float
    meas_oper: QGoper
    pointers: tuple[int, int]

    @property
    def signal(self) -> float:
        # Measurement signal, that is, displacement between pointer states
        # along the quadrature defined by the measurement operator.
        return np.abs(self.signal_A - self.signal_B)

    @property
    def noise(self) -> float:
        # Measurement noise along the quadrature defined by the measurement
        # operator, not including noise_rest.
        return self.noise_A + self.noise_B

    def measrate(self, noise_rest: float = 0) -> float:
        if (self.noise + 2*noise_rest) == 0:
            if self.signal == 0:
                return 0
            else:
                return np.NaN
        # Measurement rate for the pair of pointer states.
        return 0.25 * self.signal**2 / (self.noise + 2*noise_rest)


@dataclass(frozen=True)
class MeasRateResultArray:
    """
    Measurement rate results between all pairs of pointer states. Implementation
    is lazy as the data arrays are identical across the diagonal.
 
    ---- Attributes ----
    signal_A :  np.ndarray
        Projection of pointer states A onto the measurement quadrature.
    signal_B :  np.ndarray
        Projection of pointer states B onto the measurement quadrature.
    noise_A :  np.ndarray
        Measurement noise contribution from pointer states A along the
        measurement quadrature
    noise_B :  np.ndarray
        Measurement noise contribution from pointer states B along the
        measurement quadrature
    meas_oper : list[list[QGoper]]
        Linear measurement operators used (or found to be optimal) for all
        pairs of pointer states.
    """
    signal_A: np.ndarray
    signal_B: np.ndarray
    noise_A: np.ndarray
    noise_B: np.ndarray
    meas_oper: list[list[QGoper]]

    @property
    def signal(self) -> float:
        # Measurement signals, that is, displacement between all pairs of pointer
        # states along the quadrature defined by the measurement operators.
        return np.abs(self.signal_A - self.signal_B)

    @property
    def noise(self) -> float:
        # Total measurement noise along the quadrature defined by the measurement
        # operator, not including noise_rest.
        return self.noise_A + self.noise_B
    
    def measrate(self, noise_rest: float | np.ndarray = 0) -> np.ndarray:
        # Measurement rate for every pair of pointer states.
        _noise_total = self.noise + 2 * noise_rest
        _signal = self.signal
        _measrate = np.empty_like(_signal, dtype=float)
        # Mask for nonzero noise
        _mask = _noise_total != 0
        _measrate[_mask] = 0.25 * _signal[_mask]**2 / _noise_total[_mask]
        # Mask for zero noise
        _zero_noise = np.logical_not(_mask)
        _zero_signal = _zero_noise & (_signal == 0)
        _measrate[_zero_signal] = 0.0
        _measrate[_zero_noise & np.logical_not(_zero_signal)] = np.nan
        return _measrate

    def __getitem__(self, pointers: tuple[int, int]) -> MeasRateResult:
        _row, _col = pointers
        return MeasRateResult(signal_A = self.signal_A[_row,_col],
                              signal_B = self.signal_B[_row,_col],
                              noise_A = self.noise_A[_row,_col],
                              noise_B = self.noise_B[_row,_col],
                              meas_oper = self.meas_oper[_row][_col],
                              pointers = pointers)

    
def measurement_rate(HLE: QGhle,
                     pointers: tuple[int,int] = None, 
                     meas_oper: QGoper = None, 
                     meas_mode: int | list[int] = None, 
                     freq: float = 0, 
                    ) -> MeasRateResult | MeasRateResultArray:
    """
    ---- Procedure ----
    Routine to calculate the steady-state measurement rate between pointer 
    states of an FLS, defined in terms of the SNR as:
        measurement_rate = lim_t→∞ SNR^2(t)/t
                         = signal / noise.
    The measurement "signal" corresponds to the magnitude of the separation
    between the two pointer states, while "noise" represents the total noise
    along the measured quadrature plus any noise from other sources.

    ---- Parameters ----
    HLE : QGoper
        QGhle object representing the Heisenberg-Langevin equations, which 
        encodes the system Hamiltonian, system-environment Hamiltonian, and 
        input state of the environment.
    pointers : tuple[int,int]
        The measurement rate is to be calculated between the pointer states for 
        these two elements of the FLS density matrix. The measurement rate will 
        be zero if both pointer states are the same. If no argument is passed, 
        the measurement rate between all pairs are solved, including redundant 
        pairs.
    meas_oper : QGoper
        The operator to be measured at the output of the system. This operator 
        acts only on the Hilbert space of the
        environment modes and must be linear.
    meas_mode : int or list(int)
        Output modes that are monitored during the measurement. To be used in 
        case no meas_oper is specified. If none are given, then it is assumed 
        that all are monitored. Measurement operator is constructed using 
        weights from Bultink et al, Appl. Phys. Lett. 112, 092601 (2018).
    freq : float
        Frequency at which the measurement is performed, default is zero.

    ---- Returns ----
    MeasRateResult or MeasRateResultArray
        Dataclass storing resulant measurement rate for the pair(s) of pointer 
        states, along with the measurement operator(s).
    """
    if not HLE.isfls:
        raise ValueError("No FLS coupled to system. Measurement rate cannot be defined.")
    elif HLE.isfls and not HLE.iscvs:
        raise ValueError("No CVS component coupled to the FLS component. " \
                         "The measurement rate cannot be defined.")

    # --------------------------------------------------------------------------
    # FLS pointer states specified, solve the corresponding measurement rate. 
    elif HLE.isfls and pointers is not None:
        _pointer_A = HLE.output_env(freq = freq, index = pointers[0])
        _pointer_B = HLE.output_env(freq = freq, index = pointers[1])

        (_signal_A, _signal_B, _noise_A, _noise_B, _meas_oper) = \
            _measurement_rate_solver(pointer_A = _pointer_A, 
                                     pointer_B = _pointer_B, 
                                     meas_oper = meas_oper, 
                                     meas_mode = meas_mode)

        return MeasRateResult(signal_A = _signal_A,
                              signal_B = _signal_B,
                              noise_A = _noise_A,
                              noise_B = _noise_B,
                              meas_oper = _meas_oper,
                              pointers = pointers)
    
    # --------------------------------------------------------------------------
    # No FLS pointer states is specified, solve for all measurment rates.
    elif HLE.isfls and pointers is None:
        _row_total = np.prod(HLE.dims_fls[0])
        _col_total = np.prod(HLE.dims_fls[1])

        _signal_A = np.empty([_row_total,_col_total], dtype=complex)
        _signal_B = np.empty([_row_total,_col_total], dtype=complex)
        _noise_A = np.empty([_row_total,_col_total])
        _noise_B = np.empty([_row_total,_col_total])
        _meas_oper = [[None for _ in range(0,_col_total)] 
                       for _ in range(0, _row_total)]

        for _row in range(0,_row_total):
            for _col in range(0,_col_total):
                _pointer_A = HLE.output_env(freq = freq, index = _row)
                _pointer_B = HLE.output_env(freq = freq, index = _col)

                (_signal_A[_row,_col], 
                 _signal_B[_row,_col], 
                 _noise_A[_row,_col],
                 _noise_B[_row,_col],
                 _meas_oper[_row][_col]) = \
                    _measurement_rate_solver(pointer_A = _pointer_A,
                                             pointer_B = _pointer_B, 
                                             meas_oper = meas_oper, 
                                             meas_mode = meas_mode)

        return MeasRateResultArray(signal_A = _signal_A,
                                   signal_B = _signal_B,
                                   noise_A = _noise_A,
                                   noise_B = _noise_B,
                                   meas_oper = _meas_oper)


def _measurement_rate_solver(pointer_A: QGstate, 
                             pointer_B: QGstate, 
                             meas_oper: QGoper, 
                             meas_mode: int | list[int], 
                             noise_rest: float = 0
                            ):
    """
    Private helper function for measurement_rate, from which it takes its 
    parameters, and to which is returns the components of the measurement rate.

    ---- Parameters ----
    pointer_A : QGstate
        Outout pointer state A, CVS only.
    pointer_B : QGstate
        Outout pointer state B, CVS only.
    meas_oper : QGoper
        The operator to be measured at the output of the system.
    meas_mode : int or list(int)
        Output modes that are monitored during the measurement. To be used in 
        case no meas_oper is specified.
    noise_rest : float
        Added noise from downstream components, default is zero.

    ---- Returns ----
    signal_A : complex
        Projection of pointer state A onto the measurement quadrature.
    signal_B : complex
        Projection of pointer state B onto the measurement quadrature.
    noise_A : float
        Measurement noise contribution from pointer state A along the
        measurement quadrature, not including noise_rest.
    noise_B : float
        Measurement noise contribution from pointer state B along the
        measurement quadrature, not including noise_rest.
    meas_oper : QGoper
        Linear measurement operator used (or found to be optimal) for this
        pair of pointer states.
    """
    # If both pointer states are the same, the measurement signal and hence rate 
    # are always zero. The noise is also set to zero if no measurement operator
    # is provided, since there is no optimum quadrature that may be measured.
    if pointer_A == pointer_B:
        if meas_oper is None:
            _meas_oper = QGoper(data_1st = np.zeros(2*pointer_A.dims_cvs),
                                dims_cvs = pointer_A.dims_cvs)
            _signal_A = 0
            _signal_B = 0
            _noise_A = 0
            _noise_B = 0
        else:
            _meas_oper = meas_oper
            _signal_A = pointer_A.data_1st @ _meas_oper.data_1st
            _signal_B = pointer_B.data_1st @ _meas_oper.data_1st
            _noise_A = (_meas_oper.data_1st 
                        @ pointer_A.data_2nd 
                        @ _meas_oper.data_1st).real
            _noise_B = (_meas_oper.data_1st 
                        @ pointer_B.data_2nd 
                        @ _meas_oper.data_1st).real

    else:
        # Check if a measurement operator has been provided, and if not pick the
        # quadrature which maximizes the difference between both output states.
        if meas_oper is None:
            _meas_oper = _optimum_measurement_operator(pointer_A = pointer_A,
                                                       pointer_B = pointer_B,
                                                       meas_mode = meas_mode)
        else:
            _meas_oper = meas_oper
        
        _signal_A = pointer_A.data_1st @ _meas_oper.data_1st
        _signal_B = pointer_B.data_1st @ _meas_oper.data_1st
        _noise_A = (_meas_oper.data_1st 
                    @ pointer_A.data_2nd 
                    @ _meas_oper.data_1st).real
        _noise_B = (_meas_oper.data_1st 
                    @ pointer_B.data_2nd 
                    @ _meas_oper.data_1st).real

    return _signal_A,_signal_B,_noise_A,_noise_B,_meas_oper


def _optimum_measurement_operator(pointer_A: QGstate, 
                                  pointer_B: QGstate, 
                                  meas_mode: int | list[int] = None
                                 )-> QGoper:
    """
    Private helper function to calculate the optimal output measurement operator 
    to distinguish between two pointer state. The optimal operator is identified 
    as the one which returns the maximum signal component, and hence, maximizes 
    the separation between the two pointer state. In the case of amplified 
    noise, this may not always give the optimal measurement rate.

    ---- Parameters ----
    pointer_A : QGstate
        Outout pointer state A, CVS only.
    pointer_B : QGstate
        Outout pointer state B, CVS only.
    meas_mode : int or list(int)
        Output modes that are monitored during the measurement. To be used in 
        case no meas_oper is specified.

    ---- Returns ----
    meas_oper : QGoper
        Linear measurement operator which maximizes the measurement signal.
    """
    _dims_env = pointer_A.dims_cvs
    # IF no measurement mode(s) provided, assume all environment modes are monitored.
    if meas_mode is None:
        _mode_index = np.ones(2*_dims_env)
    # Else, generate a vector with zeroes in the quadratures of unmonitored 
    # modes, and ones in the position of quadratures of monitored modes.
    else:
        if isinstance(meas_mode, int): 
            meas_mode = [meas_mode]
        _mode_index = np.zeros(2*_dims_env)
        for i in meas_mode: _mode_index[2*i-2:2*i] = 1
    # Solve for the optimal weights, and use to normalize vector get the 
    # optimum measurement operator.
    _weights = (pointer_A.data_1st - pointer_B.data_1st)*_mode_index

    if all([w == 0 for w in np.abs(_weights)]):
        # If all weights are zero, then no operator is optimal. 
        # Return the zero operator and warn the user.
        warnings.warn("No optimal measurement operator found, returning the " \
                      "zero operator. Measurement rate will be ill-defined.")
        _meas_oper = QGoper(dims_cvs = _dims_env)
    else:
        _meas_oper = QGoper(data_1st = _weights / np.sqrt(np.sum(_weights**2)),
                            dims_cvs = _dims_env)
    
    return _meas_oper