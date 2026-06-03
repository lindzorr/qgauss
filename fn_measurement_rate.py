import sys
import warnings
import numpy.typing as npt
import qgauss
import numpy as np

from .qgstate import QGstate
from .qgoper import QGoper
from .qghle import QGhle
from .fn_utilities import *

__all__ = ['measurement_rate']


def measurement_rate(HLE: QGhle,
                     pointers: tuple[int,int] = None, 
                     meas_oper: QGoper = None, 
                     meas_mode: int | list[int] = None, 
                     noise_rest: float = 0, 
                     freq: float = 0, 
                    ):
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
    noise_rest : float
        Added noise from downstream components, default is zero.
    freq : float
        Frequency at which the measurement is performed, default is zero.

    ---- Returns ----
    meas_rate : float or array
        Measurement rate for the pair(s) of pointer states.
    meas_signal : float or array
        Measurement signal, that is, displacement between pointer states along 
        the quadrature defined by the measurement operator.
    meas_noise : float or array
        Measurement noise along with quadrature defined by the measurement 
        operator, not including noise_rest.
    """
    # ----------------------------------------------------------------------   
    # Check if there is an FLS component is present, and raise error if not.
    if not HLE.isfls:
        raise ValueError("No FLS coupled to system. Measurement rate cannot be defined.")

    # ----------------------------------------------------------------------
    # FLS pointer states are specified, solve the corresponding 
    # measurement rate.
    elif HLE.isfls and pointers is not None:
        # Generate the output states of the system
        _pointer_A = HLE.output_env(freq = freq, index = pointers[0])
        _pointer_B = HLE.output_env(freq = freq, index = pointers[1])

        (meas_rate, meas_signal, meas_noise) = \
            _measurement_rate_solver(pointer_A = _pointer_A, 
                                     pointer_B = _pointer_B, 
                                     meas_oper = meas_oper, 
                                     meas_mode = meas_mode, 
                                     noise_rest = noise_rest)
        
    # ----------------------------------------------------------------------
    # No FLS pointer states is specified, solve for all measurment rates
    elif HLE.isfls and pointers is None:
        _row_total = np.prod(HLE.dims_fls[0])
        _col_total = np.prod(HLE.dims_fls[1])

        meas_signal = np.empty([_row_total,_col_total])
        meas_noise = np.empty([_row_total,_col_total])
        meas_rate = np.empty([_row_total,_col_total])

        for _row in range(0,_row_total):
            for _col in range(0,_col_total):
                _pointer_A = HLE.output_env(freq = freq, index = _row)
                _pointer_B = HLE.output_env(freq = freq, index = _col)

                (meas_rate[_row,_col], meas_signal[_row,_col], meas_noise[_row,_col]) = \
                    _measurement_rate_solver(pointer_A = _pointer_A,
                                             pointer_B = _pointer_B, 
                                             meas_oper = meas_oper, 
                                             meas_mode = meas_mode, 
                                             noise_rest = noise_rest)

    return meas_rate,meas_signal,meas_noise


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
    meas_rate : float
        Measurement rate of the pair of pointer stated at the given frequency.
    meas_signal : float
        Measurement signal, that is, displacement between pointer states.
    meas_noise : float
        Measurement noise along with quadrature defined by the measurement 
        operator, not including noise_rest.
    """
    # If both pointer states are the same, the measurement signal and hence rate 
    # are always zero. The noise is also set to zero if no measurement operator
    # is provided, since there is no optimum quadrature that may be measured.
    if pointer_A == pointer_B:
        meas_signal = 0
        meas_rate = 0
        if meas_oper is None:
            meas_noise = 0
        else:
            _noise_A = meas_oper.data_1st @ pointer_A.data_2nd @ meas_oper.data_1st
            _noise_B = meas_oper.data_1st @ pointer_B.data_2nd @ meas_oper.data_1st
            meas_noise = _noise_A.real + _noise_B.real

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
        _noise_A = _meas_oper.data_1st @ pointer_A.data_2nd @ _meas_oper.data_1st
        _noise_B = _meas_oper.data_1st @ pointer_B.data_2nd @ _meas_oper.data_1st

        meas_signal = (1/4)*np.abs(_signal_A - _signal_B)**2
        meas_noise = _noise_A.real + _noise_B.real
        meas_rate = meas_signal/(meas_noise + 2*noise_rest)

    return meas_rate,meas_signal,meas_noise


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
        # Return the zero operator and warn the use.
        warnings.warn("No optimal measurement operator found, returning the " \
                      "zero operator. Measurement rate will be ill-defined.")
        meas_oper = QGoper(dims_cvs = _dims_env)
    else:
        meas_oper = QGoper(data_1st = _weights / np.sqrt(np.sum(_weights**2)),
                           dims_cvs = _dims_env)
    
    return meas_oper