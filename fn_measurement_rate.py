import sys
import warnings
import types
import typing
import numbers
import numpy.typing as npt

import qgauss
import numpy as np
from scipy import linalg as la

from .qgstate import *
from .qgoper import *
from .qgsuper import *
from .fn_constructor import *
from .fn_superoperator import *
from .fn_utilities import *

__all__ = ['measurement_rate','output_state']


def measurement_rate(H_system: QGoper, 
                     H_system_bath: QGoper, 
                     input_state: QGstate, 
                     pointers: str = None, 
                     meas_oper: QGoper = None, 
                     meas_mode: int | list[int] = None, 
                     noise_rest: float = 0, 
                     freq: float = 0, 
                     tol: float = qgauss.settings.atol
                     ):
    """
    ---- Procedure ----
    Routine to calculate the steady-state measurement rate of some number of qubits, defined in terms of the SNR as:
        measurement_rate = lim_t→∞ SNR^2(t)/t.
    This function follows the theory developed in the following paper to describe the system-bath coupling and 
    input-ouput theory:
        Gardiner & Collett, Phys. Rev. A 31, 3761 (1985).
    For the multimode case, supporting expressions can be found in Appendices A3 and A4 in Miller and Orr et al,
    arXiv:2603.12312 (2026). To summarize, we represent the system quadratures as
        r_sys = (q1,p1,...,qN,pN),
    while the bath quadratures are implicitly frequency dependent and are expressed as
        r_bath = (q1(ω),p1(ω),...,qM(ω),pM(ω)).
    In representing the system-bath Hamiltonian, the corresponding form, following Gardiner and Collett, will be
        (1/√2π)*Integral[ (r_sys,r_bath)^T.H_system_bath.(r_sys,r_bath) dω],
    where "H_system_bath" is the matrix of coefficients for the bilinear system-bath Hamiltonian. When constructing this
    Hamiltonian, omit the integral and instead construct the QGoper using
        (r_sys,r_bath)^T.H_system_bath.(r_sys,r_bath)
    where the bath modes must come after the system modes in the tensor product. The data in the system-bath Hamiltonian
    will have the form H_system_bath.data_2nd = [[Hsb,0],[0,Hsb]]. If we represent the input states, ρ_in, as a QGstate,
    then the corresponding dissipation component in the Lindbladian will be:
        Γ = Hsb@(ρ_in.data_2nd + iΩ_2M/2)@Hsb^T where Ω_2M is the 2Mx2M symplectic form.
    This deomnstrates why this function does not use the QGsuper as its input, since Hsb is required to formulate the
    input-ouput theory for the measurement rate, but knowledge of Γ and ρ_in is insufficient to uniquqly determine the
    system-bath coupling Hsb.

    ---- Parameters ----
    H_system : QGoper
        System Hamiltonian. This is a linear operator on the Hilbert space of the system modes and qubit(s) only.
    H_system_bath : QGoper
        System-bath Hamiltonian. This operator acts on the Hilbert space of both the system and bath modes but not the 
        qubit(s). The system modes must be first in the tensor product, followed by the bath modes.
    input_state : QGstate
        Input state of the system. Must be specified to use input-output relations.
    pointers : string
        The measurement rate is to be calculated between the pointer states for these two elements of the qubit density
        matrix. This is to be passed as string. For a single qubit example, 'g,e' represents the measurement rate 
        between |g> and |e>, and will be the same as 'e,g'. For multiple qubits, multiple measurement rates may be
        defined between each pair of pointer states. For a two qubit system, examples include 'eg,ee' or 'gg,ge'. The
        measurement rate will be zero if both pointer states are the same. Scales to arbitrary number of qubits. If no 
        string is passed, the measurement rate between all pairs are solved if the including, including redundant pairs.
        Alternatively, one can use the '1' and '0' in place of 'e' and 'g', respectively.
    meas_oper : QGoper
        The operator to be measured at the output of the system. This operator acts only on the Hilbert space of the
        bath modes and must be linear.
    meas_mode : int or list(int)
        Output modes that are monitored during the measurement. To be used in case no meas_oper is specified. If none 
        are given, then it is assumed that all are monitored. Measurement operator is constructed using weigts from 
        Bultink et al, Appl. Phys. Lett. 112, 092601 (2018).
    noise_rest : float
        Added noise from downstream components, default is zero.
    freq : float
        Frequency at which the measurement is performed, default is zero.
    tol : float
        Set tolerance for the magnitude of real or imaginary parts of numbers. Parts of numbers below this
        tolerance value are set to zero.

    ---- Returns ----
    meas_rate : float or array
        Measurement rate for the pair(s) of pointer states at the given frequency.
    meas_signal : float or array
        Measurement signal, that is, displacement between pointer states along the quadrature defined by
        the measurement operator.
    meas_noise : float or array
        Measurement noise along with quadrature defined by the measurement operator, not including noise_rest.
    """
    # ----------------------------------------------------------------------
    # System-bath coupling cannot currently handle coupling to any qubits
    if H_system_bath.isfls:
        sys.exit("Qubit(s) dissipation cannot currently be handled by this function.")
    
    # No qubits are present, exit immediately
    if not H_system.isfls:
        sys.exit("No qubit(s) coupled to system. Measurement rate cannot be defined.")

    # ----------------------------------------------------------------------
    # Qubit pointer states are specified, solve the corresponding measurement rate
    elif H_system.isfls and pointers is not None:
        # Select index for the pointer state
        qbinary = pointers.replace('e', '1').replace('g', '0')
        q_A = int(qbinary.split(',')[0], 2)
        q_B = int(qbinary.split(',')[1], 2)

        # Generate the output states of the system
        pointer_A = output_state(H_system[q_A,q_A], H_system_bath, input_state, freq, tol)
        pointer_B = output_state(H_system[q_B,q_B], H_system_bath, input_state, freq, tol)

        (meas_rate, meas_signal, meas_noise) = \
            _measurement_rate_solver(pointer_A = pointer_A, 
                                     pointer_B = pointer_B, 
                                     meas_oper = meas_oper, 
                                     meas_mode = meas_mode, 
                                     noise_rest = noise_rest)
        
    # ----------------------------------------------------------------------
    # No qubit pointer states is specified, solve for all measurment rates
    elif H_system.isfls and pointers is None:
        row_total = np.prod(H_system.dims_fls[0])
        col_total = np.prod(H_system.dims_fls[1])

        meas_signal = np.empty([row_total,col_total])
        meas_noise = np.empty([row_total,col_total])
        meas_rate = np.empty([row_total,col_total])

        for q_A in range(0,row_total):
            for q_B in range(0,col_total):
                pointer_A = output_state(H_system[q_A,q_A], H_system_bath, input_state, freq, tol)
                pointer_B = output_state(H_system[q_B,q_B], H_system_bath, input_state, freq, tol)

                (meas_rate[q_A,q_B], meas_signal[q_A,q_B], meas_noise[q_A,q_B]) = \
                    _measurement_rate_solver(pointer_A = pointer_A,
                                             pointer_B = pointer_B, 
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
    Private helper function for measurement_rate, from which it takes its parameters, and to which is returns
    the components of the measurement rate.

    ---- Parameters ----
    pointer_A : QGstate
        Outout pointer state A, CVS only.
    pointer_B : QGstate
        Outout pointer state B, CVS only.
    meas_oper : QGoper
        The operator to be measured at the output of the system.
    meas_mode : int or list(int)
        Output modes that are monitored during the measurement. To be used in case no meas_oper is specified.
    noise_rest : float
        Added noise from downstream components, default is zero.

    ---- Returns ----
    meas_rate : float
        Measurement rate of the pair of pointer stated at the given frequency.
    meas_signal : float
        Measurement signal, that is, displacement between pointer states.
    meas_noise : float
        Measurement noise along with quadrature defined by the measurement operator, not including noise_rest.
    """
    # If both pointer states are the same, the measurement signal and hence rate are always zero. The noise is also set 
    # to zero if no measurement operator is provided, since there is no optimum quadrature that may be measured.
    if pointer_A == pointer_B:
        meas_signal = 0.
        meas_rate = 0.
        if meas_oper == None:
            meas_noise = 0.
        else:
            meas_noise = np.real((meas_oper.data_1st@pointer_A.data_2nd@meas_oper.data_1st 
                                  + meas_oper.data_1st@pointer_B.data_2nd@meas_oper.data_1st))

    else:
        # Check if a measurement operator has been provided, and if not pick the quadrature operator which 
        # maximizes the difference between the displacement of both output states
        if meas_oper == None:
            meas_oper = _optimum_measurement_operator(pointer_A = pointer_A, 
                                                      pointer_B = pointer_B, 
                                                      meas_mode = meas_mode)

        meas_signal = 0.25*np.abs(pointer_A.data_1st@meas_oper.data_1st 
                                  - pointer_B.data_1st@meas_oper.data_1st)**2
        meas_noise = np.real((meas_oper.data_1st@pointer_A.data_2nd@meas_oper.data_1st 
                              + meas_oper.data_1st@pointer_B.data_2nd@meas_oper.data_1st))
        meas_rate = meas_signal/(meas_noise + 2*noise_rest)

    return meas_rate,meas_signal,meas_noise


def _optimum_measurement_operator(pointer_A: QGstate, 
                                  pointer_B: QGstate, 
                                  meas_mode: int | list[int] = None
                                  )-> QGoper:
    """
    Private helper function to calculate the optimal output measurement operator to distinguish between two pointer 
    state. The optimal operator is identified as the one which returns the maximum signal component, and hence, 
    maximizes the separation between the two pointer state. In the case of amplified noise, this may not always give 
    the optimal measurement rate.

    ---- Parameters ----
    pointer_A : QGstate
        Outout pointer state A, CVS only.
    pointer_B : QGstate
        Outout pointer state B, CVS only.
    meas_mode : int or list(int)
        Output modes that are monitored during the measurement. To be used in case no meas_oper is specified.

    ---- Returns ----
    meas_oper : QGoper
        Linear measurement operator which maximizes the measurement signal.
    """
    dims_bath = pointer_A.dims_cvs
    # IF no measurement mode(s) provided, assume all bath modes are monitored.
    if meas_mode == None:
        mode_index = np.ones(2*dims_bath)
    # Else, generate a vector with zeroes in the quadratures of unmonitored modes,
    # and ones in the position of quadratures of monitored modes.
    else:
        if isinstance(meas_mode, int): 
            meas_mode = [meas_mode]
        mode_index = np.zeros(2*dims_bath)
        for i in meas_mode: mode_index[2*i-2:2*i] = 1
    # Solve for the optimal weights, and use to normalize vector get the optimum measurement operator.
    weights = (pointer_A.data_1st - pointer_B.data_1st)*mode_index

    if all([w == 0 for w in np.abs(weights)]):
        # If all weights are zero, then no operator is optimal and return the zero operator
        meas_oper = QGoper(dims_cvs = dims_bath)
    else:
        meas_oper = QGoper(data_1st = weights/np.sqrt(np.sum(weights**2)),
                           dims_cvs = dims_bath)
    
    return meas_oper


def output_state(H_system: QGoper, 
                 H_system_bath: QGoper, 
                 input_state: QGstate, 
                 freq: float = 0, 
                 tol: float = qgauss.settings.atol
                 ) -> QGstate:
    """
    Routine to calculate the output state of the bath in frequency space. Follows the quantum input-ouput theory of
    Gardiner & Collett.

    ---- Parameters ----
    H_system : QGoper
        System Hamiltonian. This is a linear operator on the Hilbert space of the system modes and qubit only.
    H_system_bath : QGoper
        System-bath Hamiltonian. This operator acts on the Hilbert space of both the system and bath modes but not the 
        qubit. The system modes must be first in the tensor product, followed by the bath modes.
    input_state : QGstate
        Input state of the system. Must be specified to use input-output relations.
    freq : float
        Frequency, default is zero.
    tol : float
        Set tolerance for the magnitude of real or imaginary parts of numbers. Parts of numbers below this
        tolerance value are set to zero.

    ---- Returns ----
    output_state : QGstate
        Output state at specified frequency. Note, displacement may be complex-valued, and hence this may not 
        correspond to a real state.
    """
    # Set dimensions of the system (dims_sys), and the number of environments (dims_bath)
    dims_sys = H_system.dims_cvs
    dims_bath = input_state.dims_cvs

    Hsb = H_system_bath.data_2nd[0:2*dims_sys,2*dims_sys:2*dims_sys+2*dims_bath]
    symform_sys = symplectic_form(dims_sys)
    symform_bath = symplectic_form(dims_bath)
    A = symform_sys@H_system.data_2nd + 0.5*symform_sys@Hsb@symform_bath@np.transpose(Hsb)

    s_mat_freq = (-symform_bath@np.transpose(Hsb)@np.linalg.inv(A + 1j*freq*np.identity(2*dims_sys))@symform_sys@Hsb
                   + np.identity(2*dims_bath))
    s_mat_neg_freq = (-symform_bath@np.transpose(Hsb)@np.linalg.inv(A - 1j*freq*np.identity(2*dims_sys))@symform_sys@Hsb
                       + np.identity(2*dims_bath))
    t_mat_freq = -symform_bath@np.transpose(Hsb)@np.linalg.inv(A + 1j*freq*np.identity(2*dims_sys))

    output_mean = s_mat_freq@input_state.data_1st + t_mat_freq@symform_sys@H_system.data_1st
    output_covariance = 0.5*(s_mat_freq@input_state.data_2nd@np.transpose(s_mat_neg_freq)
                             + s_mat_neg_freq@input_state.data_2nd@np.transpose(s_mat_freq)
                             )

    return QGstate(data_2nd = output_covariance,
                   data_1st = output_mean,
                   dims_cvs = dims_bath)