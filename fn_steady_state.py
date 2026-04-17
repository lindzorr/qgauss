import sys
import warnings
import types
import typing
import numbers
import numpy.typing as npt

import qgauss
import numpy as np
from scipy import linalg as la

from .qgstate import QGstate
from .qgsuper import QGsuper

__all__ = ['backaction_rate_steadystate','moment_solver_steadystate']


def backaction_rate_steadystate(LV: QGsuper, 
                                qubit: str = None, 
                                tol: float = qgauss.settings.atol
                                ):
    """
    ---- Procedure ----
    Steady-state solver for the backaction rates of a CV system on a qubit, or a CV system coupled to
    a system of qubits. The function will automatically check if the evolution is Gaussian,
    and will raise an error if not. The function will also check if a steady-state exists and will exit
    if not. The component of the qubit state can be specified if the system has an FLS component, and if
    none is given, the backaction on every component will be solved. If the system is only CV, no qubit 
    state need be specified. The function will return the total dephasing and frequency shift, along with 
    the bare, measurement-induced, and parasitic dephasing subcomponents. 

    ---- Parameters ----
    LV : QGsuper
        System Lindbladian/Liouvillian. The Liouvillian need not describe the evolution of a true master equation.
    qubit : string
        Element of qubit density matrix to solve passed as string, for example, 'e,e' or 'g,e' 
        for a single qubit, 'eg,ee' or 'gg,ge' etc, for a two qubit system. Scales to arbitrary number
        of qubits. If no string is passed the backaction on all elements are solved if the 
        system has an FLS component, or just the steady-state of the CV system if there is not FLS.
        Alternatively, one can use the '1' and '0' in place of 'e' and 'g', respectively.
    tol : float
        Set tolerance for the magnitude of real or imaginary parts of numbers. Parts of numbers below this
        tolerance value are set to zero.

    ---- Returns ----
    ba_total : complex or array
        Steady-state total dephasing and frequency shift.
    ba_bare : complex or array
        Steady-state parasitic dephasing and frequency shift. Includes components from the innate qubit dynamics
        as well backation from the CVS state which are independent of the first and second moments.
    ba_meas_ind : complex or array
        Steady-state measurement induced dephasing and frequency shift. This is the component dependent
        on the displacement of the continuous variable system.
    ba_para : complex or array
        Steady-state parasitic dephasing and frequency shift. The dephasing part of this components arises when the 
        CVS state has variances above vacuum, while the frequency shift is present even for the vacuum shift.
    """

    # ----------------------------------------------------------------------
    # No qubits are present, solve the CV system
    if not LV.isfls and LV.iscvs:
        ba_total,ba_bare,ba_meas_ind,ba_para = _backaction_steadystate_solver(LV, tol)

    # ----------------------------------------------------------------------
    # Qubit state is specified, solve the corresponding CV component
    elif LV.isfls and qubit is not None:
        # Select index for the qubit state
        qbinary = qubit.replace('e', '1').replace('g', '0')
        qrow = int(qbinary.split(',')[0], 2)
        qcol = int(qbinary.split(',')[1], 2)
        # Index count is backwards, so 'e...e' is the 0th element and 'g...g' is last
        # To get correct index q, substract the expected index from the total length
        qindex = np.prod(LV.dims_fls[0]) - np.prod(LV.dims_fls[0][0])*qcol - qrow - 1

        # Check if spin-z Pauli operators are QND-observables, and the dynamics Gaussian
        if LV.issubqauss(qindex):
            pass
        else:
            sys.exit("Equation of motion for the CV system is not Gaussian. The moment method cannot be used.")

        ba_total,ba_bare,ba_meas_ind,ba_para = _backaction_steadystate_solver(LV[qindex,qindex], tol)

    # ----------------------------------------------------------------------
    # No qubit state is specified, solve for all components
    elif LV.isfls and qubit is None:
        # Check if spin-z Pauli operators are QND-oberservables, and the dynamics Gaussian
        if LV.isqauss:
            pass
        else:
            sys.exit("Equation of motion for the CV system is not Gaussian. The moment method cannot be used.")
        
        row_total = np.prod(LV.dims_fls[0][0])
        col_total = np.prod(LV.dims_fls[0][1])

        ba_total = np.empty([row_total,col_total], dtype = complex)
        ba_bare = np.empty([row_total,col_total], dtype = complex)
        ba_meas_ind = np.empty([row_total,col_total], dtype = complex)
        ba_para = np.empty([row_total,col_total], dtype = complex)

        for qrow in range(0,row_total):
            for qcol in range(0,col_total):
                qindex = np.prod(LV.dims_fls[0]) - np.prod(LV.dims_fls[0][1])*qcol - qrow - 1
                (ba_total[qrow,qcol], ba_bare[qrow,qcol], ba_meas_ind[qrow,qcol], ba_para[qrow,qcol]) = \
                    _backaction_steadystate_solver(LV[qindex,qindex], tol)

    return ba_total,ba_bare,ba_meas_ind,ba_para


def _backaction_steadystate_solver(LV: QGsuper,
                                   tol: float = qgauss.settings.atol
                                   ):
    """
    The backaction on an operator ρ is defined as tr[ρ] = exp[-v]. The backaction rate is then extracted from dv/dt,
    defined by
        dv/dt = G + μ.D + (1/2)*μ.B.μ + (1/2)*trace[B.Σ]
    where Σ and μ are the 2nd and 1st central moments of ρ, respectively. The elements B,D,G are extracted from the 
    'input' Liouvillian with governs the dynamics of ρ. The dephasing rate and frequency shift correspond to the real 
    and imaginary parts of 'v', respectively. The backaction may be broken into bare/innate 'ba_bare', measurement 
    induced 'ba_meas_ind', and parasitic 'ba_para', components:
        bare = G
        meas_ind = μ.D + (1/2)*μ.B.μ
        para = (1/2)*trace[B.Σ]

    ---- Parameters ----
    LV : QGsuper
        System QGsuper whose steady-state is to be solved, and the backaction extracted. Must be CVS-only.
    tol : float
        Set tolerance for the magnitude of real or imaginary parts of numbers. Parts of numbers below this
        tolerance value are set to zero.
    
    ---- Returns ----
    ba_total : complex
    ba_bare : complex
    ba_meas_ind : complex
    ba_para : complex
    """
    # Solve the steady-state components of the system evolving under the 'input' QGsuper
    steady_state = moment_solver_steadystate(LV,tol)
    cov = steady_state.data_2nd
    mean = steady_state.data_1st

    # Generate arrays from the QGsuper input
    B = input.wigner_2nd_var
    D = input.wigner_1st_var
    G = input.wigner_0th[0]

    # Calculate the parasitic and measurement induced parts of the backaction
    ba_bare = G
    ba_meas_ind = np.dot(mean,D) + 0.5*np.einsum("j,jk,k",mean,B,mean)
    ba_para = 0.5*np.trace(B@cov)
    ba_total = ba_bare + ba_meas_ind + ba_para

    return ba_total,ba_bare,ba_meas_ind,ba_para


def moment_solver_steadystate(LV: QGsuper, 
                              tol: float = qgauss.settings.atol
                              ) -> QGstate:
    """
    Steady-state solver for the steady-state for a corresponding CV system Liouvillian, L0. The function will exit if
    it is found that no steady-state solution to L0 exists.

    ---- Parameters ----
    LV : QGsuper
        System Lindbladian/Liouvillian. The Liouvillian need not describe the evolution of a true master equation.
        Must be CVS-only, and so contain no FLS component.
    tol : float
        Set tolerance for the magnitude of real or imaginary parts of numbers. Parts of numbers below this
        tolerance value are set to zero.

    ---- Returns ----
    output : QGstate
        Steady-state intracavity state associated with the input superoperator.
    """

    # Generate arrays
    """
    Generate arrays for the moment equations. These are the arrays obtained by
    mapping the Liouvillian to a partial differential equation in the Wigner
    phase space of the form, where W(r;t) is the Wigner function
        ∂W(r;t)/∂t = (-G - (∂/∂r).F - r.D + ½*(∂/∂r).C.(∂/∂r) - ½*r.B.r - (∂/∂r).A.r) W(r;t)
    """
    dims = LV.dims_cvs
    A = LV.wigner_2nd_deriv_var
    B = LV.wigner_2nd_var
    C = LV.wigner_2nd_deriv
    D = LV.wigner_1st_var
    F = LV.wigner_1st_deriv

    # Check if system is stable before proceeding
    if not all(np.real(x) < 0 for x in la.eigvals(A)):
        sys.exit("System has eigenvalues with positive real part "
                 +"and is therefore unstable: no steady state solution exists.")

    # Solve steady-state moment equations generated by the arrays
    if (np.all(np.abs(B) < tol) and np.all(np.abs(D) < tol)):
        """
        Solver for the steady-state of a norm-preserving Liouvillian. This solver converts the input data into the 
        following matrix equations:
            0 = Σ.A^T + Σ.V + C
            0 = A.μ + F
        Σ is the covariance matrix, and μ is an array containing the means. A Scipy function is used here to solve 
        the Lyapunov equation for the covariance matrix.
        """
        # Solve covariance matrix
        cov = la.solve_continuous_lyapunov(A,-C)
        # Solve means
        mean = la.solve(A,-F)

    else:
        """
        Solver for the steady-state of a non-norm-preserving Liouvillian. This solver converts the input data into the 
        following non-linear matrix equations:
            0 = A.Σ + Σ.A^T - Σ.B.Σ + C
            0 = (A - B.Σ).μ + F - Σ.D
        Σ is the covariance matrix, and μ is an array containing the means. For the covariance matrix, the continuous 
        algebraic Riccati equation (CARE) is solved using the stable-subspace solution method.
        """
        # Solve covariance matrix
        H = np.block([[np.transpose(A), -B], [-C, -A]]) # "Hamiltonian" matrix, H-matrix
        evals, evecs = np.linalg.eig(H)                 # Eigendecomposition of H-matrix
        stbl = np.where(np.real(evals) < 0)[0]          # Vector of stable eigenvalues
        if len(stbl) != 2*dims:
            # Terminate program if dimensions of stable subspace are too small or large
            sys.exit("No stable solution can be found for the covariance matrix.")
        soln = evecs[:,stbl]                           # Array of stable eigenvectors
        cov = soln[2*dims:4*dims,0:2*dims]@np.linalg.inv(soln[0:2*dims,0:2*dims])
        # Solve means
        mean = la.solve(A - cov@B,-F + cov@D)

    return QGstate(data_2nd = cov,
                   data_1st = mean,
                   dims_cvs = dims)