import warnings
import numpy.typing as npt
import qgauss
import numpy as np
from scipy.linalg import solve,solve_continuous_lyapunov

from ..core.qgstate import QGstate
from ..core.qgsuper import QGsuper

__all__ = ['moment_steadystate','backaction_steadystate']


def moment_steadystate(L0: QGsuper,
                       elem: tuple[int,int] = None,
                       tol: float = qgauss.settings.atol
                      ) -> QGstate:
    """
    Returns the steady-state density operator for the Liouvillian, L0, or
    for the sub-system superoperator if elem is specified.

    ---- Parameters ----
    L0 : QGsuper
        System Lindbladian/Liouvillian. The Liouvillian need not describe the 
        evolution of a true master equation.
    elem : tuple[int,int]
        Matrix element of FLS density matrix passed as a tuple. If no variable
        is passed the entire steady-state density operator is solved if the
        system has an FLS component, or just the steady-state of the CV system
        if there is no FLS component.
    tol : float
        Set tolerance for the magnitude of real or imaginary parts of numbers. 
        Parts of numbers below this tolerance value are set to zero.

    ---- Returns ----
    rhof : QGstate
        Steady-state intracavity state associated with the input superoperator.
    """
    # --------------------------------------------------------------------------
    # No CVS component is present, raise error.
    if L0.isfls and not L0.iscvs:
        raise ValueError("No CVS component coupled to the FLS component. " \
                         "Steady-state moments cannot be defined.")
    
    # No FLS component is present, solve the CV system.
    elif not L0.isfls and L0.iscvs:
        norm,mean,cov = _moment_steadystate_solver(L0, tol)
        rhof = QGstate(data_2nd = cov,
                       data_1st = mean,
                       data_0th = norm,
                       dims_cvs = L0.dims_cvs)
        
    # --------------------------------------------------------------------------
    # Element of the FLS state is specified, solve the corresponding CV component
    elif L0.isfls and elem is not None:
        _index = np.prod(L0.dims_fls[0][0])*elem[1] + elem[0]

        if not L0.issubgauss(_index):
            warnings.warn("Evolution of the CV system is not Gaussian. " \
            "Terms which violate this assumption will be ignored.")

        norm,mean,cov = _moment_steadystate_solver(L0[_index,_index], tol)
        rhof = QGstate(data_2nd = cov,
                       data_1st = mean,
                       data_0th = norm,
                       dims_cvs = L0.dims_cvs)
        
    # --------------------------------------------------------------------------
    # No element of the FLS state is specified, solve for all components.
    elif L0.isfls and elem is None:
        if not L0.isgauss:
            warnings.warn("Evolution of the CV system is not Gaussian. " \
            "Terms which violate this assumption will be ignored.")
        
        _fls_row = np.prod(L0.dims_fls[0][0])
        _fls_col = np.prod(L0.dims_fls[0][1])
        _cvs_dim = 2*L0.dims_cvs

        cov = np.empty([_fls_row,_fls_col,_cvs_dim,_cvs_dim], dtype=complex)
        mean = np.empty([_fls_row,_fls_col,_cvs_dim], dtype=complex)
        norm = np.empty([_fls_row,_fls_col], dtype=complex)

        for _row in range(0,_fls_row):
            for _col in range(0,_fls_col): 
                _index = np.prod(L0.dims_fls[0][0])*_col + _row
            
                norm[_row,_col],mean[_row,_col],cov[_row,_col] = \
                    _moment_steadystate_solver(L0[_index,_index], tol)
                
        rhof = QGstate(data_2nd = cov,
                       data_1st = mean,
                       data_0th = norm,
                       dims_cvs = L0.dims_cvs,
                       dims_fls = L0.dims_fls[0])
        
    return rhof


def _moment_steadystate_solver(L0: QGsuper,
                               tol: float = qgauss.settings.atol
                               ) -> QGstate:
    """
    Internal steady-state solver for the steady-state a corresponding CVS-only
    systemLiouvillian, L0. The function warn the use if it is found that no
    steady-state solution to L0 exists.

    ---- Parameters ----
    L0 : QGsuper
        System Lindbladian/Liouvillian. The Liouvillian need not describe the 
        evolution of a true master equation. Must be CVS-only, and so contain 
        no FLS component.
    tol : float
        Set tolerance for the magnitude of real or imaginary parts of numbers. 
        Parts of numbers below this tolerance value are set to zero.

    ---- Returns ----
    cov : array[complex]
        Steady-state covariance matrix.
    mean : array[complex]
        Steady-state vector of means.  
    norm : complex
        Steady-state value of the norm, either 0 or 1.  
    """
    # Generate arrays for the moment equations. These are the arrays obtained by
    # mapping the Liouvillian to a partial differential equation in the Wigner
    # phase space of the form, where W(r;t) is the Wigner function
    #    ∂W(r;t)/∂t = (-G - (∂/∂r).F - r.D 
    #                   + ½*(∂/∂r).C.(∂/∂r) - ½*r.B.r - (∂/∂r).A.r) W(r;t).
    _A = L0.wigner_2nd_rdr
    _B = L0.wigner_2nd_rr
    _C = L0.wigner_2nd_drdr
    _D = L0.wigner_1st_r
    _F = L0.wigner_1st_dr

    if (np.all(np.abs(_B) < tol) and np.all(np.abs(_D) < tol)):
        # Solver for the steady-state of a norm-preserving Liouvillian. This 
        # solver converts the input data into the following matrix equations:
        #    0 = Σ.A^T + Σ.V + C
        #    0 = A.μ + F
        # Σ is the covariance matrix, and μ is an array containing the means.
        if np.any(np.real(np.linalg.eigvals(_A)) >= 0):
            warnings.warn("System has no steady-state solution.")

        cov = solve_continuous_lyapunov(_A,-_C)
        mean = solve(_A,-_F)
        norm = 1

    else:
        # Solver for the steady-state of a non-norm-preserving Liouvillian. This 
        # solver converts the input data into the following non-linear matrix 
        # equations:
        #    0 = A.Σ + Σ.A^T - Σ.B.Σ + C
        #    0 = (A - B.Σ).μ + F - Σ.D
        # Σ is the covariance matrix, and μ is an array containing the means. 
        # For the covariance matrix, the continuous algebraic Riccati equation
        # (CARE) is solved using the stable-subspace solution method. In this
        # method, we solve the linearized CARE:
        #   0 = H.[U \\ V]  where  H = [[A^T,-B],[-C,-A]].
        # The matrix [U \\ V] is constructed from the stable eigenvectors of H,
        # after which we can use Σ = V.U^-1.
        
        _H = np.block([[_A.T, -_B], [-_C, -_A]])
        _evals, _evecs = np.linalg.eig(_H)
        _indices_stable = np.where(np.real(_evals) < 0)[0]

        if len(_indices_stable) != 2*L0.dims_cvs:
            warnings.warn("Covariance matrix has no steady-state solution.")

        _soln = _evecs[:,_indices_stable]                           
        cov = (_soln[2*L0.dims_cvs:4*L0.dims_cvs,0:2*L0.dims_cvs] 
                @ np.linalg.inv(_soln[0:2*L0.dims_cvs,0:2*L0.dims_cvs]))
        
        if np.any(np.real(np.linalg.eigvals(cov)) < 0):
            warnings.warn("Solution for the covariance matrix is not positive-definite. " \
                          "Wigner function is not integrable.")
        if np.any(np.real(np.linalg.eigvals(_A - cov @ _B)) >= 0):
            warnings.warn("Vector of means has no steady-state solution.")

        mean = solve(_A - cov @ _B,-_F + cov @ _D)
        norm = 0

    return norm,mean,cov


def backaction_steadystate(L0: QGsuper,
                           elem: tuple[int,int] = None,
                           tol: float = qgauss.settings.atol
                          ) -> tuple[complex,complex,complex,complex] | \
                           tuple[npt.NDArray[complex],npt.NDArray[complex],
                                 npt.NDArray[complex],npt.NDArray[complex]] :
    """
    ---- Procedure ----
    Steady-state solver for the backaction rates of a CV system on a 
    finite-level system. The function will automatically check if the evolution 
    is Gaussian, and will raise an error if not. The function will also check if 
    a steady-state exists and will exit if not. The component of the FLS state 
    can be specified if the system has an FLS component, and if none is given, 
    the backaction on every component will be solved. If the system is only CV, 
    no FLS state need be specified. The function will return the total dephasing 
    and frequency shift, along with the bare, measurement-induced, and parasitic 
    dephasing subcomponents. 

    ---- Parameters ----
    L0 : QGsuper
        System Lindbladian/Liouvillian. The Liouvillian need not describe the 
        evolution of a true master equation.
    elem : tuple[int,int]
        Matrix element of FLS density matrix passed as a tuple. If no variable
        is passed the backaction on all elements are solved if the system has an 
        FLS component, or just the steady-state of the CV system if there is 
        no FLS component.
    tol : float
        Set tolerance for the magnitude of real or imaginary parts of numbers. 
        Parts of numbers below this tolerance value are set to zero.

    ---- Returns ----
    ba_total : complex or array[complex]
        Steady-state total dephasing and frequency shift.
    ba_bare : complex or array[complex]
        Steady-state parasitic dephasing and frequency shift. Includes 
        components from the innate FLS dynamics as well backation from the 
        CVS state which are independent of the first and second moments.
    ba_meas_ind : complex or array[complex]
        Steady-state measurement induced dephasing and frequency shift. This is 
        the component dependent on the displacement of the continuous variable 
        system.
    ba_para : complex or array[complex]
        Steady-state parasitic dephasing and frequency shift. The dephasing part 
        of this components arises when the CVS state has variances above vacuum, 
        while the frequency shift is present even for the vacuum shift.
    """
    # --------------------------------------------------------------------------
    # No CVS component is present, raise error.
    if L0.isfls and not L0.iscvs:
        raise ValueError("No CVS component coupled to the FLS component. " \
                         "Bakaction from the CVS component cannot be defined.")
    
    # No FLS component is present, solve the CV system.
    elif not L0.isfls and L0.iscvs:
        ba_total,ba_bare,ba_meas_ind,ba_para = \
            _backaction_steadystate_solver(L0, tol)
        
    # --------------------------------------------------------------------------
    # Element of the FLS state is specified, solve the corresponding CV component
    elif L0.isfls and elem is not None:
        _index = np.prod(L0.dims_fls[0][0])*elem[1] + elem[0]

        if not L0.issubgauss(_index):
            warnings.warn("Evolution of the CV system is not Gaussian. " \
            "Terms which violate this assumption will be ignored.")

        ba_total,ba_bare,ba_meas_ind,ba_para = \
            _backaction_steadystate_solver(L0[_index,_index], tol)
        
    # --------------------------------------------------------------------------
    # No element of the FLS state is specified, solve for all components.
    elif L0.isfls and elem is None:
        if not L0.isgauss:
            warnings.warn("Evolution of the CV system is not Gaussian. " \
            "Terms which violate this assumption will be ignored.")
        
        _fls_row = np.prod(L0.dims_fls[0][0])
        _fls_col = np.prod(L0.dims_fls[0][1])

        ba_total = np.empty([_fls_row,_fls_col], dtype=complex)
        ba_bare = np.empty([_fls_row,_fls_col], dtype=complex)
        ba_meas_ind = np.empty([_fls_row,_fls_col], dtype=complex)
        ba_para = np.empty([_fls_row,_fls_col], dtype=complex)

        for _row in range(0,_fls_row):
            for _col in range(0,_fls_col): 
                _index = np.prod(L0.dims_fls[0][0])*_col + _row
            
                (ba_total[_row,_col], ba_bare[_row,_col], 
                 ba_meas_ind[_row,_col], ba_para[_row,_col]) = \
                    _backaction_steadystate_solver(L0[_index,_index], tol)

    return ba_total,ba_bare,ba_meas_ind,ba_para


def _backaction_steadystate_solver(L0: QGsuper,
                                   tol: float
                                  ) -> tuple[complex,complex,complex,complex]:
    """
    The backaction on an operator ρ is defined as tr[ρ] = exp[-v]. The 
    backaction rate is then extracted from dv/dt, defined by
        dv/dt = G + μ.D + (1/2)*μ.B.μ + (1/2)*trace[B.Σ]
    where Σ and μ are the 2nd and 1st central moments of ρ, respectively. The 
    elements B,D,G are extracted from the L0 Liouvillian with governs the 
    dynamics of ρ. The dephasing rate and frequency shift correspond to the real 
    and imaginary parts of 'v', respectively. The backaction may be broken into 
    bare/innate 'ba_bare', measurement induced 'ba_meas_ind', and parasitic 
    'ba_para', components:
        bare = G
        meas_ind = μ.D + (1/2)*μ.B.μ
        para = (1/2)*trace[B.Σ]

    ---- Parameters ----
    L0 : QGsuper
        System QGsuper whose steady-state is to be solved, and the backaction 
        extracted. Must be CVS-only.
    tol : float
        Set tolerance for the magnitude of real or imaginary parts of numbers. 
        Parts of numbers below this tolerance value are set to zero.
    
    ---- Returns ----
    ba_total : complex
    ba_bare : complex
    ba_meas_ind : complex
    ba_para : complex
    """
    # Solve the steady-state of the system evolving under the QGsuper L0.
    _data = _moment_steadystate_solver(L0, tol)

    # Generate arrays and calculate the components of the backaction.
    _B = L0.wigner_2nd_rr
    _D = L0.wigner_1st_r
    _G = L0.wigner_0th[0]

    ba_bare = _G
    ba_meas_ind = _data[1] @ _D + (1/2)*_data[1] @ _B @ _data[1]
    ba_para = (1/2)*np.trace(_B @ _data[2])
    ba_total = ba_bare + ba_meas_ind + ba_para

    return ba_total,ba_bare,ba_meas_ind,ba_para