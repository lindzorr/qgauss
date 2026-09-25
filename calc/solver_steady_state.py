from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import NamedTuple
import numbers
import numpy.typing as npt
import qgauss
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve,solve_continuous_lyapunov

from ..core.qgstate import QGstate
from ..core.qgsuper import QGsuper

__all__ = ['moment_steadystate','backaction_steadystate',
           'BackactionResult','BackactionResultArray']


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
    # No CVS or FLS component is present, raise error.
    if not L0.isfls and not L0.iscvs:
        raise ValueError("No CVS or FLS component. " \
                         "Steady-state moments cannot be defined.")
    
    # No CVS component is present, raise error.
    elif L0.isfls and not L0.iscvs:
        raise ValueError("No CVS component coupled to the FLS component. " \
                         "Steady-state moments cannot be defined.")
    
    # No FLS component is present, solve the CV system.
    elif not L0.isfls and L0.iscvs:
        norm,mean,cov = _moment_steadystate_solver(L0, tol=tol)
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

        norm,mean,cov = _moment_steadystate_solver(L0[_index,_index], tol=tol)
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
                    _moment_steadystate_solver(L0[_index,_index], tol=tol)
                
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
    system Liouvillian, L0. The function warn the use if it is found that no
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
    norm : complex
        Steady-state value of the norm, either 0 or 1. 
    mean : array[complex]
        Steady-state vector of means.  
    cov : array[complex]
        Steady-state covariance matrix.
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
        #    0 = A.Σ + Σ.A^T + C
        #    0 = A.μ + F
        # Σ is the covariance matrix, and μ is an array containing the means.
        if np.any(np.real(np.linalg.eigvals(_A)) >= 0):
            raise LinAlgError(
                f"Covariance matrix has no steady-state solution: "
                f"the dynamical matrix has {np.sum(np.real(np.linalg.eigvals(_A)) >= 0)} "
                f"unstable eigenevalues.")

        cov = solve_continuous_lyapunov(_A,-_C)
        mean = solve(_A,-_F)
        norm = 1

    else:
        # Solver for the steady-state of a non-norm-preserving Liouvillian. This 
        # solver converts the input data into the following non-linear matrix 
        # equations:
        #    0 = A.Σ + Σ.A^T - Σ.B.Σ + C
        #    0 = (A - Σ.B).μ + F - Σ.D
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
            raise LinAlgError(
                f"Covariance matrix has no steady-state solution: "
                f"stable subspace requires a dimension of {2 * L0.dims_cvs}, "
                f"however, the computed dimension is {len(_indices_stable)}.")

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


@dataclass(frozen=True)
class BackactionComponent:
    """
    The components of the backaction computed here are generally complex. This 
    class splits them into dephasing (real part) and ac-Stark frequency shift 
    (imaginary part) components. It can handle the backaction on a single
    element of an FLS-state, along with arrays corresponding to the backaction
    on multiple elements of an FLS-state.
    """
    value: complex | npt.NDArray[complex] = field()

    def __post_init__(self):
        if isinstance(self.value, BackactionComponent):
            # Allow re-wrapping an existing instance without double-nesting.
            object.__setattr__(self, 'value', self.value.value)
        elif isinstance(self.value, numbers.Number):
            object.__setattr__(self, 'value', complex(self.value))
        else:
            # The case where value is array_like.
            object.__setattr__(self, 'value', np.asarray(self.value, dtype=complex))

    @property
    def dephasing(self) -> float:
        return self.value.real

    @property
    def freq_shift(self) -> float:
        return self.value.imag

    def __add__(self, other) -> BackactionComponent:
        if isinstance(other, BackactionComponent):
            return BackactionComponent(self.value + other.value)
        else:
            return BackactionComponent(self.value + other)

    def __radd__(self, other) -> BackactionComponent: 
        return self.__add__(other)

    def __sub__(self, other) -> BackactionComponent: 
        return self.__add__(other.__neg__())

    def __rsub__(self, other) -> BackactionComponent: 
        return (self.__neg__()).__add__(other)

    def __neg__(self) -> BackactionComponent: 
        return BackactionComponent(-self.value)

    def __mul__(self, other) -> BackactionComponent:
        if isinstance(other, (numbers.Number, np.number)):
            return BackactionComponent(other*self.value)
        return NotImplemented

    def __rmul__(self, other) -> BackactionComponent: 
        return self.__mul__(other)
    
    def __truediv__(self, other) -> BackactionComponent:
        if isinstance(other, (numbers.Number, np.number)):
            if other == 0:
                return ZeroDivisionError
            return BackactionComponent(self.value/other)
        return NotImplemented
        
    def __repr__(self) -> str:
        return f"{self.value!r}"

    
class BackactionComponentTable(NamedTuple):
    """
    The backaction is a linear function which can be broken down into several
    components corresponding to different interactions between the CVS and FLS,
    along with the nature of the CVS state (whether it is in vacuum or not). 
    These components are stored here as a namedtuple.
    """
    total: BackactionComponent
    bare: BackactionComponent
    meas_long: BackactionComponent
    meas_disp: BackactionComponent
    para: BackactionComponent

    @property
    def meas(self) -> BackactionComponent: 
        return self.meas_long + self.meas_disp
    
    def __add__(self, other) -> BackactionComponentTable:
        if isinstance(other, BackactionComponentTable):
            return BackactionComponentTable(*(a + b for a, b in zip(self, other)))
        # scalar/complex broadcast: add the same value to every component
        elif isinstance(other, (numbers.Number, np.number)):
            return BackactionComponentTable(total = self.total + other,
                                            bare = self.bare + other,
                                            meas_long = self.meas_long,
                                            meas_disp = self.meas_disp,
                                            para = self.para)
        else:
            return NotImplemented

    def __radd__(self, other) -> BackactionComponentTable:
        return self.__add__(other)

    def __sub__(self, other) -> BackactionComponentTable:
        return self.__add__(other.__neg__())

    def __rsub__(self, other) -> BackactionComponentTable:
        return (self.__neg__()).__add__(other)
    
    def __neg__(self) -> BackactionComponentTable:
        return BackactionComponentTable(*(-a for a in self))

    def __mul__(self, other) -> BackactionComponentTable:
        if isinstance(other, (numbers.Number, np.number)):
            return BackactionComponentTable(*(a * other for a in self))
        else:
            return NotImplemented

    def __rmul__(self, other) -> BackactionComponentTable:
        return self.__mul__(other)

    def __truediv__(self, other) -> BackactionComponentTable:
        if isinstance(other, (numbers.Number, np.number)):
            if other == 0:
                return ZeroDivisionError
            return BackactionComponentTable(*(a / other for a in self))
        else:
            return NotImplemented


@dataclass(frozen=True)
class BackactionResult:
    """
    Dataclass holding the results of a calculation of the backaction on an
    individual component of an FLS state due to coupling to a CVS system.
    The individual components are stored as BackactionComponentTable, though
    various properties are define to allow for easier access by the user. 
    Calling these will return a number. 
    For a physical meaing of these components consult the description for the
    _backaction_component_solver function.
    """
    backaction: BackactionComponentTable
    state: QGstate = field(repr=False)
    super: QGsuper = field(repr=False)
    elem: tuple[int, int]

    def __post_init__(self):
        object.__setattr__(
            self, 'backaction',
            BackactionComponentTable(*(BackactionComponent(c) for c in self.backaction)))

    # Property forwarding to allow for easier user access.
    @property
    def dephasing(self) -> float: 
        return self.backaction.total.dephasing
    @property
    def freq_shift(self) -> float: 
        return self.backaction.total.freq_shift
    @property
    def total(self) -> BackactionComponent: 
        return self.backaction.total
    @property
    def bare(self) -> BackactionComponent: 
        return self.backaction.bare
    @property
    def meas_long(self) -> BackactionComponent: 
        return self.backaction.meas_long
    @property
    def meas_disp(self) -> BackactionComponent: 
        return self.backaction.meas_disp
    @property
    def meas(self) -> BackactionComponent: 
        return self.backaction.meas
    @property
    def para(self) -> BackactionComponent: 
        return self.backaction.para
    

@dataclass(frozen=True)
class BackactionResultArray:
    """
    Dataclass holding the results of a calculation of the backaction multiple
    components of an FLS state due to coupling to a CVS system. Includes a
    getitem method which will return an instance of BackactionResult.
    The individual components are stored as BackactionComponentTable, though
    various properties are define to allow for easier access by the user.
    Calling these will return an array.
    For a physical meaing of these components consult the description for the
    _backaction_component_solver function.
    """
    backaction: BackactionComponentTable
    state: QGstate = field(repr=False)
    super: QGsuper = field(repr=False)

    def __post_init__(self):
        object.__setattr__(
            self, 'backaction',
            BackactionComponentTable(*(BackactionComponent(c) for c in self.backaction)))

    # Property forwarding to allow for easier user access.
    @property
    def dephasing(self) -> float: 
        return self.backaction.total.dephasing
    @property
    def freq_shift(self) -> float: 
        return self.backaction.total.freq_shift
    @property
    def total(self) -> BackactionComponent: 
        return self.backaction.total
    @property
    def bare(self) -> BackactionComponent: 
        return self.backaction.bare
    @property
    def meas_long(self) -> BackactionComponent: 
        return self.backaction.meas_long
    @property
    def meas_disp(self) -> BackactionComponent: 
        return self.backaction.meas_disp
    @property
    def meas(self) -> BackactionComponent: 
        return self.backaction.meas
    @property
    def para(self) -> BackactionComponent: 
        return self.backaction.para

    def __getitem__(self, elem: tuple[int, int]) -> BackactionResult:
        _row, _col = elem
        _ba_elem = BackactionComponentTable(*(c[_row,_col] for c in self.backaction))
        return BackactionResult(backaction = _ba_elem,
                                state = self.state[_row,_col],
                                super = self.super[_row,_col],
                                elem = elem)


def backaction_steadystate(L0: QGsuper,
                           elem: tuple[int,int] = None,
                           tol: float = qgauss.settings.atol
                          ) -> BackactionResult | BackactionResultArray :
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
    BackactionResult | BackactionResultArray
        Dataclass storing resulant backaction dephasing rates and frequency
        shifts, either for the individual elements of the FLS state matrix or 
        the FLS state in its entirety, along with the QGsuper and QGstate used 
        to compute these quantities.
    """
    # --------------------------------------------------------------------------
    # No CVS or FLS component is present, raise error.
    if not L0.isfls and not L0.iscvs:
        raise ValueError("No CVS or FLS component. " \
                         "Backaction cannot be defined.")
    
    # No CVS component is present, raise error.
    elif L0.isfls and not L0.iscvs:
        raise ValueError("No CVS component coupled to the FLS component. " \
                         "Bakaction from the CVS component cannot be defined.")
    
    # No FLS component is present, solve the CV system.
    elif not L0.isfls and L0.iscvs:
        rhof = moment_steadystate(L0, tol=tol)
        ba_table = _backaction_component_solver(L0, rhof)

        return BackactionResult(backaction = BackactionComponentTable(*ba_table),
                                state = rhof,
                                super = L0,
                                elem = elem)
        
    # --------------------------------------------------------------------------
    # Element of the FLS state is specified, solve the corresponding CV component
    elif L0.isfls and elem is not None:
        _index = np.prod(L0.dims_fls[0][0])*elem[1] + elem[0]

        if not L0.issubgauss(_index):
            warnings.warn("Evolution of the CV system is not Gaussian. " \
            "Terms which violate this assumption will be ignored.")

        _L0_index = L0[_index, _index]
        rhof = moment_steadystate(_L0_index, tol=tol)
        ba_table = _backaction_component_solver(_L0_index, rhof)

        return BackactionResult(backaction = BackactionComponentTable(*ba_table),
                                state = rhof,
                                super = _L0_index,
                                elem = elem)
        
    # --------------------------------------------------------------------------
    # No element of the FLS state is specified, solve for all components.
    elif L0.isfls and elem is None:
        if not L0.isgauss:
            warnings.warn("Evolution of the CV system is not Gaussian. " \
            "Terms which violate this assumption will be ignored.")
        
        _fls_row = np.prod(L0.dims_fls[0][0])
        _fls_col = np.prod(L0.dims_fls[0][1])

        rhof = moment_steadystate(L0, tol=tol)
        _ba_total = np.empty([_fls_row,_fls_col], dtype=complex)
        _ba_bare = np.empty([_fls_row,_fls_col], dtype=complex)
        _ba_meas_long = np.empty([_fls_row,_fls_col], dtype=complex)
        _ba_meas_disp = np.empty([_fls_row,_fls_col], dtype=complex)
        _ba_para = np.empty([_fls_row,_fls_col], dtype=complex)

        for _row in range(0,_fls_row):
            for _col in range(0,_fls_col):
                _index = np.prod(L0.dims_fls[0][0])*_col + _row

                (_ba_total[_row,_col], 
                 _ba_bare[_row,_col], 
                 _ba_meas_long[_row,_col],
                 _ba_meas_disp[_row,_col], 
                 _ba_para[_row,_col]) = \
                _backaction_component_solver(L0[_index,_index],
                                             rhof[_row,_col])
                
        ba_table = BackactionComponentTable(_ba_total,
                                            _ba_bare,
                                            _ba_meas_long,
                                            _ba_meas_disp,
                                            _ba_para)
        
        return BackactionResultArray(backaction = ba_table,
                                     state = rhof,
                                     super = L0)


def _backaction_component_solver(L0: QGsuper,
                                 rho: QGstate
                                ) -> tuple[complex,complex,complex,complex,complex]:
    """
    The backaction on an operator ρ is defined as tr[ρ] = exp[-v]. The 
    backaction rate is then extracted from dv/dt, defined by
        dv/dt = G + μ.D + (1/2)*μ.B.μ + (1/2)*trace[B.Σ]
    where Σ and μ are the 2nd and 1st central moments of ρ, respectively. The 
    elements B,D,G are extracted from the Wigner representation of some 
    superoperator acting on the Wigner representation of ρ, denoted W(ρ), and
    correspond to the following elements of the full phase-space PDE:
        B_jk*r_j*r_k*W(ρ) : second order in the quadrature variables
        D_j*r_j*W(ρ)      : first order in the quadrature variables
        G*W(ρ)            : zeroth order/constant in the quadrature variables
    The dephasing rate and frequency shift correspond to the real and imaginary 
    parts of 'v', respectively. The backaction may be broken into bare/innate 
    'ba_bare', measurement induced component from logintudinal or dispersive 
    type interactions 'ba_meas_long' and 'ba_meas_disp', and parasitic 
    'ba_para', components:
        bare = G
        meas_long = μ.D
        meas_disp = (1/2)*μ.B.μ
        para = (1/2)*trace[B.Σ]

    ---- Parameters ----
    L0 : QGsuper
        Backaction due to QGsuper on the provided steady-state. Must be CVS-only.
    rho : QGstate
        Steady-state intracavity state whose moments are to be used to calculate
        the backaction. Must be CVS-only. 

    ---- Returns ----
    ba_total : complex
        Total backaction.
    ba_bare : complex
        Bare backation. The dephasing part of this components arises when 
        the CVS state has variances above vacuum, while the frequency shift is 
        present even for the vacuum shift. Also may include dephasing and 
        frequency component innate to the FLS system, and so independent of
        the moments of the CVS.
    ba_meas_long : complex
        Measurement induced backaction from longitudinal interactions. This 
        component is dependent on the displacement of the CVS.
    ba_meas_disp : complex
        Measurement induced backaction from dispersive interactions. This 
        component is dependent on the displacement of the CVS.
    ba_para : complex
        Parasitic backaction. This is due to fluctuations above vacuum in the 
        CVS and so is only dependent on the second moments.
    """
    # Generate arrays.
    _B,_D,_G = L0.wigner_2nd_rr, L0.wigner_1st_r, L0.wigner_0th[0]
    _mean, _cov = rho.data_1st, rho.data_2nd

    # Calculate the components of the backaction.
    ba_bare = _G
    ba_meas_long = _mean @ _D
    ba_meas_disp = (1/2)*(_mean @ _B @ _mean)
    ba_para = (1/2)*np.trace(_B @ _cov)
    ba_total = ba_bare + ba_meas_long + ba_meas_disp + ba_para

    return ba_total,ba_bare,ba_meas_long,ba_meas_disp,ba_para
