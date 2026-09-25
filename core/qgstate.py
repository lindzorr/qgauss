from __future__ import annotations

from functools import cached_property
import numbers
import operator
import math
import numpy.typing as npt
import qgauss
import numpy as np
from scipy.linalg import eigvals,eigh,fractional_matrix_power
from ..calc.utilities import trim

__all__ = ['QGstate']


class QGstate(object):

    """
    ---- Structure ----
    A class for representing density operators of continuous variable system 
    (CVS) Gaussian quantum states coupled to finite-level systems (FLS). The FLS 
    component of the density operator is still represented as a linear operator 
    acting on a Hilbert space, while the CVS component is represented in terms 
    of the various moments/cumulants of the Wigner quasi-probability 
    distribution (Wigner QPD). The basis of this representation is that the 
    total state may be written in the L×M FLS basis as:
        ρ_T = [[ρ_00, ρ_01, ... , ρ_0M],
               [ρ_10, ρ_11, ... , ρ_1M],
                //...//
               [ρ_L0, ρ_L1, ... , ρ_LM]]
    Assuming that the cavity component is Gaussian, then the characteristic 
    function of the Wigner QPD of ρ_jk, w[ξ]_jk, may be written as:
        ρ_jk --(Wigner)--> W[r]_jk --(Fourier, r->ξ)--> w[ξ]_jk , 
        where
        w[ξ]_jk = Exp[-(½)ξ·Σ_jk·ξ + iξ·μ_jk + ν_jk]
    The component ρ_jk is entirely characterized by three generally complex 
    quantities. Assuming that there are "N" CVS Gaussian modes:
        · The zeroth-order cumulant, ν_jk. Exp[ν_jk] represents the total 
        mass/norm of the Wigner QPD, in addition to any weight from the 
        FLS-component of the density matrix, and is the quantity that is stored.
        · The vector of raw first moments, or first cumulants, μ_jk, 
        representing the means. The dimensions are 1×2N.
        · The matrix of central second moments, or second cumulants, Σ_jk. Also 
        called the covariance matrix, even when the moments are complex, 
        (Σ_jk)^T = Σ_jk. The dimensions are 2N×2N.
    When ρ_jk is a true state then ν_jk = 1, μ_jk and Σ_jk are entirely real, 
    and Σ_jk + (i/2)*Ω ≥ 0, where Ω is the symplectic form. Even if ν_jk != 1, 
    ρ_jk may still represent a true Gaussian state up to a constant rescaling.
    
    In order to represent states of this form, we use four arrays to hold the 
    various moments/cumulants and FLS coefficients. These are structured as:
        data_2nd = [[Σ_00, ... , Σ_0M],    data_1st = [[μ_00, ... , μ_0M],    
                    [Σ_10, ... , Σ_1M],                [μ_10, ... , μ_1M],                
                     //...//                            //...//                            
                    [Σ_L0, ... , Σ_LM]]                [μ_L0, ... , μ_LM]]
        data_0th = [[Exp[ν_00], ... , Exp[ν_0M]],
                    [Exp[ν_10], ... , Exp[ν_1M]],
                     //...//                        
                    [Exp[ν_L0], ... , Exp[ν_LM]]]
    While the FLS component in data_0th may be constructed with respect to any 
    basis, the CVS component is defined with respect to the quadrature basis 
    only, with a specific ordering of the components. Currently, the ordering of 
    the quadrature basis and commutation relations always takes the form:
        r = [q_1, p_1, q_2, p_2, ... , q_M, p_M] , 
        where the commutator is [q_j,p_k] = i*δ_jk, 
        or more generally, [r_j,r_k] = i*Ω_jk.
    As a result, pairs of quadratures for the same CVS mode are always 
    neighbours, and the symplectic form always has the form:
        Ω = I_N ⊗ [[0,1],[-1,0]] = ⊕_{j=1}^N [[0,1],[-1,0]].
    Due to this form, when taking the tensor of two QGstate objects, the FLS 
    components will be combined in the usual way when taking the tensor product 
    of operators. However, the CVS components will be combined together using a 
    direct sum of the moment/cumulant arrays, since this takes the place of the 
    tensor product when working in phase space.

    ---- Parameters ----
    inpt : QGstate
        Create a copy of another QGstate.
    data_2nd : array_like
        Data for initialising the second-order central moments/second-order 
        cumulants/covariances of the CVS component.
    data_1st : array_like
        Data for initialising the first-order raw moments/first-order 
        cumulants/means of the CVS component.
    data_0th : array_like
        Data for initialising the zeroth-order cumulants of the CVS component. 
        The default value is 0.
    dims_cvs : int
        Total number of continuous variable cavity modes.
    dims_fls : array_like
        List of dimensions of the finite level systems, used to keep track of 
        the tensor structure.

    ---- ATtributes ----
    data_2nd/data_cov : array
        Tensor of 2D arrays containing the covariances, or second-order 
        cumulants/central moments, E[X^2]-E[X]^2.
    data_1st/data_mean : array
        Tensor of 1D arrays containing the means, or first-order cumulants/raw 
        moments, E[X^1].
    data_0th/data_norm/data_fls : array
        Tensor containing the norms, or zeroth-order cumulants of the 
        distribution, E[X^0].
    dims_cvs : int
        Number of continuous-variable cavity modes.
    dims_fls : list
        List of dimensions of the finite level systems, used to keep track of 
        the tensor structure.
    shape_2nd : tuple
        Underlying shape of data_2nd.
    shape_1st : tuple
        Underlying shape of data_1st.
    shape_0th : tuple
        Underlying shape of data_0th.
    iscvs : bool
        Does the QGstate have a CVS component.
    isfls : bool
        Does the QGstate have an FLS component.
    isherm : bool
        Is QGstate a Hermitian operator.
    isnormalized : bool
        Is QGstate is properly normalized, that is, does it have trace one.
    isintegrable : bool
        Is the state integrable. Done by checking whether the CVS components on 
        the diagonal are integrable for mixed state, or just the state if it is 
        CVS-only. FLS-only states are automatically integrable. The FLS 
        component must be square.
    ispositive : bool
        Does the QGstate represent a positive semi-definite operator on the
        Hilbert space. Currently only implemented for CVS and FLS-only states.
        The conditions is implemented differently for CVS and FLS-only states. 
        For the FLS-only state, it is required that the eigenvalues are 
        non-negative real numbers. For a CVS-only state, it is required that the
        eigenvalues of (data_2nd + i*Ω/2) be non-negative real numbers.
    isdensity : bool
        Checks if a CVS or FLS-only quantum state satisfies the conditions of a
        density operator, and so represents a true quantum state. In both cases, 
        it is required that the state be Hermitian, positive semi-definite, and
        of trace one.
    symform : array
        Symplectic form, for a system with N = dims_cvs, which has the form: : 
        Ω = I_N ⊗ [[0,1],[-1,0]] = ⊕_{j=1}^N [[0,1],[-1,0]].
        
    ---- Methods ----
    add/sub : (QGstate, 0 for now) -> QGstate
        Returns sum/difference.
    neg : QGstate -> QGstate
        Returns negative of QGstate.
    mult : (QGstate, complex) -> QGstate
        Multiplication of QGstate by a scaler.
    div : (QGstate, complex) -> QGstate
        Division of QGstate by a scaler.
    eq : (QGstate, QGstate) -> bool
        Check equality of two QGstates.
    and/&/tensor : (QGstate, QGstate) -> QGstate
        Shorthand for the tensor of two QGstates.
    getitem : QGstate (FLS-CVS) -> QGstate (CVS)
        Extract elements of QGstate with FLS and CV component, to create a 
        CVS-only QGstate.
    drop : (QGstate, int | array[int] | tuple[int]) -> QGstate
        Remove all specified CVS modes from QGstate. 
    keep : (QGstate, int | array[int] | tuple[int]) -> QGstate
        Keep only the specified CVS modes in QGstate.
    mode : (QGstate, int | array[int] | tuple[int]) -> QGstate
        Alternate naming for the "keep" method.     
    conj : QGstate -> QGstate
        Complex-conjugate of all elements of QGstate.
    trans : QGstate -> QGstate
        Transpose of all elements of QGstate.
    dag : QGstate -> QGstate
        Adjoint (dagger) of QGstate.
    trace : QGstate -> number
        Returns trace of the entire density operator represented by QGstate, 
        which is encoded in the diagonal elements of data_0th. Raises an error
        if the integral of the CVS component does not converge.
    purity : QGstate -> number
        Returns the purity of the density operator. Only possible for CVS and
        FLS only states.
    normalize :
        Checks if QGstate is normalized, and if not, updates data_0th so that 
        the trace is unity.
    tidyup(tol) :
        Removes small elements from QGstate below some cut-off "tol".

    """
    
    ### Quantum-Gaussian State (QGstate) Initialisation ###
    def __init__(self,
                 inpt: QGstate = None,
                 data_2nd: npt.ArrayLike = None,
                 data_1st: npt.ArrayLike = None,
                 data_0th: npt.ArrayLike | complex = None,
                 dims_cvs: int = None,
                 dims_fls: tuple[list[int],list[int]] = None
                ): 
        
        # QGstate as inpt, copy data.
        if isinstance(inpt, QGstate):
            self._dims_cvs = inpt.dims_cvs
            self._dims_fls = [list(d) for d in inpt.dims_fls]

            self._data_0th = inpt.data_0th.copy()
            self._data_1st = inpt.data_1st.copy()
            self._data_2nd = inpt.data_2nd.copy()
               
        # In all other cases, specific components of QGstate must be arguments.     
        elif inpt is None:
            # Set dimensions of FLS and CV components using specified dimensions.
            self._dims_cvs = QGstate._set_dims_cvs(dims_cvs)
            self._dims_fls = QGstate._set_dims_fls(dims_fls)

            # Set data arrays from input data. Setters check consistency with 
            # shapes derived from stored dimensions.
            self.data_0th = data_0th
            self.data_1st = data_1st
            self.data_2nd = data_2nd

            if qgauss.settings.auto_tidyup is True: 
                self.tidyup()
                
        else:
            raise TypeError("Input for constructing QGstate is either " \
            "ill-formatted or of incorrect type.")

    '''
    ------------------
        Properties
    ------------------
    '''

    @property
    def data_2nd(self) -> npt.NDArray:
        return self._data_2nd
    @data_2nd.setter
    def data_2nd(self, data):
        # Initialize array of covariances/2nd-order cumulants. 
        # Should this use the data_0th component to eliminate any CV-cumulants 
        # which should not be present if the norm is zero? That is:
        # np.sign(np.abs(data)) or np.where(self.data_0th!=0, 1, 0).
        if data is None:
            self._data_2nd = np.zeros(self.shape_2nd, dtype=complex)
        elif isinstance(data, (np.ndarray, list)):
            if np.shape(data) != self.shape_2nd:
                raise ValueError("Dimensions of data_2nd do not agree with stored dimensions.")  
            _data = np.array(data, dtype=complex)
            _data_T = _data.transpose([0,1,3,2] if self.isfls else [1,0])
            self._data_2nd = (_data + _data_T)/2
        else:
            raise TypeError("data_2nd is not of a supported type: array or list.")
        # Invalidate any dependent cached properties
        self._invalidate(['isherm','isnormalized','isintegrable'])

    @property
    def data_1st(self) -> npt.NDArray:
        return self._data_1st
    @data_1st.setter
    def data_1st(self, data):
        # Initialise array of means/1st-order cumulants.
        if data is None:
            self._data_1st = np.zeros(self.shape_1st, dtype=complex)
        elif isinstance(data, (np.ndarray, list)):
            if np.shape(data) != self.shape_1st:
                raise ValueError("Dimensions of data_1st do not agree with stored dimensions.") 
            self._data_1st = np.array(data, dtype=complex)
        else:
            raise TypeError("data_1st is not of a supported type: array or list.")
        # Invalidate any dependent cached properties
        self._invalidate(['isherm'])

    @property
    def data_0th(self) -> npt.NDArray:
        return self._data_0th
    @data_0th.setter
    def data_0th(self, data):
        # Initialize array of norms/zeroth-order cumulants.
        if data is None:
            self._data_0th = np.full(self.shape_0th, 1, dtype=complex)  
        elif isinstance(data, (np.ndarray, list)):
            if np.ndim(data) == 0:
                if self.shape_0th != (1,):
                    raise ValueError("Dimensions of data_0th do not agree with stored dimensions.")
                self._data_0th = np.array([data], dtype=complex)
            elif np.shape(data) != self.shape_0th:
                raise ValueError("Dimensions of data_0th do not agree with stored dimensions.")
            else:
                self._data_0th = np.array(data, dtype=complex) 
        elif isinstance(data, (numbers.Number, np.number)):
            if self.shape_0th != (1,):
                raise ValueError("Dimensions of data_0th do not agree with stored dimensions.")
            self._data_0th = np.array([data], dtype=complex)                
        else:
            raise TypeError("data_0th is not of a supported type: array, list, or number.")
        # Invalidate any dependent cached properties
        self._invalidate(['isherm','isnormalized'])

    # Set aliases to access the data arrays that are more human-readable, 
    # along with associated getattr and setattr.
    _aliases = {'data_cov': 'data_2nd', 
                'data_mean': 'data_1st', 
                'data_norm': 'data_0th',
                'data_fls': 'data_0th'}
    
    def __getattr__(self, name):
        if name in self._aliases:
            return getattr(self, self._aliases[name])
        else:
            raise AttributeError(f"QGstate has no attribute '{name}'.")

    def __setattr__(self, name, data):
        if name in self._aliases:
            return setattr(self, self._aliases[name], data)
        else:
            super().__setattr__(name, data)
    
    @property
    def dims_cvs(self) -> int:
        return self._dims_cvs
    @staticmethod
    def _set_dims_cvs(dims) -> int:
        if dims is None:
            return 0
        try:
            _mode_num = operator.index(dims)
        except TypeError:
            raise TypeError("dims_cvs must be an integer or None.") from None
        if _mode_num < 0:
            raise ValueError("dims_cvs must be non-negative.")
        return _mode_num

    @property
    def dims_fls(self) -> list[list[int]]:
        return self._dims_fls
    @staticmethod
    def _set_dims_fls(dims) -> list[list[int]]:
        # Normalise to a fresh [rows, cols] list of ints.
        if dims is None:
            return [[], []]
        try:
            rows, cols = dims
            rows = [operator.index(x) for x in rows]
            cols = [operator.index(x) for x in cols]
        except (TypeError, ValueError):
            raise TypeError("dims_fls must have the form [rows, cols] "
                            "with integer entries.") from None
        if not rows and not cols:
            return [[], []]
        if not rows or not cols:
            raise ValueError("dims_fls row and column dims must both be empty "
                             "or both be non-empty.")
        if min(rows + cols) < 1:
            raise ValueError("dims_fls entries must be positive integers.")
        return [rows, cols]

    @property
    def shape_2nd(self) -> tuple[int,int,int,int] | tuple[int,int]:
        if self.isfls:
            return (math.prod(self.dims_fls[0]), 
                    math.prod(self.dims_fls[1]), 
                    2*self.dims_cvs, 
                    2*self.dims_cvs)
        else:
            return (2*self.dims_cvs, 
                    2*self.dims_cvs)
    
    @property
    def shape_1st(self) -> tuple[int,int,int] | tuple[int]:
        if self.isfls:
            return (math.prod(self.dims_fls[0]), 
                    math.prod(self.dims_fls[1]), 
                    2*self.dims_cvs)
        else:
            return (2*self.dims_cvs,)
    
    @property
    def shape_0th(self) -> tuple[int,int] | tuple[int]:
        if self.isfls:
            return (math.prod(self.dims_fls[0]), 
                    math.prod(self.dims_fls[1]))
        else:
            return (1,)
    
    @cached_property
    def iscvs(self) -> bool:
        return self._dims_cvs > 0
    
    @cached_property
    def isfls(self) -> bool:
        return (len(self._dims_fls[0]) > 0 and 
                len(self._dims_fls[1]) > 0)
    
    @cached_property
    def isherm(self) -> bool:
        if self == self.dag():
            return True
        else:
            return False
        
    @cached_property
    def isnormalized(self) -> bool:
        if not self.isintegrable:
            return False
        else:
            if QGstate._iszero(self.trace() - 1):
            # self.trace() == 1 with customized tolerances
                return True
            else:
                return False
    
    @cached_property
    def isintegrable(self) -> bool:
        if self.isfls and not self.iscvs:
            return True
        elif self.isfls and self.iscvs:
            if self.shape_0th[0] == self.shape_0th[1]:
                return all([QGstate._integrable_cv(self[k,k]) 
                            for k in range(self.shape_0th[1])])
            else:
                return False
        else:
            return QGstate._integrable_cv(self)

    @staticmethod
    def _integrable_cv(self: QGstate) -> bool:
        # Determines if the specific CVS QGstate is integrable by checking if 
        # the real part of the precision matrix is positive definite. If 
        # data_0th is 0, then the state is automatically integrable.
        if self.data_0th == 0:
                return True
        else:
            try:
                _re_precision_mat = np.real(np.linalg.inv(self.data_2nd))
            except np.linalg.LinAlgError:
                return False
            evals = eigvals(_re_precision_mat)
            if (np.all(evals > 0) and 
                np.all(self.data_2nd == self.data_2nd.T)
                ):
                return True
            else:
                return False
            
    @cached_property
    def ispositive(self: QGstate) -> bool:
        if self.isfls and self.iscvs:
            # Still need to determine how to calculate eigenvalues CVS-FLS of
            # states where the CVS component is represented using its moments.
            raise NotImplementedError("Property not yet implemented for QGstates " \
            "which mix continuous and finite-level components.")
        elif self.isfls:
            if self.isherm:
                return np.all(eigh(self.data_0th, eigvals_only=True) >= 0)
            else:
                _evals = trim(eigvals(self.data_0th))
                return (np.all(np.isreal(_evals)) and np.all(_evals >= 0))
        else:
            if self.isherm:
                return np.all(eigh(self.data_2nd + (1j/2)*self.symform, 
                                   eigvals_only=True) >= 0)
            else:
                _evals = trim(eigvals(self.data_2nd + (1j/2)*self.symform))
                return (np.all(np.isreal(_evals)) and np.all(_evals >= 0))
        
    @cached_property
    def isdensity(self) -> bool:
        if self.isfls and self.iscvs:
            raise NotImplementedError("Property not yet implemented for QGstates " \
                        "which mix continuous and finite-level components.")
        else:
            return (self.isherm and self.isnormalized and self.ispositive)

    @cached_property
    def symform(self) -> npt.NDArray:
        return np.kron(np.identity(self.dims_cvs), np.array([[0,1],[-1,0]]))
    
    def _invalidate(self, attr_list):
        # Remove cached properties that have been set.
        for attr in attr_list:
            if attr in self.__dict__: 
                del self.__dict__[attr]
    
    '''
    ---------------
        Methods
    ---------------
    '''
    
    ### Addition and subtraction of QGstates ###
    '''
    Addition and subtraction for density operators. Currently undergoing major
    revisions. Trying to determine how to implement two distinct types of
    addition: one that results in superposition states or Gaussian states in
    certain circumstanses, and one that allows for the modification of moments.
    Also a result, only operations involving 0 are allowed.
    '''             
      
    def __add__(self, other) -> QGstate:
        # Addition with self.QGstate on the left
        if isinstance(other, (numbers.Number, np.number)) and other == 0:
            return QGstate(self)
        else:
            return NotImplemented
            
    def __radd__(self, other: QGstate) -> QGstate:
        # Addition with the self.QGstate on the right
        return self.__add__(other)
        
    def __sub__(self, other: QGstate) -> QGstate:
        # Subtraction with self.QGstate on the left
        if isinstance(other, (numbers.Number, np.number)):
            return self.__add__(other.__neg__())
        else:
            return NotImplemented
    
    def __rsub__(self, other: QGstate) -> QGstate:
        # Subtraction with self.QGstate on the right
        return (self.__neg__()).__add__(other)
        
    def __neg__(self) -> QGstate:
        # Negation of self.QGstate; only negates the norm
        return QGstate(data_2nd = self.data_2nd,
                       data_1st = self.data_1st,
                       data_0th = -self.data_0th,
                       dims_fls = self.dims_fls,
                       dims_cvs = self.dims_cvs)
        
    ### Multiplication and division of QGstates ###
    '''
    Multiplication and division for density operators. Currently undergoing major
    revisions. Trying to determine how to implement three distinct types of
    multiplication: one that implements the true Moyal/operator multiplication
    of moments (which may result in Gaussian states of their superposition),
    one that rescales all moments, and one that represents true scalar
    multiplication of the state by a scalar. Currently, only the last one is
    supported, allowing for multiplication and division by numbers, which 
    simply act on self.data_oth.
    '''

    def __mul__(self, other: numbers.Number | np.number) -> QGstate:
        # Multiplication by a number with self.QGstate on the left
        if isinstance(other, (numbers.Number, np.number)):
            return QGstate(data_2nd = self.data_2nd,
                           data_1st = self.data_1st,
                           data_0th = other*self.data_0th,
                           dims_fls = self.dims_fls,
                           dims_cvs = self.dims_cvs)
        else:
            return NotImplemented

    def __rmul__(self, other: numbers.Number | np.number) -> QGstate:
        # Multiplication by a number with self.QGstate on the right
        if isinstance(other, (numbers.Number, np.number)):
            return self.__mul__(other)
        else:
            return NotImplemented
        
    def __truediv__(self, other: numbers.Number | np.number) -> QGstate:
        # Division of self.QGstate by a number
        if isinstance(other, (numbers.Number, np.number)):
            if other == 0:
                raise ZeroDivisionError
            return QGstate(data_2nd = self.data_2nd,
                           data_1st = self.data_1st,
                           data_0th = self.data_0th/other,
                           dims_fls = self.dims_fls,
                           dims_cvs = self.dims_cvs)
        else:
            return NotImplemented
        
    ### Assorted Methods ###

    def __eq__(self, other: QGstate) -> bool:
        # Check equality of QGstates
        if isinstance(other, QGstate):
            same_dims = (self.dims_fls == other.dims_fls and
                         self.dims_cvs == other.dims_cvs)
            if same_dims:
                same_elems = \
                (QGstate._allclose(self.data_2nd, other.data_2nd) and
                 QGstate._allclose(self.data_1st, other.data_1st) and
                 QGstate._allclose(self.data_0th, other.data_0th))
                if same_elems: return True
                else: return False
            else: return False
        else: return False
        
    def __and__(self, other: QGstate) -> QGstate:
        # Returns tensor product of self and other
        return qgauss.tensor(self, other)
    
    def __getitem__(self, index) -> QGstate:
        # Grab CV elements from self at index in the FLS component
        # and return a QGstate with a CV component only
        if self.isfls and self.iscvs:
            return QGstate(data_2nd = self.data_2nd[index],
                           data_1st = self.data_1st[index],
                           data_0th = self.data_0th[index],
                           dims_cvs = self.dims_cvs)
        else:
            raise ValueError(
                "QGstate requires an FLS and CV component to use this method. " \
                "Access QGstate data arrays individually if specific elements are required.")
    
    def drop(self, *args) -> QGstate:
        # Removes CV modes specified in args from self, and return a new QGstate
        # List, array, or tuple of indices passed as args, convert to set to
        # remove repeated indices.
        if not args:
            return QGstate(self) 
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple, set)):
            args = set(args[0])
        else:
            args = set(args)
        if min(args) < 1 or max(args) > self.dims_cvs:
            raise ValueError(f"Mode indices must be integers in the range [1,{self.dims_cvs}].")
        # Generate indices to remove from CVS part
        _ind = [n for x in sorted(args) for n in (2*x-2, 2*x-1)]

        if self.isfls:
            return QGstate(data_2nd = \
                           np.delete(np.delete(self.data_2nd, _ind, axis=3), _ind, axis=2),
                           data_1st = np.delete(self.data_1st, _ind, axis=2),
                           data_0th = np.copy(self.data_0th),
                           dims_cvs = self.dims_cvs - len(args),
                           dims_fls = self.dims_fls)
        else:
            return QGstate(data_2nd = \
                           np.delete(np.delete(self.data_2nd, _ind, axis=1), _ind, axis=0),
                           data_1st = np.delete(self.data_1st, _ind, axis=0),
                           data_0th = np.copy(self.data_0th),
                           dims_cvs = self.dims_cvs - len(args))
   
    def keep(self, *args) -> QGstate:
        # Keeps CV modes specified in args from self, and return a new QGstate
        # List, array, or tuple of indices passed as args, convert to set to
        # remove repeated indices.
        if not args:
            return QGstate(data_0th = self.data_0th,
                           dims_cvs = 0,
                           dims_fls = self.dims_fls)
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple, set)):
            args = set(args[0])
        else:
            args = set(args)
        if min(args) < 1 or max(args) > self.dims_cvs:
            raise ValueError(f"Mode indices must be integers in the range [1,{self.dims_cvs}].")
        # Generate list of modes to keep from CVS part
        _ind = [n for x in sorted(args) for n in (2*x-2, 2*x-1)]

        if self.isfls:
            return QGstate(data_2nd = \
                           np.take(np.take(self.data_2nd, _ind, axis=3), _ind, axis=2),
                           data_1st = np.take(self.data_1st, _ind, axis=2),
                           data_0th = np.copy(self.data_0th),
                           dims_cvs = len(args),
                           dims_fls = self.dims_fls)
        else:
            return QGstate(data_2nd = \
                           np.take(np.take(self.data_2nd, _ind, axis=1), _ind, axis=0),
                           data_1st = np.take(self.data_1st, _ind, axis=0),
                           data_0th = np.copy(self.data_0th),
                           dims_cvs = len(args))
    
    def mode(self, *args) -> QGstate:
        # Alternate naming for the "keep" method
        return self.keep(*args)
    
    def conj(self) -> QGstate:
        # Complex-conjugate of all elements
        return QGstate(data_2nd = self.data_2nd.conj(),
                       data_1st = self.data_1st.conj(),
                       data_0th = self.data_0th.conj(),
                       dims_fls = self.dims_fls,
                       dims_cvs = self.dims_cvs)

    def trans(self, level = None) -> QGstate:
        # Transpose of arrays within the QGstate. Can specify the level at which 
        # it is applied, either "FLS"/"fls" or "CVS"/"cvs", or the entire array 
        # if none is passed.
        if self.isfls:
            if level is None:
                return QGstate(data_2nd = self.data_2nd.transpose([1,0,3,2]),
                               data_1st = self.data_1st.transpose([1,0,2]),
                               data_0th = self.data_0th.transpose([1,0]),
                               dims_fls = [self.dims_fls[1],self.dims_fls[0]],
                               dims_cvs = self.dims_cvs)
            elif level in ('FLS', 'fls'):
                return QGstate(data_2nd = self.data_2nd.transpose([1,0,2,3]),
                               data_1st = self.data_1st.transpose([1,0,2]),
                               data_0th = self.data_0th.transpose([1,0]),
                               dims_fls = [self.dims_fls[1],self.dims_fls[0]],
                               dims_cvs = self.dims_cvs)
            elif level in ('CVS', 'cvs'):
                return QGstate(data_2nd = self.data_2nd.transpose([0,1,3,2]),
                               data_1st = self.data_1st,
                               data_0th = self.data_0th,
                               dims_fls = self.dims_fls,
                               dims_cvs = self.dims_cvs)
            else:
                raise ValueError(f"QGstate transpose passed unsuitable argument '{level}'.")
        else:
            if level in ('CVS', 'cvs') or level is None:
                return QGstate(data_2nd = self.data_2nd.T,
                               data_1st = self.data_1st,
                               data_0th = self.data_0th,
                               dims_cvs = self.dims_cvs)
            elif level in ('FLS', 'fls'):
                return self
            else:
                raise ValueError(f"QGstate transpose passed unsuitable argument '{level}'.")

    def dag(self) -> QGstate:
        # Adjoint/complex-conjugate/dagger of QGstate
        if self.isfls:
            return QGstate(data_2nd = self.data_2nd.conj().transpose([1,0,3,2]),
                           data_1st = self.data_1st.conj().transpose([1,0,2]),
                           data_0th = self.data_0th.conj().transpose([1,0]),
                           dims_fls = [self.dims_fls[1],self.dims_fls[0]],
                           dims_cvs = self.dims_cvs)
        else:
            return QGstate(data_2nd = self.data_2nd.conj().T,
                           data_1st = self.data_1st.conj().T,
                           data_0th = self.data_0th.conj().T,
                           dims_cvs = self.dims_cvs)

    def trace(self) -> complex:
        # Trace of entire density. Checks that FLS component is square and that 
        # all CVS components on the diagonal are integrable.
        if self.isintegrable:
            if self.isfls:
                return np.trace(self.data_0th)
            else:
                return self.data_0th[0]
        else:
            raise ValueError("The trace of this state does not converge to a finite value.")
    
    def purity(self) -> complex:
        # Density of entire density. For FLS states, checks that the matrix is
        # square and for CVS it checks that all elements on the diagonal are 
        # integrable.
        if self.isfls and self.iscvs:
            raise NotImplementedError("Property not yet implemented for QGstates " \
                        "which mix continuous and finite-level components.")
        elif self.isfls:
            return np.trace(self.data_0th @ self.data_0th)
        else:
            if self.isintegrable:
                return np.power(self.data_0th[0],2)/np.sqrt(np.linalg.det(2*self.data_2nd))
            else:
                raise ValueError("The trace of this state does not converge to a finite value.")
    
    def normalize(self):
        # Check if QGstate is normalized, and if not, normalize data_0th.
        if self.isintegrable:
            if self.isfls and np.trace(self.data_0th) != 1:
                self.data_0th *= 1/np.trace(self.data_0th)
            elif self.data_0th[0] != 1:
                self.data_0th *= 1/self.data_0th[0]
        else:
            raise ValueError("The trace of this state does not converge to a finite value.")
        
    def tidyup(self, tol: float = None) -> QGstate:
        # Set real/imaginary components with magnitude below tol to zero.
        # Modifies the operator in place and returns it (allows chaining).
        if tol is None:
            tol = qgauss.settings.tidyup_atol
        for data in (self.data_2nd, self.data_1st, self.data_0th):
            data.real[np.abs(data.real) < tol] = 0
            data.imag[np.abs(data.imag) < tol] = 0
        # Cached flags may have changed now that small elements are zero.
        self._invalidate(['isherm', 'isnormalized', 'isintegrable', 
                          'ispositive', 'isdensity'])
        return self

    @staticmethod
    def _iszero(data: numbers.Integral | npt.NDArray) -> bool:
        # Checks whether the magnitude of all elements in the array are
        # within tolerance of zero
        return np.all(np.abs(data) < qgauss.settings.atol)

    @staticmethod
    def _allclose(a: npt.NDArray, b: npt.NDArray) -> bool:
    # True if a and b agree within atol (absolute) + rtol*|b| (relative).
    # Acts as a wrapper for the numpy function, but also checks shape.
        a = np.asarray(a)
        b = np.asarray(b)
        return a.shape == b.shape and bool(
            np.allclose(a, b,
                        atol=qgauss.settings.atol, 
                        rtol=qgauss.settings.rtol))