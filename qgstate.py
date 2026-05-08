from __future__ import annotations

from functools import cached_property
import numbers
import numpy.typing as npt
import qgauss
import numpy as np
from scipy.linalg import eigvals,eigh
from .fn_utilities import trim

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
        Ω = ⊗_{j=1}^N [[0,1],[-1,0]].
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
    isquantumstate : bool
        Checks if a CVS or FLS-only quantum state satisfies the conditions of a
        density operator, and so represents a true quantum state. In both cases, 
        it is required that the state be Hermitian, positive semi-definite, and
        of trace one.
    symform : array
        Symplectic form, for a system with N = dims_cvs, which has the form: : 
        Ω = ⊗_{j=1}^N [[0,1],[-1,0]].
        
    ---- Methods ----
    add/sub : (QGstate, QGstate) -> QGstate
        Returns sum/difference of two QGstates. Intended for creating 
        superposition states, not actual addition.
    neg : QGstate -> QGstate
        Returns negative of QGstate.
    mult : (QGstate, complex) -> QGstate
        Multiplication of QGoper by a scaler.
    div : (QGstate, complex) -> QGstate
        Division of QGstate by a scaler.
    eq : (QGstate, QGstate) -> bool
        Check equality of two QGstates.
    and/&/tensor : (QGstate, QGstate) -> QGstate
        shorthand for the tensor of two QGstates.
    getitem : QGstate (FLS-CVS) -> QGstate (CVS)
        Extract elements of QGstate with FLS and CV component, to create a 
        CVS-only QGstate.
    drop : (QGstate, int | array[int] | tuple[int]) -> QGstate
        Remove all specified CVS modes from QGstate. 
    keep : (QGstate, int | array[int] | tuple[int]) -> QGstate
        Keep only the specified CVS modes in QGstate. 
    conj : QGstate -> QGstate
        Complex-conjugate of all elements of QGstate.
    trans : QGstate -> QGstate
        Transpose of all elements of QGstate.
    dag : QGstate -> QGstate
        Adjoint (dagger) of QGstate.
    trace : QGstate -> number
        Returns trace of the entire density matrix represented by QGstate, which 
        is encoded in the diagonal elements of data_0th. Raises an error is the 
        integral of the CVS component does not converge.
    normalize : self
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
            self._dims_fls = inpt.dims_fls 
            self._iscvs = inpt.iscvs
            self._isfls = inpt.isfls

            self._data_0th = inpt.data_0th
            self._data_1st = inpt.data_1st
            self._data_2nd = inpt.data_2nd
               
        # In all other cases, specific components of QGstate must be arguments.     
        elif inpt is None:
            # Set dimensions of FLS and CV components. Use dimensions if 
            # specified, otherwise, calculate from input data and check for 
            # consistency. Also sets the isfls and iscvs properties.
            if dims_fls is not None:
                self.dims_fls = dims_fls
            elif dims_fls is None:
                self.dims_fls = QGstate._set_dims_fls(data_2nd, data_1st, data_0th)

            if dims_cvs is not None:
                self.dims_cvs = dims_cvs
            elif dims_cvs is None:
                self.dims_cvs = QGstate._set_dims_cvs(data_2nd, data_1st, data_0th)

            # Set data arrays from input data. Setters check consistency with 
            # shapes derived from stored dimensions.
            self.data_0th = data_0th
            self.data_1st = data_1st
            self.data_2nd = data_2nd

            if qgauss.settings.auto_tidyup == True: 
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
        # Initialize array of covariances/2nd-order cumulants. Uses data_0th 
        # component to eliminate any CV-cumulants which should not be present 
        # if the norm is zero:
        # np.sign(np.abs(data)) or np.where(self.data_0th!=0, 1, 0).
        if isinstance(data, (np.ndarray, list)):
            if np.shape(data) == self.shape_2nd:
                if self.isfls:
                    symm = (np.asarray(data, dtype=complex) 
                            + np.transpose(np.asarray(data, dtype=complex), [0,1,3,2]))/2
                    self._data_2nd = np.einsum("jk,jklm->jklm", 
                                               np.where(self.data_0th!=0, 1, 0), 
                                               symm)
                else:
                    symm = (np.asarray(data, dtype=complex) 
                            + np.transpose(np.asarray(data, dtype=complex)))/2
                    self._data_2nd = np.where(self.data_0th!=0, 1, 0)*symm
            else:
                raise ValueError("Dimensions of data_2nd do not agree with stored dimensions.")                     
        elif data is None:
            self._data_2nd = np.zeros(self.shape_2nd, dtype=complex)
        else:
            raise TypeError("data_2nd is not of a supported type: array or list.")
        # Invalidate any dependent cached properties
        self._invalidate(['isherm','isnormalized','isintegrable'])

    @property
    def data_1st(self) -> npt.NDArray:
        return self._data_1st
    @data_1st.setter
    def data_1st(self, data):
        # Initialise array of means/1st-order cumulants. Uses data_0th component 
        # to eliminate any CV-cumulants which should not be present if the norm 
        # is zero.
        if isinstance(data, (np.ndarray, list)):
            if np.shape(data) == self.shape_1st:
                if self.isfls:
                    self._data_1st = np.einsum("jk,jkl->jkl", 
                                               np.where(self.data_0th!=0, 1, 0), 
                                               np.asarray(data, dtype=complex))
                else:
                    self._data_1st = np.asarray(data, dtype=complex)
            else:
                raise ValueError("Dimensions of data_1st do not agree with stored dimensions.")  
        elif data is None:
            self._data_1st = np.zeros(self.shape_1st, dtype=complex)
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
        if isinstance(data, (np.ndarray, list)):
            if np.shape(data) == self.shape_0th:
                self._data_0th = np.asarray(data, dtype=complex)
            else:
                raise ValueError("Dimensions of data_0th do not agree with stored dimensions.")
        elif isinstance(data, (numbers.Number, np.number)):
            if self.shape_0th == (1,):
                self._data_0th = np.array([data], dtype=complex)
            else:
                raise ValueError("Dimensions of data_0th do not agree with stored dimensions.")
        elif data is None:
            self._data_0th = np.full(self.shape_0th, 1, dtype=complex)                             
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
    @dims_cvs.setter
    def dims_cvs(self, dims):
        if isinstance(dims, numbers.Integral):
            self._dims_cvs = int(dims)
        else:
            raise TypeError("dims_cvs is not of a supported type: number.")
        # Set iscvs property.
        self.iscvs = dims

    @property
    def dims_fls(self) -> list[list[int]]:
        return self._dims_fls
    @dims_fls.setter
    def dims_fls(self, dims):
        if isinstance(dims, (np.ndarray, list)):
            self._dims_fls = list(dims)
        else:
            raise TypeError("dims_fls is not of a supported type: array or list.")
        # Set isfls property.
        self.isfls = dims

    @property
    def shape_2nd(self) -> tuple[int,int,int,int] | tuple[int,int]:
        if self.isfls:
            return (np.prod(self.dims_fls[0]).item(), 
                    np.prod(self.dims_fls[1]).item(), 
                    2*self.dims_cvs, 
                    2*self.dims_cvs)
        else:
            return (2*self.dims_cvs, 
                    2*self.dims_cvs)
    
    @property
    def shape_1st(self) -> tuple[int,int,int] | tuple[int]:
        if self.isfls:
            return (np.prod(self.dims_fls[0]).item(), 
                    np.prod(self.dims_fls[1]).item(), 
                    2*self.dims_cvs)
        else:
            return (2*self.dims_cvs,)
    
    @property
    def shape_0th(self) -> tuple[int,int] | tuple[int]:
        if self.isfls:
            return (np.prod(self.dims_fls[0]).item(), 
                    np.prod(self.dims_fls[1]).item())
        else:
            return (1,)
    
    @property
    def iscvs(self) -> bool:
        return self._iscvs
    @iscvs.setter
    def iscvs(self, dims):
        if dims == 0 or dims is None:
            self._iscvs = False
        else:
            self._iscvs = True
    
    @property
    def isfls(self) -> bool:
        return self._isfls     
    @isfls.setter
    def isfls(self, dims):
        if dims == [[],[]] or dims is None:
            self._isfls = False
        else:
            self._isfls = True
    
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
            if self.trace() == 1:
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
                False
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
                evals = eigvals(_re_precision_mat)
                if (np.all(evals > 0) and 
                    np.all(_re_precision_mat == np.transpose(_re_precision_mat))
                   ):
                    return True
                else:
                    return False
            except:
                return False
            
    @cached_property
    def ispositive(self: QGstate) -> bool:
        if self.isfls and self.iscvs:
            # Still need to determine how to calculate eigenvalues CVS-FLS of
            # states where the CVS component is represented using its moments.
            return NotImplemented
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
    def isquantumstate(self) -> bool:
        if self.isfls and self.iscvs:
            return NotImplemented
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
    Addition and subtraction for density operators, modified to account for the 
    fact that adding two Gaussians with different moments results in a 
    non-Gaussian. These operations perform element-by-element addition on the 
    FLS density matrix. Addition of the CV-only subcomponents is only permitted 
    if all moments for one subcomponent are zero, or if both subcomponents have 
    identical second and first moments in which case only the norms are 
    combined. In all other cases an error is returned as the result is not a 
    Gaussian. It is intended that these functions be used for the combination of
    different QGstates of the same size during the initialisation of 
    superposition states.
    '''          
    
    @staticmethod
    def _addcv_(self: QGstate, 
                other: QGstate
               ) -> QGstate:
        # Adder for two CV-only QGstates
        if (QGstate._iszero(self.data_2nd - other.data_2nd) and
            QGstate._iszero(self.data_1st - other.data_1st)
           ):
            return QGstate(data_2nd = self.data_2nd,
                           data_1st = self.data_1st,
                           data_0th = self.data_0th + other.data_0th,
                           dims_fls = self.dims_fls,
                           dims_cvs = self.dims_cvs)
        elif (QGstate._iszero(self.data_2nd) and
              QGstate._iszero(self.data_1st) and
              QGstate._iszero(self.data_0th)
             ):
            return QGstate(other)
        elif (QGstate._iszero(other.data_2nd) and
              QGstate._iszero(other.data_1st) and
              QGstate._iszero(other.data_0th)
             ):
            return QGstate(self)
        else:
            raise ValueError("Addition of QGstates produces result which is not Gaussian.")

    @staticmethod
    def fls_to_list(input: QGstate) -> list[QGstate]:
        # Convert QGstate with FLS component to a list of QGstates which 
        # are CVS systems only
        return [[input[qr,qc] 
                 for qc in range(np.prod(input.dims_fls[0]))]
                 for qr in range(np.prod(input.dims_fls[1]))]
    
    @staticmethod
    def list_to_fls(input: QGstate, 
                    dims_fls: list[list[int]]
                   ) -> QGstate:
        # Convert list of QGstates which are CV systems only to a single 
        # QGstate with FLS component
        _col = range(np.prod(dims_fls[0]))
        _row = range(np.prod(dims_fls[1]))
        return QGstate(data_2nd = np.asarray([[input[qr][qc].data_2nd 
                                               for qc in _col]
                                               for qr in _row]),
                        data_1st = np.asarray([[input[qr][qc].data_1st 
                                                for qc in _col]
                                                for qr in _row]),
                        data_0th = np.asarray([[input[qr][qc].data_0th.item()
                                                for qc in _col]
                                                for qr in _row]),                      
                        dims_cvs = input[0][0].dims_cvs,
                        dims_fls = dims_fls)     
      
    def __add__(self, other: QGstate) -> QGstate:
        # Addition with self.QGstate on the left
        if isinstance(other, QGstate):
            if ((self.dims_cvs == other.dims_cvs) and 
                (self.dims_fls == other.dims_fls)
               ):
                if self.isfls and not self.iscvs:
                    return QGstate(data_0th = self.data_0th + other.data_0th,
                                   dims_fls = self.dims_fls)
                elif not self.isfls and self.iscvs:
                    return QGstate._addcv_(self,other)
                elif self.isfls and self.iscvs:
                    _self = QGstate.fls_to_list(self)
                    _other = QGstate.fls_to_list(other)
                    out = [[_self[qr][qc] + _other[qr][qc] 
                            for qc in range(np.prod(self.dims_fls[0]))]
                            for qr in range(np.prod(self.dims_fls[1]))]
                    return QGstate.list_to_fls(out, self.dims_fls)
            else:
                raise ValueError("Cannot perform addition of QGstates with different dimensions.")
        elif other == 0:
            return QGstate(self)
        else:
            raise TypeError("Cannot perform addition between QGstate and type" 
                            + type(other).__name__ + ".")
            
    def __radd__(self, other: QGstate) -> QGstate:
        # Addition with the self.QGstate on the right
        return self.__add__(other)
        
    def __sub__(self, other: QGstate) -> QGstate:
        # Subtraction with self.QGstate on the left
        return self.__add__(other.__neg__())
    
    def __rsub__(self, other: QGstate) -> QGstate:
        # Subtraction with self.QGstate on the right
        return other.__add__(self.__neg__())
        
    def __neg__(self) -> QGstate:
        # Negation of self.QGstate; only negates the norm
        return QGstate(data_2nd = self.data_2nd,
                       data_1st = self.data_1st,
                       data_0th = -self.data_0th,
                       dims_fls = self.dims_fls,
                       dims_cvs = self.dims_cvs)
        
    ### Multiplication and division of QGstates ###
    '''
    Currently, only multiplication or division by scalars is supported. In the
    future, it may be possible to multiply QGstate objects representing purely 
    CV states, with no FLS components; this will be equivalent to implementing 
    the Moyal star product between the two Wigner QPDs. By default, the FLS 
    component is rescaled; however, when absent, the zeroth-order cumulant of 
    the CV component is rescaled.
    '''

    def __mul__(self, other: complex) -> QGstate:
        # Multiplication by a number with self.QGstate on the left
        if isinstance(other, (numbers.Number, np.number)):
            return QGstate(data_2nd = self.data_2nd,
                           data_1st = self.data_1st,
                           data_0th = other*self.data_0th,
                           dims_fls = self.dims_fls,
                           dims_cvs = self.dims_cvs)
        else:
            raise TypeError("Cannot perform multiplication between QGstate "
            "and type" + type(other).__name__ + ".")

    def __rmul__(self, other: complex) -> QGstate:
        # Multiplication by a number with self.QGstate on the right
        return self.__mul__(other)

    def __truediv__(self, other: complex) -> QGstate:
        # Division of self.QGstate by a number
        if isinstance(other, (numbers.Number, np.number)): 
            return QGstate(data_2nd = self.data_2nd,
                           data_1st = self.data_1st,
                           data_0th = self.data_0th/other,
                           dims_fls = self.dims_fls,
                           dims_cvs = self.dims_cvs)
        else:
            raise TypeError("Cannot perform division between the QGstate "
            "and type" + type(other).__name__ + ".")

    ### Assorted Methods ###

    def __eq__(self, other: QGstate) -> bool:
        # Check equality of QGstates
        same_dims = (self.dims_fls == other.dims_fls and
                     self.dims_cvs == other.dims_cvs)
        same_elems = (QGstate._iszero(self.data_2nd - other.data_2nd) and
                      QGstate._iszero(self.data_1st - other.data_1st) and
                      QGstate._iszero(self.data_0th - other.data_0th))
        if (isinstance(other, QGstate) and 
            same_dims and 
            same_elems
           ):
            return True
        else:
            return False
        
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
            raise ValueError("QGstate requires an FLS and CV component to use "\
            "this method. Access arrays individually for specific elements.")
    
    def drop(self, *args) -> QGstate:
        # Removes CV modes specified in args from self, and return a new QGstate
        # List, array, or tuple of indices passed as args, convert to tuple
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple)):
            args = tuple(args[0])
        # Generate indices to remove from CVS part
        _ind = [n for x in args for n in (2*x-2, 2*x-1)]

        if self.isfls:
            return QGstate(data_2nd = \
                           np.delete(np.delete(self.data_2nd, _ind, axis=3), _ind, axis=2),
                           data_1st = np.delete(self.data_1st, _ind, axis=2),
                           data_0th = self.data_0th,
                           dims_fls = self.dims_fls,
                           dims_cvs = self.dims_cvs - len(args))
        else:
            return QGstate(data_2nd = \
                           np.delete(np.delete(self.data_2nd, _ind, axis=1), _ind, axis=0),
                           data_1st = np.delete(self.data_1st, _ind, axis=0),
                           data_0th = self.data_0th,
                           dims_cvs = self.dims_cvs - len(args))

   
    def keep(self, *args) -> QGstate:
        # Keeps CV modes specified in args from self, and return a new QGstate
        # List, array, or tuple of indices passed as args, convert to tuple
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple)):
            args = tuple(args[0])

        # Generate list of modes to remove from CVS part by taking difference 
        # with the set of all modes
        _ind = list(set(range(1,self.dims_cvs+1)) - set(args))
        return self.drop(_ind)
    
    def conj(self) -> QGstate:
        # Complex-conjugate of all elements
        return QGstate(data_2nd = np.conj(self.data_2nd),
                       data_1st = np.conj(self.data_1st),
                       data_0th = np.conj(self.data_0th),
                       dims_fls = self.dims_fls,
                       dims_cvs = self.dims_cvs)

    def trans(self, level = None) -> QGstate:
        # Transpose of arrays within the QGstate. Can specify the level at which 
        # it is applied, either "FLS" or "CVS", or the entire array if none 
        # is passed.
        if self.isfls:
            if level is None:
                return QGstate(data_2nd = np.transpose(self.data_2nd, [1,0,3,2]),
                               data_1st = np.transpose(self.data_1st, [1,0,2]),
                               data_0th = np.transpose(self.data_0th, [1,0]),
                               dims_fls = [self.dims_fls[1],self.dims_fls[0]],
                               dims_cvs = self.dims_cvs)
            elif level == 'FLS':
                return QGstate(data_2nd = np.transpose(self.data_2nd, [1,0,2,3]),
                               data_1st = np.transpose(self.data_1st, [1,0,2]),
                               data_0th = np.transpose(self.data_0th, [1,0]),
                               dims_fls = [self.dims_fls[1],self.dims_fls[0]],
                               dims_cvs = self.dims_cvs)
            elif level == 'CVS':
                return QGstate(data_2nd = np.transpose(self.data_2nd, [0,1,3,2]),
                               data_1st = self.data_1st,
                               data_0th = self.data_0th,
                               dims_fls = self.dims_fls,
                               dims_cvs = self.dims_cvs)
        else:
            if level == 'CVS' or level is None:
                return QGstate(data_2nd = np.transpose(self.data_2nd),
                               data_1st = self.data_1st,
                               data_0th = self.data_0th,
                               dims_cvs = self.dims_cvs)
            elif level == 'FLS':
                return self

    def dag(self) -> QGstate:
        # Adjoint/complex-conjugate/dagger of QGstate
        if self.isfls:
            return QGstate(data_2nd = np.transpose(np.conj(self.data_2nd), [1,0,3,2]),
                           data_1st = np.transpose(np.conj(self.data_1st), [1,0,2]),
                           data_0th = np.transpose(np.conj(self.data_0th), [1,0]),
                           dims_fls = [self.dims_fls[1],self.dims_fls[0]],
                           dims_cvs = self.dims_cvs)
        else:
            return QGstate(data_2nd = np.transpose(np.conj(self.data_2nd)),
                           data_1st = np.transpose(np.conj(self.data_1st)),
                           data_0th = np.transpose(np.conj(self.data_0th)),
                           dims_cvs = self.dims_cvs)

    def trace(self) -> QGstate:
        # Trace of entire density. Checks that FLS component is square and that 
        # all CVS components on the diagonal are integrable.
        if self.isintegrable:
            if self.isfls:
                return np.trace(self.data_0th)
            else:
                return self.data_0th[0]
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
        
    def tidyup(self, tol: float = qgauss.settings.tidyup_atol) -> QGstate:
        # Private void function to remove small magnitude elements 
        # from the data arrays.
        np.real(self.data_2nd)[np.abs(np.real(self.data_2nd)) < tol] = 0
        np.imag(self.data_2nd)[np.abs(np.imag(self.data_2nd)) < tol] = 0

        np.real(self.data_1st)[np.abs(np.real(self.data_1st)) < tol] = 0
        np.imag(self.data_1st)[np.abs(np.imag(self.data_1st)) < tol] = 0

        np.real(self.data_0th)[np.abs(np.real(self.data_0th)) < tol] = 0
        np.imag(self.data_0th)[np.abs(np.imag(self.data_0th)) < tol] = 0

    @staticmethod
    def _iszero(data: npt.NDArray) -> bool:
        # Checks whether the magnitude of all elements in the array are
        # within tolerance of zero
        return np.all(np.abs(data) < qgauss.settings.atol)
    
    @staticmethod
    def _set_dims_fls(data_2nd, data_1st, data_0th) -> list[list[int]]:
        # Static method to extract dimensions of the FLS component of the data 
        # during class initialization if none are provided. Checks whether the 
        # shape of the input data is consistent, and returns dims_fls.

        # Determine shape of FLS-component of data_2nd. If no component exists 
        # or data is None, set shapes to 0. 
        _data_2nd_shape_fls_row = 0
        _data_2nd_shape_fls_col = 0
        if data_2nd is not None:
            _data_2nd_shape = np.array(data_2nd).shape
            _data_2nd_axes = len(_data_2nd_shape)
            if _data_2nd_axes == 4:
                _data_2nd_shape_fls_row = _data_2nd_shape[0]
                _data_2nd_shape_fls_col = _data_2nd_shape[1]
            elif _data_2nd_axes == 2:
                pass
            else:
                raise ValueError("Shape of data_2nd cannot be handled by the" \
                " QGstate class and should be reformatted.")
        else:
            _data_2nd_axes = 0

        # Determine shape of FLS-component of data_1st. If no component exists 
        # or data is None, set shapes to 0. 
        _data_1st_shape_fls_row = 0
        _data_1st_shape_fls_col = 0
        if data_1st is not None:
            _data_1st_shape = np.array(data_1st).shape
            _data_1st_axes = len(_data_1st_shape)
            if _data_1st_axes == 3:
                _data_1st_shape_fls_row = _data_1st_shape[0]
                _data_1st_shape_fls_col = _data_1st_shape[1]
            elif _data_1st_axes == 1:
                pass
            else:
                raise ValueError("Shape of data_1st cannot be handled by the" \
                " QGstate class and should be reformatted.")
        else:
            _data_1st_axes = 0

        # Determine shape of FLS-component of data_0th. If no component exists 
        # or data is None, set shapes to 0. 
        _data_0th_shape_fls_row = 0
        _data_0th_shape_fls_col = 0
        if data_0th is not None:
            if isinstance(data_0th, (np.ndarray, list)):
                _data_0th_shape = np.array(data_0th).shape
                _data_0th_axes = len(_data_0th_shape)
                if _data_0th_axes == 2:
                    _data_0th_shape_fls_row = _data_0th_shape[0]
                    _data_0th_shape_fls_col = _data_0th_shape[1]
            elif isinstance(data_0th, (numbers.Number, np.number)):
                _data_0th_axes = 1
            else:
                raise ValueError("Shape of data_0th cannot be handled by the" \
                " QGstate class and should be reformatted.")
        else:
            _data_0th_axes = 0

        # Check if the data has no FLS component.
        if (_data_2nd_axes in (0,2) and 
            _data_1st_axes in (0,1) and
            _data_0th_axes in (0,1)
           ):
            return [[],[]]
        # Else, check that the data has the correct number of axes.
        elif (_data_2nd_axes in (0,4) and
              _data_1st_axes in (0,3) and
              _data_0th_axes in (0,2)
             ):
            # Check that the FLS dimensions are consistent.
            _data_shape_fls_row = list(set((_data_2nd_shape_fls_row,
                                            _data_1st_shape_fls_row,
                                            _data_0th_shape_fls_row,
                                            0)))
            _data_shape_fls_col = list(set((_data_2nd_shape_fls_col,
                                            _data_1st_shape_fls_col,
                                            _data_0th_shape_fls_col,
                                            0)))
            if len(_data_shape_fls_row) == 2 and len(_data_shape_fls_col) == 2:
                _len_fls_row = _data_shape_fls_row[np.nonzero(_data_shape_fls_row)][0]
                _len_fls_col = _data_shape_fls_col[np.nonzero(_data_shape_fls_col)][0]
                return [[_len_fls_row],[_len_fls_col]]
            else:
                raise ValueError("The FLS dimensions are inconsistent, and" \
                " so the class cannot be intialized.")
        # Else, the number of axes of the data input is inconsistent
        else:
            raise ValueError("Number of axes is inconsistent, and so the" \
            " class cannot be intialized.")
        
    @staticmethod
    def _set_dims_cvs(data_2nd, data_1st, data_0th) -> int:
        # Static method to extract dimensions of the CVS component of the data 
        # during class initialization if none are  provided. Checks whether the
        # shape of the input data is consistent, and returns dims_cvs.
        
        # Determine shape of CVS-component of data_2nd. If no component exists
        # or data is None, set shapes to 0. 
        _data_2nd_shape_cvs_row = 0
        _data_2nd_shape_cvs_col = 0
        if data_2nd is not None:
            _data_2nd_shape = np.array(data_2nd).shape
            _data_2nd_axes = len(_data_2nd_shape)
            if _data_2nd_axes == 4:
                _data_2nd_shape_cvs_row = _data_2nd_shape[2]
                _data_2nd_shape_cvs_col = _data_2nd_shape[3]
            elif _data_2nd_axes == 2:
                _data_2nd_shape_cvs_row = _data_2nd_shape[0]
                _data_2nd_shape_cvs_col = _data_2nd_shape[1]
            else:
                raise ValueError("Shape of data_2nd cannot be handled by the" \
                " QGstate class and should be reformatted.")
        else:
            _data_2nd_axes = 0
    
        # Determine shape of CVS-component of data_1st. If no component exists 
        # or data is None, set shapes to 0. 
        _data_1st_shape_cvs = 0
        if data_1st is not None:
            _data_1st_shape = np.array(data_1st).shape
            _data_1st_axes = len(_data_1st_shape)
            if _data_1st_axes == 3:
                _data_1st_shape_cvs = _data_1st_shape[2]
            elif _data_1st_axes == 1:
                _data_1st_shape_cvs = _data_1st_shape[0]
            else:
                raise ValueError("Shape of data_1st cannot be handled by the" \
                " QGstate class and should be reformatted.")
        else:
            _data_1st_axes = 0

        if data_0th is not None:
            if isinstance(data_0th, (np.ndarray, list)):
                _data_0th_axes = len(np.array(data_0th).shape)
            elif isinstance(data_0th, (numbers.Number, np.number)):
                _data_0th_axes = 1
            else:
                _data_0th_axes = 0
            if _data_0th_axes not in (0,1,2):
                raise ValueError("Shape of data_0th cannot be handled by the" \
                " QGstate class and should be reformatted.")
        
        # Check if the data has no CVS component.
        if ((_data_2nd_axes == 0) and 
            (_data_1st_axes == 0) and 
            (_data_0th_axes in (0,2))
           ):
            return 0
        # Check that the CVS dimensions are consistent.
        _data_shape_cvs = list(set((_data_2nd_shape_cvs_row,
                                    _data_2nd_shape_cvs_col,
                                    _data_1st_shape_cvs,
                                    0)))

        if len(_data_shape_cvs) == 2:
            _len_cvs = _data_shape_cvs[np.nonzero(_data_shape_cvs)][0]
            if _len_cvs % 2 == 0:
                return int(_len_cvs / 2)
            else:
                raise ValueError("Shape of CVS-component cannot be an odd number.")
        else:
            raise ValueError("The CVS dimensions are inconsistent, and" \
                " so the class cannot be intialized.")