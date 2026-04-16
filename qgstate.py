from __future__ import annotations

import sys
import warnings
import types
import typing
import numbers
import numpy.typing as npt

import qgauss
import numpy as np

__all__ = ['QGstate']

class QGstate(object):

    """
    ---- Structure ----
    A class for representing density operators of continuous variable (CV) Gaussian quantum states coupled to 
    finite-level systems (FLS). The FLS component of the density operator is still represented as a linear operator 
    acting on a Hilbert space, while the CV component is represented in terms of the various moments/cumulants of the 
    Wigner quasi-probability distribution (Wigner QPD). The basis of this representation is that the total state may 
    be written in the L×M FLS basis as:
        ρ_T = [[ρ_00, ρ_01, ... , ρ_0M],
               [ρ_10, ρ_11, ... , ρ_1M],
                //...//
               [ρ_L0, ρ_L1, ... , ρ_LM]]
    Assuming that the cavity component is Gaussian, then the characteristic function of the Wigner QPD of ρ_jk, 
    w[ξ]_jk, may be written as:
        ρ_jk --(Wigner transform)--> W[r]_jk --(Fourier transform, r->ξ)--> w[ξ]_jk , 
        where
        w[ξ]_jk = Exp[-(½)ξ·Σ_jk·ξ + iξ·μ_jk + ν_jk]
    The component ρ_jk is entirely characterized by three generally complex quantities. Assuming that there are "N" 
    CV Gaussian modes:
        · The zeroth-order cumulant, ν_jk. Exp[ν_jk] represents the total mass of the Wigner QPD, in addition to any 
            weight from the FLS-component of the density matrix, and is the quantity that is stored.
        · The vector of raw first moments, or first cumulants, μ_jk, representing the means. The dimensions are 1×2N.
        · The matrix of central second moments, or second cumulants, Σ_jk. Also called the covariance matrix, 
            even when the moments are complex, (Σ_jk)^T = Σ_jk. The dimensions are 2N×2N.
    When ρ_jk is a true state then ν_jk = 1, μ_jk and Σ_jk are entirely real, and Σ_jk + (i/2)*Ω ≥ 0, where Ω is the 
    symplectic form. Even if ν_jk != 1, ρ_jk may still represent a true Gaussian CV state up to a constant rescaling.
    
    In order to represent states of this form, we use four arrays to hold the various moments/cumulants and FLS 
    coefficients. These are structured as:
        data_2nd = [[Σ_00, ... , Σ_0M],    data_1st = [[μ_00, ... , μ_0M],    data_0th = [[Exp[ν_00], ... , Exp[ν_0M]],
                    [Σ_10, ... , Σ_1M],                [μ_10, ... , μ_1M],                [Exp[ν_10], ... , Exp[ν_1M]],
                     //...//                            //...//                            //...//
                    [Σ_L0, ... , Σ_LM]]                [μ_L0, ... , μ_LM]]                [Exp[ν_L0], ... , Exp[ν_LM]]]
    While the FLS component in data_0th may be constructed with respect to any basis, the CV component is defined with 
    respect to the quadrature basis only, with a specific ordering of the components. Currently, the ordering of the 
    quadrature basis and commutation relations always takes the form:
        r = [q_1, p_1, q_2, p_2, ... , q_M, p_M] , 
        where the commutator is [q_j,p_k] = i*δ_jk, 
        or more generally, [r_j,r_k] = i*Ω_jk.
    As a result, pairs of quadratures for the same CV mode are always neighbours, and the symplectic form always 
    has the form:
        Ω = ⊗_{j=1}^N [[0,1],[-1,0]].
    Due to this form, when taking the tensor of two QGstate objects, the FLS components will be combined in the usual 
    way when taking the tensor product of operators. However, the CV components will be combined together using a direct 
    sum of the moment/cumulant arrays, since this takes the place of the tensor product when working in phase space.

    ---- Parameters ----
    inpt : QGstate
        Create a copy of another QGstate.
    data_2nd : array_like
        Data for initialising the second-order central moments/second-order cumulants/covariances of the CV component.
    data_1st : array_like
        Data for initialising the first-order raw moments/first-order cumulants/means of the CV component.
    data_0th : array_like
        Data for initialising the zeroth-order cumulants of the CV component. The default value is 0.
    dims_cvs : int
        Total number of continuous variable cavity modes.
    dims_fls : array_like
        List of dimensions of the finite level systems, used to keep track of the tensor structure.

    ---- ATtributes ----
    data_2nd : array
        Tensor of 2D arrays containing the covariances, or second-order cumulants/central moments, E[X^2]-E[X]^2.
    data_1st : array
        Tensor of 1D arrays containing the means, or first-order cumulants/raw moments/, E[X].
    data_0th : array
        Tensor of zeroth-order cumulants of the distribution, ln(E[X^0]).
    dims_cvs : int
        Number of continuous-variable cavity modes.
    dims_fls : list
        List of dimensions of the finite level systems, used to keep track of the tensor structure.
    shape_2nd : tuple
        Underlying shape of data_2nd.
    shape_1st : tuple
        Underlying shape of data_1st.
    shape_0th : tuple
        Underlying shape of data_0th.
    iscvs : bool
        Does the QGstate have a CV component.
    isfls : bool
        Does the QGstate have an FLS component.
    isherm : bool
        Is QGstate a Hermitian operator.
    symform : array
        Symplectic form, for a system with N = dims_cvs this has the form: Ω = ⊗_{j=1}^N [[0,1],[-1,0]].
        
    ---- Methods ----
    add/sub : (QGstate, QGstate) -> QGstate
        Returns sum/difference of two QGstates. Intended for creating superposition states, not actual addition.
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
    getitem : QGstate (FLS-CV) -> QGstate (CV)
        Extract elements of QGstate with FLS and CV component, to create a CV-only QGstate.
    drop : (QGstate, int | array[int] | tuple[int]) -> QGstate
        Remove all specified CV modes from QGstate. 
    keep : (QGstate, int | array[int] | tuple[int]) -> QGstate
        Keep only the specified CV modes in QGstate. 
    conj() : QGstate -> QGstate
        Complex-conjugate of all elements of QGstate.
    trans() : QGstate -> QGstate
        Transpose of all elements of QGstate.
    dag() : QGstate -> QGstate
        Adjoint (dagger) of QGstate.
    trace() : QGstate -> number
        Returns trace of the entire density matrix represented by QGstate,
        which is encoded in the diagonal elements of data_0th.
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
                 dims_fls: list[list[int]] = None
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
               
        # In all other cases, specific components of QGstate must be included as arguments.     
        elif inpt is None:
            # Set dimensions of FLS and CV components from input data. Also sets the isfls and iscvs properties.
            self.dims_cvs = dims_cvs
            self.dims_fls = dims_fls

            # Set data arrays from input data.
            self.data_0th = data_0th
            self.data_1st = data_1st
            self.data_2nd = data_2nd

            if qgauss.settings.auto_tidyup == True: 
                self.tidyup()
                
        else:
            raise TypeError("Input for constructing QGstate is either ill-formatted or of incorrect type.")

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
        # Uses data_0th component to eliminate any CV-cumulants which  
        # should not be present through multiplication by zero,
        # np.sign(np.abs(data)) or np.where(self.data_0th!=0,1,0).
        if isinstance(data, (np.ndarray, list)):
            if np.shape(data) == self.shape_2nd:
                if self.isfls:
                    symm = (np.asarray(data, dtype = complex) 
                            + np.transpose(np.asarray(data, dtype = complex),[0,1,3,2]))/2
                    self._data_2nd = np.einsum("jk,jklm->jklm", np.where(self.data_0th!=0,1,0), symm)
                else:
                    symm = (np.asarray(data, dtype = complex) 
                            + np.transpose(np.asarray(data, dtype = complex)))/2
                    self._data_2nd = np.where(self.data_0th!=0,1,0)*symm
            else:
                raise ValueError("Dimensions of data_2nd do not agree with stored dimensions.")                     
        elif data is None:
            self._data_2nd = np.zeros(self.shape_2nd, dtype = complex)
        else:
            raise TypeError("Input of data_2nd is not of a supported type: np.ndarray or list.")
            
    @property
    def data_1st(self) -> npt.NDArray:
        return self._data_1st
    @data_1st.setter
    def data_1st(self, data):
        # Initialise array of means/1st-order cumulants.
        # Uses data_0th component to eliminate any CV-cumulants which  
        # should not be present through multiplication by zero.
        if isinstance(data, (np.ndarray, list)):
            if np.shape(data) == self.shape_1st:
                if self.isfls:
                    self._data_1st = np.einsum("jk,jkl->jkl", 
                                               np.where(self.data_0th!=0,1,0), 
                                               np.asarray(data, dtype = complex)
                                               )
                else:
                    self._data_1st = np.asarray(data, dtype = complex)
            else:
                raise ValueError("Dimensions of data_1st do not agree with stored dimensions.")  
        elif data is None:
            self._data_1st = np.zeros(self.shape_1st, dtype = complex)
        else:
            raise TypeError("Input of data_1st is not of a supported type: np.ndarray or list.")
            
    @property
    def data_0th(self) -> npt.NDArray:
        return self._data_0th
    @data_0th.setter
    def data_0th(self, data):
        # Initialize array of zeroth-order cumulants.
        if isinstance(data, (np.ndarray, list)):
            if np.shape(data) == self.shape_0th:
                self._data_0th = np.asarray(data, dtype = complex)
            else:
                raise ValueError("Dimensions of data_0th do not agree with stored dimensions.")
        elif isinstance(data, (numbers.Number, np.number)):
            if self.shape_0th == (1,):
                self._data_0th = np.array([data], dtype = complex)
            else:
                raise ValueError("Dimensions of data_0th do not agree with stored dimensions.")
        elif data is None:
            self._data_0th = np.full(self.shape_0th, 1, dtype = complex)                             
        else:
            raise TypeError("Input of data_0th is not of a supported type: np.ndarray, list, or number.")
    
    @property
    def dims_cvs(self) -> int:
        return self._dims_cvs
    @dims_cvs.setter
    def dims_cvs(self, dims):
        if isinstance(dims, numbers.Integral):
            self._dims_cvs = int(dims)
        elif dims is None:
            self._dims_cvs = 0
        else:
            raise TypeError("Input to dims_cvs is not of a supported type: number.")
        # Set iscvs property.
        self.iscvs = dims

    @property
    def dims_fls(self) -> list[list[int]]:
        return self._dims_fls
    @dims_fls.setter
    def dims_fls(self, dims):
        if isinstance(dims, (np.ndarray, list)):
            self._dims_fls = list(dims)
        elif dims is None:
            self._dims_fls = [[],[]]
        else:
            raise TypeError("Input to dims_fls is not of a supported type: np.ndarray or list.")
        # Set isfls property.
        self.isfls = dims

    @property
    def shape_2nd(self) -> tuple[int,int,int,int] | tuple[int,int]:
        if self._isfls:
            return (np.prod(self.dims_fls[0]).item(), 
                    np.prod(self.dims_fls[1]).item(), 
                    2*self.dims_cvs, 
                    2*self.dims_cvs
                    )
        else:
            return (2*self.dims_cvs, 
                    2*self.dims_cvs
                    )
    
    @property
    def shape_1st(self) -> tuple[int,int,int] | tuple[int]:
        if self._isfls:
            return (np.prod(self.dims_fls[0]).item(), 
                    np.prod(self.dims_fls[1]).item(), 
                    2*self.dims_cvs
                    )
        else:
            return (2*self.dims_cvs,)
    
    @property
    def shape_0th(self) -> tuple[int,int] | tuple[int]:
        if self._isfls:
            return (np.prod(self.dims_fls[0]).item(), 
                    np.prod(self.dims_fls[1]).item()
                    )
        else:
            return (1,)
    
    @property
    def iscvs(self) -> bool:
        return self._iscvs
    @iscvs.setter
    def iscvs(self, dims):
        if dims == 0 or dims == None:
            self._iscvs = False
        else:
            self._iscvs = True
    
    @property
    def isfls(self) -> bool:
        return self._isfls     
    @isfls.setter
    def isfls(self, dims):
        if dims == [[],[]] or dims == None:
            self._isfls = False
        else:
            self._isfls = True
    
    @property
    def isherm(self) -> bool:
        if self == self.dag():
            return True
        else:
            return False
     
    @property
    def symform(self) -> npt.NDArray:
        return np.kron(np.identity(self.dims_cvs),np.array([[0,1],[-1,0]]))
    
    '''
    ---------------
        Methods
    ---------------
    '''
    
    ### Addition and subtraction of QGstates ###
    '''
    Addition and subtraction for density operators, modified to account for the fact that adding two Gaussians with
    different moments results in a non-Gaussian. These operations perform element-by-element addition on the FLS 
    density matrix. Addition of the CV-only subcomponents is only permitted if all moments for one subcomponent are 
    zero, or if both subcomponents have identical second and first moments in which case only the norms are combined. 
    In all other cases an error is returned as the result is not a Gaussian. It is intended that these functions be used 
    for the combination of different QGstates of the same size during the initialisation of superposition states.
    '''          
    
    @staticmethod
    def _addcv_(self: QGstate, 
                other: QGstate
                ) -> QGstate:
        # Adder for two CV-only QGstates
        if (np.all(np.abs(self.data_2nd - other.data_2nd) < qgauss.settings.atol) and
            np.all(np.abs(self.data_1st - other.data_1st) < qgauss.settings.atol)
            ):
            return QGstate(data_2nd = self.data_2nd,
                           data_1st = self.data_1st,
                           data_0th = self.data_0th + other.data_0th,
                           dims_fls = self.dims_fls,
                           dims_cvs = self.dims_cvs
                           )
        elif (np.all(np.abs(self.data_2nd) < qgauss.settings.atol) and
              np.all(np.abs(self.data_1st) < qgauss.settings.atol) and
              np.abs(self.data_0th) < qgauss.settings.atol
              ):
            return QGstate(other)
        elif (np.all(np.abs(other.data_2nd) < qgauss.settings.atol) and
              np.all(np.abs(other.data_1st) < qgauss.settings.atol) and
              np.abs(other.data_0th) < qgauss.settings.atol
              ):
            return QGstate(self)
        else:
            raise ValueError("Addition of QGstates produces result which is not Gaussian.")

    @staticmethod
    def fls_to_list(input: QGstate) -> list[QGstate]:
        # Convert QGstate with FLS component to a list of QGstates which are CV systems only
        return [[input[qr,qc] 
                 for qc in range(np.prod(input.dims_fls[0]))]
                 for qr in range(np.prod(input.dims_fls[1]))]
    
    @staticmethod
    def list_to_fls(input: QGstate, 
                    dims_fls: list[list[int]]
                    ) -> QGstate:
        # Convert list of QGstates which are CV systems only to a single QGstate with FLS component
        return QGstate(data_2nd = np.asarray([[input[qr][qc].data_2nd 
                                               for qc in range(np.prod(dims_fls[0]))]
                                               for qr in range(np.prod(dims_fls[1]))]),
                        data_1st = np.asarray([[input[qr][qc].data_1st 
                                                for qc in range(np.prod(dims_fls[0]))]
                                                for qr in range(np.prod(dims_fls[1]))]),
                        data_0th = np.asarray([[input[qr][qc].data_0th.item()
                                                for qc in range(np.prod(dims_fls[0]))]
                                                for qr in range(np.prod(dims_fls[1]))]),                      
                        dims_cvs = input[0][0].dims_cvs,
                        dims_fls = dims_fls
                        )     
      
    def __add__(self, other: QGstate) -> QGstate:
        # Addition with self.QGstate on the left
        if isinstance(other, QGstate):
            if ((self.dims_cvs == other.dims_cvs) and 
                (self.dims_fls == other.dims_fls)):
                if self.isfls and not self.iscvs:
                    return QGstate(data_0th = self.data_0th + other.data_0th,
                                   dims_fls = self.dims_fls
                                   )
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
                raise ValueError("Cannot perform addition operation of QGstates with different dimensions.")
        elif other == 0:
            return QGstate(self)
        else:
            raise TypeError("Cannot perform addition operation between the types QGstate and " 
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
                       dims_cvs = self.dims_cvs
                       )
        
    ### Multiplication and division of QGstates ###
    '''
    Currently, only multiplication or division by scalars is supported. In future, it may be possible to multiply 
    QGstate objects representing purely CV states, with no FLS components; this will be equivalent to implementing the 
    Moyal star product between the two Wigner QPDs. By default, the FLS component is rescaled; however, when absent, 
    the zeroth-order cumulant of the CV component is rescaled.
    '''

    def __mul__(self, other: complex) -> QGstate:
        # Multiplication by a number with self.QGstate on the left
        if isinstance(other, (numbers.Number, np.number)):
            return QGstate(data_2nd = self.data_2nd,
                           data_1st = self.data_1st,
                           data_0th = other*self.data_0th,
                           dims_fls = self.dims_fls,
                           dims_cvs = self.dims_cvs
                           )
        else:
            raise TypeError("Cannot perform multiplication operation between the types QGstate and " 
                            + type(other).__name__ + ".")

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
                           dims_cvs = self.dims_cvs
                          )
        else:
            raise TypeError("Cannot perform division operation between the types QGstate and " 
                            + type(other).__name__ + ".")

    ### Assorted Methods ###

    def __eq__(self, other: QGstate) -> bool:
        # Check equality of QGstates
        if (isinstance(other, QGstate) and
            (self.dims_fls == other.dims_fls) and
            (self.dims_cvs == other.dims_cvs) and
            np.all(np.abs(self.data_2nd - other.data_2nd) < qgauss.settings.atol) and 
            np.all(np.abs(self.data_1st - other.data_1st) < qgauss.settings.atol) and
            np.all(np.abs(self.data_0th - other.data_0th) < qgauss.settings.atol)
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
                           dims_cvs = self.dims_cvs
                           )
        else:
            raise ValueError("QGstate requires an FLS and CV component to use this method. "
                             + "Access QGstate data arrays individually if specific elements are required.")
    
    def drop(self, *args) -> QGstate:
        # Removes CV modes specified in args from self, and return a new QGstate
        # List, array, or tuple of indices passed as args, convert to tuple
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple)):
            args = tuple(args[0])
        # Generate indices to remove from CVS part
        ind = [n for x in args for n in (2*x-2, 2*x-1)]

        if self.isfls == False:
            return QGstate(data_2nd = np.delete(np.delete(self.data_2nd, ind, axis=1), ind, axis=0),
                           data_1st = np.delete(self.data_1st, ind, axis=0),
                           data_0th = self.data_0th,
                           dims_cvs = self.dims_cvs - len(args)
                           )
        else:
            return QGstate(data_2nd = np.delete(np.delete(self.data_2nd, ind, axis=3), ind, axis=2),
                           data_1st = np.delete(self.data_1st, ind, axis=2),
                           data_0th = self.data_0th,
                           dims_fls = self.dims_fls,
                           dims_cvs = self.dims_cvs - len(args)
                           )
   
    def keep(self, *args) -> QGstate:
        # Keeps CV modes specified in args from self, and return a new QGstate
        # List, array, or tuple of indices passed as args, convert to tuple
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple)):
            args = tuple(args[0])

        # Generate list of modes to remove from CVS part by taking difference with set of all modes
        ind = list(set(range(1,self.dims_cvs+1)) - set(args))
        return self.drop(ind)
    
    def conj(self) -> QGstate:
        # Complex-conjugate of all elements
        return QGstate(data_2nd = np.conj(self.data_2nd),
                       data_1st = np.conj(self.data_1st),
                       data_0th = np.conj(self.data_0th),
                       dims_fls = self.dims_fls,
                       dims_cvs = self.dims_cvs
                       )

    def trans(self, level = None) -> QGstate:
        # Transpose of arrays within the QGstate. Can specify the level at which it is
        # applied, either "FLS" or "CVS", or the entire array if none is passed.
        if self.isfls:
            if level is None:
                return QGstate(data_2nd = np.transpose(self.data_2nd,[1,0,3,2]),
                               data_1st = np.transpose(self.data_1st,[1,0,2]),
                               data_0th = np.transpose(self.data_0th,[1,0]),
                               dims_fls = [self.dims_fls[1],self.dims_fls[0]],
                               dims_cvs = self.dims_cvs
                               )
            elif level == 'FLS':
                return QGstate(data_2nd = np.transpose(self.data_2nd,[1,0,2,3]),
                               data_1st = np.transpose(self.data_1st,[1,0,2]),
                               data_0th = np.transpose(self.data_0th,[1,0]),
                               dims_fls = [self.dims_fls[1],self.dims_fls[0]],
                               dims_cvs = self.dims_cvs
                               )
            elif level == 'CVS':
                return QGstate(data_2nd = np.transpose(self.data_2nd,[0,1,3,2]),
                               data_1st = self.data_1st,
                               data_0th = self.data_0th,
                               dims_fls = self.dims_flsh,
                               dims_cvs = self.dims_cvs
                               )
        else:
            if level == 'CVS' or level is None:
                return QGstate(data_2nd = np.transpose(self.data_2nd),
                               data_1st = self.data_1st,
                               data_0th = self.data_0th,
                               dims_cvs = self.dims_cvs
                               )
            elif level == 'FLS':
                return self

    def dag(self) -> QGstate:
        # Adjoint/complex-conjugate/dagger of QGstate
        if self.isfls:
            return QGstate(data_2nd = np.transpose(np.conj(self.data_2nd),[1,0,3,2]),
                           data_1st = np.transpose(np.conj(self.data_1st),[1,0,2]),
                           data_0th = np.transpose(np.conj(self.data_0th),[1,0]),
                           dims_fls = [self.dims_fls[1],self.dims_fls[0]],
                           dims_cvs = self.dims_cvs
                          )
        else:
            return QGstate(data_2nd = np.transpose(np.conj(self.data_2nd)),
                           data_1st = np.transpose(np.conj(self.data_1st)),
                           data_0th = np.transpose(np.conj(self.data_0th)),
                           dims_cvs = self.dims_cvs
                          )

    def trace(self) -> QGstate:
        # Trace of entire density, currently assumes that all CV components are integrable
        if self.isfls:
            return np.trace(self.data_0th)
        else:
            return self.data_0th[0]
        
    def tidyup(self, tol: float = qgauss.settings.tidyup_atol) -> QGstate:
        # Private void function to remove small magnitude elements from data arrays
        np.real(self.data_2nd)[np.abs(np.real(self.data_2nd)) < tol] = 0
        np.imag(self.data_2nd)[np.abs(np.imag(self.data_2nd)) < tol] = 0

        np.real(self.data_1st)[np.abs(np.real(self.data_1st)) < tol] = 0
        np.imag(self.data_1st)[np.abs(np.imag(self.data_1st)) < tol] = 0

        np.real(self.data_0th)[np.abs(np.real(self.data_0th)) < tol] = 0
        np.imag(self.data_0th)[np.abs(np.imag(self.data_0th)) < tol] = 0