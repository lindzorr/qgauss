from __future__ import annotations

import sys
import warnings
import types
import typing
import numbers
import numpy.typing as npt

import qgauss
import numpy as np

__all__ = ['QGoper']

class QGoper(object):
    
    """
    ---- Structure ----
    A class for representing operators acting on combined continuous variable (CV) and finite-level systems (FLS). The 
    CV is restricted to be at most bilinear functions of the quadrature operators, to ensure that any 
    or dynamics preserve the Gaussian nature of the CV component of the total state. Any generic quadrature operator 
    "Q" may be written as follows:
        Q = ½r.O(2).r + r.O(1) + O(0)
    where r is a vector of quadrature operators, herein assumed to take the following form,
        r = (q_1,p_1,q_2,p_2,...,q_N,p_N), N = dims_cvs
    where "q_j" is a position-like operator, and "p_j" is a momentum-like operator, with commutation relation:
        [q_j,p_k] = i*δ_jk. 
    The arrays O(2),O(1), and O(0) represent the coefficients of bilinear, linear, and constant-order terms, 
    respectively, of Q when expressed in this quadrature basis. These arrays are sufficient to represent Q, and in the 
    absence of any coupling to any FLSs, correspond to the following data structures of QGoper:
        data_2nd = O(2)
        data_1st = O(1)
        data_0th = O(0)
    The constructor is designed such that O(2) will always be symmetric through the application of the commutation 
    relations, so that the asymmetric component is simplified and added to data_0th. With the inclusion of FLS's, a 
    mixed CV-FLS operator "M" may instead be written as the sum over some set of FLS operators S_k and quadrature only 
    operators Q_k, as:
        M = Σ_j Q_j*S_j
    Compared to the CV operators Q_j, it is assumed that the FLS operators S_j are represented in the usual matrix 
    representation, as linear maps on a finite-dimensional Hilbert space. In this case, the data structures of QGoper 
    correspond to the following quantities:
        data_2nd = Σ_j np.einsum("kl,mn->klmn",S_j,O(2)_j)
        data_1st = Σ_j np.einsum("kl,m->klm",S_j,O(1)_j)     
        data_0th = Σ_j S_j*O(0)_j
    In this "mixed"-representation, data_2nd and data_1st are therefore arrays of the coefficient-arrays, whereas 
    data_0th which is just an array.
    Due to use of this structure, taking the tensor product works the same as usual on the FLS-level, using a 
	Kronecker product, but is instead a direct-sum on the CV-level. The elements data_2nd and data_1st therefore 
	require an extra operation to properly take the tensor.
    Although higher order combinations of quadrature operators may simplify to something which is ultimately 
	bilinear in order after application of the canonical commutation relations, these expressions in general cannot be
    handled by the QGoper class due to limitations of the representation. So, while a*a.dag()*a - a.dag()*a*a == a, an 
    error will be generated as each product individually cannot be represented as a QGoper. Note, that with braketing
    this is not a problem, and so (a*a.dag() - a.dag()*a)*a == a will evaluate just fine.

    ---- Parameters ----
    inpt : QGoper
        Create a copy of another QGoper.
    data_2nd : array_like
        Data for initialising the operator coefficients which are bilinear in the quadrature operators.
    data_1st : array_like
        Data for initialising the operator coefficients which are linear in the quadrature operators.
    data_0th : array_like
        Data for initialising the operator coefficients which are independent of any quadrature operator.
    dims_cvs : int
        Number of continuous-variable system modes.
    dims_fls : array_like
        List of dimensions of the finite level systems, used to keep track of the tensor structure.
        
    ---- Attributes ----
    data_2nd : array
        Tensor of 2D arrays containing coefficients for 2nd-order products of the CV quadrature operators.
    data_1st : array
        Tensor of 1D arrays containing coefficients for 1st-order products of the CV quadrature operators.
    data_0th : array
        Array containing coefficients for 0th-order products of the CV quadrature operators, 
        either constant terms or the FLS operators.
    dims_cvs : int
        Number of continuous-variable system modes.
    dims_fls : list
        List of dimensions of the finite level systems, used to keep track of the tensor structure.
    shape_2nd : tuple
        Underlying shape of data_2nd.
    shape_1st : tuple
        Underlying shape of data_1st.
    shape_0th : tuple
        Underlying shape of data_0th.
    iscvs : bool
        Does QGoper have a CV component.
    isfls : bool
        Does QGoper have an FLS component.
    isherm : bool
        Is QGoper a Hermitian operator.
    is2nd : bool
        Does QGoper have a 2nd-order quadrature component.
    is1st : bool
        Does QGoper have a 1st-order quadrature component.
    is0th : bool
        Does QGoper have a 0th-order quadrature component.
    symform : array
        Symplectic form, for a system with N = dims_cvs this has the form: Ω = ⊗_{j=1}^N [[0,1],[-1,0]].
        
    ---- Methods ----
    add/sub : (QGoper, QGoper | complex) -> QGoper
        Returns sum/difference of two QGopers or a QGoper and a number.
    neg : QGoper -> QGoper
        Returns negative of QGoper.
    mult : (QGoper, QGoper | complex) -> QGoper
        Multiplication of QGoper by a scaler or another QGoper.
    div : (QGoper, complex) -> QGoper
        Division of QGoper by a scaler.
    eq : (QGoper, QGoper) -> bool
        Check equality of two QGopers.
    and/&/tensor : (QGoper, QGoper) -> QGoper
        shorthand for the tensor of two QGopers.
    getitem : QGoper (FLS-CV) -> QGoper (CV)
        Extract elements of QGoper with FLS and CV component, to create a CV-only QGoper.
    drop : (QGoper, int | array[int] | tuple[int]) -> QGoper
        Remove all specified CV modes from QGoper. 
    keep : (QGoper, int | array[int] | tuple[int]) -> QGoper
        Keep only the specified CV modes in QGoper. 
    conj() : QGoper -> QGoper
        Complex-conjugate of all elements of QGoper.
    trans() : QGoper -> QGoper
        Transpose of all elements of QGoper.
    dag() : QGoper -> QGoper
        Adjoint (dagger) of QGoper.
    tidyup(tol) :
        Removes small elements from QGoper below some cut-off "tol".
    
    """
    
    ### Quantum-Gaussian Operator (QGoper) Initialisation ###
    def __init__(self, 
                 inpt: QGoper = None,
                 data_2nd: npt.NDArray = None,
                 data_1st: npt.NDArray = None,
                 data_0th: npt.NDArray | complex = None,
                 dims_cvs: int = None,
                 dims_fls: list[list[int]] = None
                ):

        # QGoper as input, copy data
        if isinstance(inpt, QGoper):
            self._dims_cvs = inpt.dims_cvs
            self._dims_fls = inpt.dims_fls
            self._iscvs = inpt.iscvs
            self._isfls = inpt.isfls

            self._data_0th = inpt.data_0th
            self._data_1st = inpt.data_1st
            self._data_2nd = inpt.data_2nd

        # In other cases, specific components of QGoper must be included as arguments        
        elif inpt is None:
            # Set dimensions of FLS and CV components from input data
            # Also sets the isfls and iscvs properties
            self.dims_cvs = dims_cvs
            self.dims_fls = dims_fls

            # Set data arrays from input data
            # Also sets the is2nd, is1st, and is0th properties
            self.data_0th = data_0th
            self.data_1st = data_1st
            self.data_2nd = data_2nd

            if qgauss.settings.auto_tidyup == True: 
                self.tidyup()

        else:
            raise TypeError("Input for constructing QGoper is ill-formatted or of incorrect type.")

    '''
    ------------------
        Properties
    ------------------
    '''

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
        # Set iscvs property
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
        # Set isfls property
        self.isfls = dims

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
    def data_2nd(self) -> npt.NDArray:
        return self._data_2nd
    @data_2nd.setter
    def data_2nd(self, data):
        # Initialize array of bilinear-order quadrature operator coefficients
        # Check size is consistent with dims, then split into symmetric and antisymmetric parts
        # Canonical commutation relations are applied to the antisymmetric, and added to data_0th
        if isinstance(data, (np.ndarray, list)):
            if np.shape(data) == self.shape_2nd:
                if self.isfls:
                    symm = (np.asarray(data, dtype = complex) 
                            + np.transpose(np.asarray(data, dtype = complex),[0,1,3,2]))/2
                    asym = (np.asarray(data, dtype = complex) 
                            - np.transpose(np.asarray(data, dtype = complex),[0,1,3,2]))/2
                    self._data_2nd = symm
                    if data.size != 0 and np.any(np.abs(asym) > qgauss.settings.atol):
                        self._data_0th += (-1j/4)*np.einsum('jknn->jk',np.einsum('ln,jknm->jklm',self.symform,asym))
                else:
                    symm = (np.asarray(data, dtype = complex) 
                            + np.transpose(np.asarray(data, dtype = complex)))/2
                    asym = (np.asarray(data, dtype = complex) 
                            - np.transpose(np.asarray(data, dtype = complex)))/2
                    self._data_2nd = symm
                    if data.size != 0 and np.any(np.abs(asym) > qgauss.settings.atol):
                        self._data_0th += (-1j/4)*np.array([np.einsum('nn',np.einsum('ln,nm->lm',self.symform,asym))])
            else:
                raise ValueError("Dimensions of data_2nd do not agree with stored dimensions.")          
        elif data is None:
            # If no data is provided, set data_2nd to zero matrix
            self._data_2nd = np.zeros(self.shape_2nd, dtype = complex)
        else:
            raise TypeError("Input of data_2nd is not of a supported type: np.ndarray or list.")
    
    @property
    def data_1st(self) -> npt.NDArray:
        return self._data_1st
    @data_1st.setter
    def data_1st(self, data):
        # Initialize array of linear-order quadrature operator coefficients
        if isinstance(data, (np.ndarray, list)):
            if np.shape(data) == self.shape_1st:
                self._data_1st = np.asarray(data, dtype = complex)
            else:
                raise ValueError("Dimensions of data_1st do not agree with stored dimensions.") 
        elif data is None:
            self._data_1st = np.zeros(self.shape_1st, dtype = complex)
        else:
            raise TypeError("Input of data_1st is not of a supported type: np.ndarray or list.")
    
    @property
    def data_0th(self) -> np.NDArray:
        return self._data_0th
    @data_0th.setter
    def data_0th(self, data):
        # Initialize array of zeroth-order quadrature operator coefficients
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
            self._data_0th = np.zeros(self.shape_0th, dtype = complex)               
        else:
            raise TypeError("data_0th is not of a supported type: np.ndarray, list, or number.")
    
    @property
    def is2nd(self) -> bool:
        if (np.all(np.abs(self.data_2nd) < qgauss.settings.atol) or 
            self.data_2nd.size == 0
            ):
            return False
        else:
            return True
    
    @property
    def is1st(self) -> bool:
        if (np.all(np.abs(self.data_1st) < qgauss.settings.atol) or 
            self.data_1st.size == 0
            ):
            return False
        else:
            return True 
    
    @property
    def is0th(self) -> bool:
        if (np.all(np.abs(self.data_0th) < qgauss.settings.atol) or 
            self.data_0th.size == 0
            ):
            return False
        else:
            return True
    
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

    ### Addition and subtraction of QGopers ###
    '''
    Addition and subtraction of QGopers behave in the normal way. The other object must be another QGoper of the same 
    dimensions, or, alternatively, a number in which case the number is multiplied by identity and added to data_0th.
    '''

    def __add__(self, other: QGoper | complex) -> QGoper:
        # Addition with QGoper on the left
        if isinstance(other, (numbers.Number, np.number)):
            # If other is a scalar, treat as identity QGoper with same dims as self
            if self._isfls:
                return QGoper(data_2nd = self.data_2nd,
                              data_1st = self.data_1st,
                              data_0th = self.data_0th + other*np.eye(self.shape_0th[0], self.shape_0th[1]),
                              dims_cvs = self.dims_cvs,
                              dims_fls = self.dims_fls
                             )
            else:
                return QGoper(data_2nd = self.data_2nd,
                              data_1st = self.data_1st,
                              data_0th = self.data_0th + other,
                              dims_cvs = self.dims_cvs
                              )
        elif isinstance(other, QGoper):
            # If other is a QGoper, must have sames dims as self
            if ((self.dims_cvs == other.dims_cvs) and 
                (self.dims_fls == other.dims_fls)
                ):
                return QGoper(data_2nd = self.data_2nd + other.data_2nd,
                              data_1st = self.data_1st + other.data_1st,
                              data_0th = self.data_0th + other.data_0th,
                              dims_cvs = self.dims_cvs,
                              dims_fls = self.dims_fls
                             )
            else:
                raise ValueError("Cannot perform addition operation between QGopers of different dimensions.")
        else:
            raise TypeError("Cannot perform addition operation between the types QGoper and " 
                            + type(other).__name__ + ".")

    def __radd__(self, other: QGoper | complex) -> QGoper:
        # Addition with QGoper on the right
        return self.__add__(other)

    def __sub__(self, other: QGoper | complex) -> QGoper:
        # Subtraction with QGoper on the left
        return self.__add__(other.__neg__())

    def __rsub__(self, other: QGoper | complex) -> QGoper:
        # Subtraction with QGoper on the right
        return (self.__neg__()).__add__(other)

    def __neg__(self) -> QGoper:
        # Negation of QGoper
        return QGoper(data_2nd = -self.data_2nd,
                      data_1st = -self.data_1st,
                      data_0th = -self.data_0th,
                      dims_cvs = self.dims_cvs,
                      dims_fls = self.dims_fls
                     )   

    ### Multiplication and division of QGopers ###
    '''
    Multiplication and division of a QGoper by a number is implemented. Additionally, multiplcation of QGopers with 
    identical dimensions is also implemented; simplification of the quadrature component is performed when constructor 
    is called using the known commutation relation between quadrature operators.
    '''

    def __mul__(self, other: QGoper | complex) -> QGoper:
        # Multiplication with QGoper on the left
        if isinstance(other, (numbers.Number, np.number)):
            # If other is a scalar, perform multiplication will all data in self
            return QGoper(data_2nd = other*self.data_2nd,
                          data_1st = other*self.data_1st,
                          data_0th = other*self.data_0th,
                          dims_cvs = self.dims_cvs,
                          dims_fls = self.dims_fls
                         )
        elif isinstance(other, QGoper):
            # If other is a QGoper, must have sames dims as self
            if ((self.dims_cvs == other.dims_cvs) and 
                (self.dims_fls == other.dims_fls)
                ):
                # Check that result is bilinear/bilinear order or less in CV quadratures
                if ((self.is2nd and other.is2nd) or
                    (self.is2nd and other.is1st) or
                    (self.is1st and other.is2nd)
                   ):
                    raise ValueError("Multiplcation of QGopers produces result which is "
                                     + "beyond bilinear order in quadrature operators.")
                else:
                # Multiplication using einsum: transformation depends on whether operators is FLS and/or CV
                # 2nd order output terms combines one 2nd and one 0th order term, or both 1st order terms
                # 1st order output terms combines one 1st and one 0th order term
                # 0th order output term combines both 0th order terms
                    if self._iscvs and self._isfls:
                        return QGoper(data_2nd = (np.einsum('jnlm,nk->jklm', 
                                                            self.data_2nd, other.data_0th)
                                                  + np.einsum('jn,nklm->jklm', 
                                                              self.data_0th, other.data_2nd)
                                                  + 2.*np.einsum('jnl,nkm->jklm', 
                                                                 self.data_1st, other.data_1st)),
                                      data_1st = (np.einsum('jnl,nk->jkl', 
                                                            self.data_1st, other.data_0th)
                                                  + np.einsum('jn,nkl->jkl', 
                                                              self.data_0th, other.data_1st)),
                                      data_0th = np.einsum('jn,nk->jk', 
                                                           self.data_0th, other.data_0th),
                                      dims_cvs = self.dims_cvs,
                                      dims_fls = self.dims_fls
                                     )
                    elif self._iscvs and not self._isfls:
                        return QGoper(data_2nd = (np.einsum('jk,k->jk', 
                                                            self.data_2nd, other.data_0th)
                                                  + np.einsum('j,jk->jk', 
                                                              self.data_0th, other.data_2nd)
                                                  + 2.*np.einsum('j,k->jk', 
                                                                 self.data_1st, other.data_1st)),
                                      data_1st = (np.einsum('j,j->j', 
                                                            self.data_1st, other.data_0th)
                                                  + np.einsum('j,j->j', 
                                                              self.data_0th, other.data_1st)),
                                      data_0th = np.einsum('j,j->j', 
                                                           self.data_0th, other.data_0th),
                                      dims_cvs = self.dims_cvs,
                                      dims_fls = self.dims_fls
                                     )
                    elif not self._iscvs and self._isfls:
                        return QGoper(data_0th = np.einsum('jn,nk->jk', 
                                                           self.data_0th, other.data_0th),
                                      dims_fls = self.dims_fls
                                     )
                    else:
                        return QGoper(dims_cvs = self.dims_cvs,
                                      dims_fls = self.dims_fls
                                     )
            else:
                raise ValueError("Cannot perform multiplcation operation between QGopers of different dimensions.")
        else:
            raise TypeError("Cannot perform multiplication operation between the types QGoper and " 
                            + type(other).__name__ + ".")

    def __rmul__(self, other: QGoper | complex) -> QGoper:
        # Multiplication with QGoper on the right
        if isinstance(other, (numbers.Number, np.number)):
            return self.__mul__(other)
        elif isinstance(other, QGoper):
            return other.__mul__(self)
        else:
            raise TypeError("Cannot perform multiplication operation between the types QGoper and " 
                            + type(other).__name__ + ".")

    def __truediv__(self, other: complex) -> QGoper:
        # Division of QGoper by a number
        if isinstance(other, (numbers.Number, np.number)):
            return QGoper(data_2nd = self.data_2nd/other,
                          data_1st = self.data_1st/other,
                          data_0th = self.data_0th/other,
                          dims_cvs = self.dims_cvs,
                          dims_fls = self.dims_fls
                         ) 
        else:
            raise TypeError("Cannot perform division operation between the types QGoper and " 
                            + type(other).__name__ + ".")

    def __pow__(self, n: int, m = None) -> QGoper:
        # Calculate powers of self.QGoper
        if ((m is not None) or
            (not isinstance(n, numbers.Integral)) or n < 0
           ):
            return NotImplemented
        elif n == 1:
            return QGoper(inpt=self)
        elif n == 2 and not self.is2nd:
            return self.__mul__(self)
        elif n >= 3 and not self.is2nd and not self.is1st:
            return QGoper(data_0th = np.linalg.matrix_power(self.data_0th, n),
                          dims_cvs = self.dims_cvs, 
                          dims_fls = self.dims_fls)
        else:
            raise ValueError("Power of QGoper produces result which is beyond bilinear order in quadrature operators.")

    ### Assorted Methods ###  

    def __eq__(self, other: QGoper) -> bool:
        # Check equality of QGopers
        if (isinstance(other, QGoper) and 
            (self.dims_fls == other.dims_fls) and
            (self.dims_cvs == other.dims_cvs) and
            np.all(np.abs(self.data_2nd - other.data_2nd) < qgauss.settings.atol) and 
            np.all(np.abs(self.data_1st - other.data_1st) < qgauss.settings.atol) and
            np.all(np.abs(self.data_0th - other.data_0th) < qgauss.settings.atol)
            ):
                return True
        else:
            return False

    def __and__(self, other: QGoper) -> QGoper:
        # Returns tensor product of self and other
        return qgauss.tensor(self, other)
    
    def __getitem__(self, index) -> QGoper:
        # Grab CV elements from self at index in the FLS component
        # and return a QGoper with a CV component only
        if self.isfls and self.iscvs:
            return QGoper(data_2nd = self.data_2nd[index],
                          data_1st = self.data_1st[index],
                          data_0th = self.data_0th[index],
                          dims_cvs = self.dims_cvs
                         )
        else:
            raise ValueError("QGoper requires an FLS and CV component to use this method. "
                             + "Access QGoper data arrays individually if specific elements are required.")     

    def drop(self, *args) -> QGoper:
        # Removes CV modes specified in args from self, and return a new QGoper
        # List, array, or tuple of indices passed as args, convert to tuple
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple)):
            args = tuple(args[0])
        # Generate indices to remove from CVS part
        ind = [n for x in args for n in (2*x-2, 2*x-1)]

        if self.isfls == False:
            return QGoper(data_2nd = np.delete(np.delete(self.data_2nd, ind, axis=1), ind, axis=0),
                          data_1st = np.delete(self.data_1st, ind, axis=0),
                          data_0th = self.data_0th,
                          dims_cvs = self.dims_cvs - len(args)
                          )
        else:
            return QGoper(data_2nd = np.delete(np.delete(self.data_2nd, ind, axis=3), ind, axis=2),
                          data_1st = np.delete(self.data_1st, ind, axis=2),
                          data_0th = self.data_0th,
                          dims_fls = self.dims_fls,
                          dims_cvs = self.dims_cvs - len(args)
                          )
   
    def keep(self, *args) -> QGoper:
        # Keeps CV modes specified in args from self, and return a new QGoper
        # List, array, or tuple of indices passed as args, convert to tuple
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple)):
            args = tuple(args[0])

        # Generate list of modes to remove from CVS part by taking difference with set of all modes
        ind = list(set(range(1,self.dims_cvs+1)) - set(args))
        return self.drop(ind)

    def conj(self) -> QGoper:
        # Complex-conjugate of all elements of the QGoper
        return QGoper(data_2nd = np.conj(self.data_2nd),
                      data_1st = np.conj(self.data_1st),
                      data_0th = np.conj(self.data_0th),
                      dims_cvs = self.dims_cvs,
                      dims_fls = self.dims_fls
                     )

    def trans(self, level = None) -> QGoper:
        # Transpose of arrays within the QGoper. Can specify the level at which it is
        # applied, either "FLS" or "CVS", or the entire array if none is passed.
        if self.isfls:
            if level is None:
                return QGoper(data_2nd = np.transpose(self.data_2nd,[1,0,3,2]),
                              data_1st = np.transpose(self.data_1st,[1,0,2]),
                              data_0th = np.transpose(self.data_0th,[1,0]),
                              dims_fls = [self.dims_fls[1],self.dims_fls[0]],
                              dims_cvs = self.dims_cvs
                              )
            elif level == 'FLS':
                return QGoper(data_2nd = np.transpose(self.data_2nd,[1,0,2,3]),
                              data_1st = np.transpose(self.data_1st,[1,0,2]),
                              data_0th = np.transpose(self.data_0th,[1,0]),
                              dims_fls = [self.dims_fls[1],self.dims_fls[0]],
                              dims_cvs = self.dims_cvs
                              )
            elif level == 'CVS':
                return QGoper(data_2nd = np.transpose(self.data_2nd,[0,1,3,2]),
                              data_1st = self.data_1st,
                              data_0th = self.data_0th,
                              dims_fls = self.dims_fls,
                              dims_cvs = self.dims_cvs
                              )
        else:
            if level == 'CVS' or level is None:
                return QGoper(data_2nd = np.transpose(self.data_2nd),
                              data_1st = self.data_1st,
                              data_0th = self.data_0th,
                              dims_cvs = self.dims_cvs
                              )
            elif level == 'FLS':
                return self
            

    def dag(self) -> QGoper:
        # Adjoint/complex-conjugate/dagger of QGoper
        if self.isfls:
            return QGoper(data_2nd = np.transpose(np.conj(self.data_2nd), [1,0,3,2]),
                          data_1st = np.transpose(np.conj(self.data_1st), [1,0,2]),
                          data_0th = np.transpose(np.conj(self.data_0th), [1,0]),
                          dims_fls = [self.dims_fls[1], self.dims_fls[0]],
                          dims_cvs = self.dims_cvs
                         )
        else:
            return QGoper(data_2nd = np.transpose(np.conj(self.data_2nd)),
                          data_1st = np.conj(self.data_1st),
                          data_0th = np.conj(self.data_0th),
                          dims_fls = [self.dims_fls[1], self.dims_fls[0]],
                          dims_cvs = self.dims_cvs
                         )

    def tidyup(self, tol: float = qgauss.settings.auto_tidyup_atol) -> QGoper:
        # Private void function to remove small magnitude elements from data arrays
        np.real(self.data_2nd)[np.abs(np.real(self.data_2nd)) < tol] = 0
        np.imag(self.data_2nd)[np.abs(np.imag(self.data_2nd)) < tol] = 0

        np.real(self.data_1st)[np.abs(np.real(self.data_1st)) < tol] = 0
        np.imag(self.data_1st)[np.abs(np.imag(self.data_1st)) < tol] = 0

        np.real(self.data_0th)[np.abs(np.real(self.data_0th)) < tol] = 0
        np.imag(self.data_0th)[np.abs(np.imag(self.data_0th)) < tol] = 0
