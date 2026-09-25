from __future__ import annotations

import numbers
import operator
import math
import numpy.typing as npt
from functools import cached_property
import qgauss
import numpy as np

__all__ = ['QGoper']


class QGoper(object):
    
    """
    ---- Structure ----
    A class for representing operators acting on combined continuous variable 
    systems (CVS) and finite-level systems (FLS). The CVS part of the operator 
    is restricted to be at most a quadratic/bilinear function of the quadrature 
    operators, to ensure that the dynamics preserve the Gaussian nature of the 
    CVS component of the total state. Any generic quadrature operator "Q" may 
    be written as follows:
        Q = ½r.O(2).r + r.O(1) + O(0)
    where r is a vector of quadrature operators, herein assumed to take the 
    following form,
        r = (q_1,p_1,q_2,p_2,...,q_N,p_N), N = dims_cvs
    where "q_j" is a position-like operator, and "p_j" is a momentum-like 
    operator, with commutation relation:
        [q_j,p_k] = i*δ_jk. 
    The arrays O(2),O(1), and O(0) represent the coefficients of quadratic/
    bilinear, linear, and constant-order terms, respectively, of Q when 
    expressed in this quadrature basis. These arrays are sufficient to represent 
    Q, and in the absence of any coupling to any FLSs, correspond to the 
    following data structures of QGoper:
        data_2nd = O(2)
        data_1st = O(1)
        data_0th = O(0)
    The constructor is designed such that O(2) will always be symmetric through 
    the application of the commutation relations, so that the asymmetric 
    component is simplified and added to data_0th. With the inclusion of FLS's, 
    a mixed CVS-FLS operator "M" may instead be written as the sum over some set 
    of FLS operators S_k and quadrature only operators Q_k, as:
        M = Σ_j Q_j*S_j
    Compared to the CVS operators Q_j, it is assumed that the FLS operators S_j 
    are represented in the usual matrix representation, as linear maps on a 
    finite-dimensional Hilbert space. In this case, the data structures of 
    QGoper correspond to the following quantities:
        data_2nd = Σ_j np.einsum("kl,mn->klmn",S_j,O(2)_j)
        data_1st = Σ_j np.einsum("kl,m->klm",S_j,O(1)_j)     
        data_0th = Σ_j S_j*O(0)_j
    In this "mixed"-representation, data_2nd and data_1st are therefore arrays 
    of the coefficient-arrays, whereas data_0th which is just an array.

    Due to use of this structure, taking the tensor product works the same as 
    usual on the FLS-level, using a Kronecker product, but is instead a 
    direct-sum on the CVS-level. The elements data_2nd and data_1st therefore 
    require an extra operation to properly take the tensor.

    Although higher order combinations of quadrature operators may simplify to 
    something which is ultimately quadratic/bilinear in order after application 
    of the canonical commutation relations, these expressions in general cannot 
    be handled by the QGoper class due to limitations of the representation. 
    So, while a*a.dag()*a - a.dag()*a*a == a, an error will be generated as each 
    product individually cannot be represented as a QGoper. Note, that with 
    braketing this is not a problem, and so (a*a.dag() - a.dag()*a)*a == a will 
    evaluate just fine.

    ---- Parameters ----
    inpt : QGoper
        Create a copy of another QGoper.
    data_2nd : array_like
        Data for initialising the operator coefficients which are 
        quadratic/bilinear in the quadrature operators.
    data_1st : array_like
        Data for initialising the operator coefficients which are linear in the 
        quadrature operators.
    data_0th : array_like
        Data for initialising the operator coefficients which are independent of 
        any quadrature operator.
    dims_cvs : int
        Number of continuous-variable system modes.
    dims_fls : array_like
        List of dimensions of the finite level systems, used to keep track of 
        the tensor structure.
        
    ---- Attributes ----
    data_2nd/data_quad : array
        Tensor of 2D arrays containing coefficients for 2nd-order products of 
        the CVS quadrature operators.
    data_1st/data_lin : array
        Tensor of 1D arrays containing coefficients for 1st-order products of 
        the CVS quadrature operators.
    data_0th/data_const : array
        Array containing coefficients for 0th-order products of the CVS 
        quadrature operators, either constant terms or the FLS operators.
    dims_cvs : int
        Number of continuous-variable system modes.
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
        Does QGoper have a CVS component.
    isfls : bool
        Does QGoper have an FLS component.
    is2nd : bool
        Does QGoper have a 2nd-order quadrature component.
    is1st : bool
        Does QGoper have a 1st-order quadrature component.
    is0th : bool
        Does QGoper have a 0th-order quadrature component.
    isherm : bool
        Is QGoper a Hermitian operator.
    isgauss : bool
        Will evolution under this operator preserve the Gaussian nature of the 
        superposition FLS-CVS state.
    symform : array
        Symplectic form, for a system with N = dims_cvs, which has the form: 
            Ω = I_N ⊗ [[0,1],[-1,0]] = ⊕_{j=1}^N [[0,1],[-1,0]].
        
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
    getitem : QGoper (FLS-CVS) -> QGoper (CVS)
        Extract elements of QGoper with FLS and CVS component, to create a 
        CVS-only QGoper.
    drop : (QGoper, int | array[int] | tuple[int]) -> QGoper
        Remove all specified CVS modes from QGoper. 
    keep : (QGoper, int | array[int] | tuple[int]) -> QGoper
        Keep only the specified CVS modes in QGoper.
    mode : (QGoper, int | array[int] | tuple[int]) -> QGoper
        Alternate naming for the "keep" method.
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
                 data_2nd: npt.ArrayLike = None,
                 data_1st: npt.ArrayLike = None,
                 data_0th: npt.ArrayLike | complex = None,
                 dims_cvs: int = None,
                 dims_fls: list[list[int]] = None
                ):

        # QGoper as input, copy data.
        if isinstance(inpt, QGoper):
            self._dims_cvs = inpt.dims_cvs
            self._dims_fls = [list(d) for d in inpt.dims_fls]

            self._asym_corr = inpt._asym_corr.copy()
            self._data_0th = inpt.data_0th.copy()
            self._data_1st = inpt.data_1st.copy()
            self._data_2nd = inpt.data_2nd.copy()

        # In other cases, specific components of QGoper must be arguments.
        elif inpt is None:
            # Set dimensions of FLS and CV components using specified dimensions.
            self._dims_cvs = QGoper._set_dims_cvs(dims_cvs)
            self._dims_fls = QGoper._set_dims_fls(dims_fls)
            
            # Set data arrays from input data.
            self._asym_corr = np.zeros(self.shape_0th, dtype=complex)
            self.data_0th = data_0th
            self.data_1st = data_1st
            self.data_2nd = data_2nd

            if qgauss.settings.auto_tidyup is True: 
                self.tidyup()

        else:
            raise TypeError("Input for constructing QGoper is ill-formatted or of incorrect type.")

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
        # Initialize array of quadratic/bilinear-order quadrature operator 
        # coefficients. Check size is consistent with dims, then split into
        # symmetric and antisymmetric parts. Canonical commutation relations 
        # are applied to the antisymmetric, and added to data_0th.
        if data is None:
            # If no data is provided, set data_2nd to zero matrix.
            _symm = np.zeros(self.shape_2nd, dtype=complex)
            _corr = np.zeros(self.shape_0th, dtype=complex)
        elif isinstance(data, (np.ndarray, list)):
            if np.shape(data) != self.shape_2nd:
                raise ValueError("Dimensions of data_2nd do not agree with stored dimensions.")
            _corr = np.zeros(self.shape_0th, dtype=complex)
            _data = np.array(data, dtype=complex)
            _data_T = _data.transpose([0,1,3,2] if self.isfls else [1,0])
            _symm = (_data + _data_T)/2
            _asym = (_data - _data_T)/2
            if (_asym.size != 0 and
                np.any(np.abs(_asym) > qgauss.settings.atol)
                ):
                if self.isfls:
                    _corr = \
                    (-1j/4)*np.einsum('jknn->jk',np.einsum('ln,jknm->jklm',self.symform,_asym))
                else:
                    _corr = \
                    (-1j/4)*np.array([np.einsum('nn',np.einsum('ln,nm->lm',self.symform,_asym))])
        else:
            raise TypeError("data_2nd is not of a supported type: array or list.")
        # Set data. Remove the constant from the old data_2nd, add the new one.
        self._data_0th = self._data_0th - self._asym_corr + _corr
        self._asym_corr = _corr
        self._data_2nd = _symm
        # Invalidate any dependent cached properties
        self._invalidate(['is2nd','is0th','isherm','isgauss'])

    @property
    def asym_corr(self) -> npt.NDArray:
        return self._asym_corr

    @property
    def data_1st(self) -> npt.NDArray:
        return self._data_1st
    @data_1st.setter
    def data_1st(self, data):
        # Initialize array of linear-order quadrature operator coefficients.
        if data is None:
            self._data_1st = np.zeros(self.shape_1st, dtype=complex)
        elif isinstance(data, (np.ndarray, list)):
            if np.shape(data) != self.shape_1st:
                raise ValueError("Dimensions of data_1st do not agree with stored dimensions.") 
            self._data_1st = np.array(data, dtype=complex)
        else:
            raise TypeError("data_1st is not of a supported type: array or list.")
        # Invalidate any dependent cached properties
        self._invalidate(['is1st','isherm','isgauss'])
    
    @property
    def data_0th(self) -> npt.NDArray:
        return self._data_0th
    @data_0th.setter
    def data_0th(self, data):
        # Initialize array of zeroth-order quadrature operator coefficients.
        if data is None:
            self._data_0th = np.zeros(self.shape_0th, dtype=complex)  
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
        self._invalidate(['is0th','isherm','isgauss'])
    
    # Set aliases to access the data arrays that are more human-readable, 
    # along with associated getattr and setattr
    _aliases = {'data_quad': 'data_2nd', 
                'data_lin': 'data_1st', 
                'data_const': 'data_0th'}
    
    def __getattr__(self, name):
        if name in self._aliases:
            return getattr(self, self._aliases[name])
        else:
            raise AttributeError(f"QGoper has no attribute '{name}'.")

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
        # Normalise to a [rows, cols] list of ints.
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
    def is2nd(self) -> bool:
        if (QGoper._iszero(self.data_2nd) or 
            self.data_2nd.size == 0
           ):
            return False
        else:
            return True
    
    @cached_property
    def is1st(self) -> bool:
        if (QGoper._iszero(self.data_1st) or 
            self.data_1st.size == 0
           ):
            return False
        else:
            return True 
    
    @cached_property
    def is0th(self) -> bool:
        if (QGoper._iszero(self.data_0th) or 
            self.data_0th.size == 0
           ):
            return False
        else:
            return True

    @cached_property
    def isherm(self) -> bool:
        if self == self.dag():
            return True
        else:
            return False
 
    @cached_property 
    def isgauss(self) -> bool:
        # No FLS component means there is no FLS off-diagonals that could mix Gaussians.
        # This covers CVS-only (at most quadratic) operators and pure scalars.
        if not self.isfls:
            return True
        else:
            return all(self._isdiag(row, col)
                       for col in range(np.prod(self.dims_fls[1]))
                       for row in range(np.prod(self.dims_fls[0])))
    
    def _isdiag(self, row: int, col: int) -> bool:
        if row == col:
            return True
        else:
            if (QGoper._iszero(self.data_2nd[row,col]) and
                QGoper._iszero(self.data_1st[row,col]) and
                QGoper._iszero(self.data_0th[row,col])
               ):
                return True
            else:
                return False
        
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

    ### Addition and subtraction of QGopers ###
    '''
    Addition and subtraction of QGopers behave in the normal way. The other 
    object must be another QGoper of the same dimensions, or a number. If QGoper
    has an FLS component, adding a number requires that the FLS dimensions
    correspond to a square array, in which case the number is multiplied by 
    identity and added to data_0th. The exception is addition by 0, which is 
    allowed in all circumstances.
    '''

    def __add__(self, other: QGoper | complex) -> QGoper:
        # Addition with QGoper on the left
        if isinstance(other, (numbers.Number, np.number)):
            # If other is number, treat as number times an identity QGoper 
            # with same dims as self. Add zero treated separately since it is
            # valid for any shape, and this is required by sum()
            if self.isfls:
                if other == 0:
                    _const = 0
                elif self.dims_fls[0] == self.dims_fls[1]:
                    _const = other*np.eye(self.shape_0th[0])
                else:
                    raise ValueError("Cannot add a number to a QGoper with "
                                     "non-square FLS dimensions (dims_fls[0] "
                                     "!= dims_fls[1]); no identity exists.")
                return QGoper(data_2nd = self.data_2nd,
                              data_1st = self.data_1st,
                              data_0th = self.data_0th + _const,
                              dims_cvs = self.dims_cvs,
                              dims_fls = self.dims_fls)
            else:
                return QGoper(data_2nd = self.data_2nd,
                              data_1st = self.data_1st,
                              data_0th = self.data_0th + other,
                              dims_cvs = self.dims_cvs)
        elif isinstance(other, QGoper):
            # If other is a QGoper, must have sames dims as self
            if ((self.dims_cvs == other.dims_cvs) and 
                (self.dims_fls == other.dims_fls)
               ):
                return QGoper(data_2nd = self.data_2nd + other.data_2nd,
                              data_1st = self.data_1st + other.data_1st,
                              data_0th = self.data_0th + other.data_0th,
                              dims_cvs = self.dims_cvs,
                              dims_fls = self.dims_fls)
            else:
                raise ValueError("Cannot perform addition operation between " \
                "QGopers of different dimensions.")
        else:
            return NotImplemented

    def __radd__(self, other: QGoper | complex) -> QGoper:
        # Addition with QGoper on the right
        return self.__add__(other)

    def __sub__(self, other: QGoper | complex) -> QGoper:
        # Subtraction with QGoper on the left
        if isinstance(other, (numbers.Number, np.number, QGoper)):
            return self.__add__(other.__neg__())
        else:
            return NotImplemented

    def __rsub__(self, other: QGoper | complex) -> QGoper:
        # Subtraction with QGoper on the right
        return (self.__neg__()).__add__(other)

    def __neg__(self) -> QGoper:
        # Negation of QGoper
        return QGoper(data_2nd = -self.data_2nd,
                      data_1st = -self.data_1st,
                      data_0th = -self.data_0th,
                      dims_cvs = self.dims_cvs,
                      dims_fls = self.dims_fls)   

    ### Multiplication and division of QGopers ###
    '''
    Multiplication and division of a QGoper by a number is implemented. 
    Additionally, multiplcation of QGopers with identical dimensions is also 
    implemented; simplification of the quadrature component is performed when 
    constructor is called using the known commutation relation between 
    quadrature operators.
    '''

    def __mul__(self, other: QGoper | complex) -> QGoper:
        # Multiplication with QGoper on the left
        if isinstance(other, (numbers.Number, np.number)):
            # If other is a scalar, perform multiplication will all data in self
            return QGoper(data_2nd = other*self.data_2nd,
                          data_1st = other*self.data_1st,
                          data_0th = other*self.data_0th,
                          dims_cvs = self.dims_cvs,
                          dims_fls = self.dims_fls)
        elif isinstance(other, QGoper):
            # If other is a QGoper, must have sames dims as self
            if ((self.dims_cvs == other.dims_cvs) and 
                (self.dims_fls[1] == other.dims_fls[0])
               ):
                # Check that result is quadratic/bilinear order or less
                if ((self.is2nd and other.is2nd) or
                    (self.is2nd and other.is1st) or
                    (self.is1st and other.is2nd)
                   ):
                    raise ValueError(
                        "Multiplcation of QGopers produces result which is " \
                        "beyond quadratic order in quadrature operators.")
                else:
            # Multiplication using einsum: transformation depends on whether 
            # operators are FLS and/or CV:
            # 2nd order output terms combines one 2nd and one 0th order term, 
            # or two 1st order terms
            # 1st order output terms combines one 1st and one 0th order term
            # 0th order output term combines both 0th order terms
                    if self.iscvs and self.isfls:
                        return QGoper(data_2nd = \
                                      (np.einsum('jnlm,nk->jklm', self.data_2nd, other.data_0th)
                                       + np.einsum('jn,nklm->jklm', self.data_0th, other.data_2nd)
                                       + 2*np.einsum('jnl,nkm->jklm', self.data_1st, other.data_1st)),
                                      data_1st = \
                                      (np.einsum('jnl,nk->jkl', self.data_1st, other.data_0th)
                                       + np.einsum('jn,nkl->jkl', self.data_0th, other.data_1st)),
                                      data_0th = \
                                      np.einsum('jn,nk->jk', self.data_0th, other.data_0th),
                                      dims_cvs = self.dims_cvs,
                                      dims_fls = [self.dims_fls[0],other.dims_fls[1]])
                    elif self.iscvs and not self.isfls:
                        return QGoper(data_2nd = \
                                      (np.einsum('jk,k->jk', self.data_2nd, other.data_0th)
                                       + np.einsum('j,jk->jk', self.data_0th, other.data_2nd)
                                       + 2*np.einsum('j,k->jk', self.data_1st, other.data_1st)),
                                      data_1st = \
                                      (np.einsum('j,j->j', self.data_1st, other.data_0th)
                                       + np.einsum('j,j->j', self.data_0th, other.data_1st)),
                                      data_0th = \
                                      np.einsum('j,j->j', self.data_0th, other.data_0th),
                                      dims_cvs = self.dims_cvs,
                                      dims_fls = self.dims_fls)
                    elif not self.iscvs and self.isfls:
                        return QGoper(data_0th = \
                                      np.einsum('jn,nk->jk', self.data_0th, other.data_0th),
                                      dims_cvs = self.dims_cvs,
                                      dims_fls = [self.dims_fls[0],other.dims_fls[1]])
                    else:
                        return QGoper(data_0th = self.data_0th*other.data_0th,
                                      dims_cvs = self.dims_cvs,
                                      dims_fls = self.dims_fls)
            else:
                raise ValueError("Cannot perform multiplcation operation " \
                "between QGopers of different dimensions.")
        else:
            return NotImplemented

    def __rmul__(self, other: QGoper | complex) -> QGoper:
        # Multiplication with QGoper on the right
        if isinstance(other, (numbers.Number, np.number)):
            return self.__mul__(other)
        elif isinstance(other, QGoper):
            return other.__mul__(self)
        else:
            return NotImplemented

    def __truediv__(self, other: complex) -> QGoper:
        # Division of QGoper by a number
        if isinstance(other, (numbers.Number, np.number)):
            if other == 0:
                raise ZeroDivisionError
            return QGoper(data_2nd = self.data_2nd/other,
                          data_1st = self.data_1st/other,
                          data_0th = self.data_0th/other,
                          dims_cvs = self.dims_cvs,
                          dims_fls = self.dims_fls) 
        else:
            return NotImplemented

    def __pow__(self, n: int, m = None) -> QGoper:
        # Calculate powers of self.QGoper
        if ((m is not None) or
            (not isinstance(n, numbers.Integral))
           ):
            return NotImplemented
        if n < 0:
            raise ValueError("Negative powers of QGoper produces result " \
            "which is not of quadratic order in quadrature operators.")
        if not isinstance(n, (int, np.integer)):
            raise ValueError("Fractional powers of QGoper produces result " \
            "which is not of quadratic order in quadrature operators.")
        if self.dims_fls[0] != self.dims_fls[1]:
            raise TypeError("FLS component must be square to take powers.")
        if n == 0:
            if self.isfls:
                return QGoper(data_0th = np.identity(self.shape_0th[0]),
                              dims_cvs = self.dims_cvs,
                              dims_fls = self.dims_fls)
            else:
                return QGoper(data_0th = np.array([1]),
                              dims_cvs = self.dims_cvs,
                              dims_fls = self.dims_fls)
        elif n == 1:
            return QGoper(inpt = self)
        elif n == 2 and not self.is2nd:
            return self.__mul__(self)
        elif n >= 3 and not self.is2nd and not self.is1st:
            if self.isfls:
                return QGoper(data_0th = np.linalg.matrix_power(self.data_0th, n),
                              dims_cvs = self.dims_cvs,
                              dims_fls = self.dims_fls)
            else:
                return QGoper(data_0th = np.power(self.data_0th, n),
                              dims_cvs = self.dims_cvs,
                              dims_fls = self.dims_fls)
        else:
            raise ValueError("Power of QGoper produces result which is " \
            "beyond quadratic order in quadrature operators.")

    ### Assorted Methods ###  

    def __eq__(self, other: QGoper) -> bool:
        # Check equality of QGopers
        if isinstance(other, QGoper):
            same_dims = (self.dims_fls == other.dims_fls and
                         self.dims_cvs == other.dims_cvs)
            if same_dims:
                same_elems = \
                (QGoper._allclose(self.data_2nd, other.data_2nd) and
                 QGoper._allclose(self.data_1st, other.data_1st) and
                 QGoper._allclose(self.data_0th, other.data_0th))
                if same_elems: return True
                else: return False
            else: return False
        else: return False

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
                          dims_cvs = self.dims_cvs)
        else:
            raise ValueError(
                "QGoper requires an FLS and CV component to use this method. " \
                "Access QGoper data arrays individually if specific elements are required.")     

    def drop(self, *args) -> QGoper:
        # Removes CV modes specified in args from self, and return a new QGoper.
        # List, array, or tuple of indices passed as args, convert to set to
        # remove repeated indices.
        if not args:
            return QGoper(self) 
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple, set)):
            args = set(args[0])
        else:
            args = set(args)
        if min(args) < 1 or max(args) > self.dims_cvs:
            raise ValueError(f"Mode indices must be integers in the range [1,{self.dims_cvs}].")
        # Generate indices to remove from CVS part
        _ind = [n for x in sorted(args) for n in (2*x-2, 2*x-1)]

        if self.isfls:
            return QGoper(data_2nd = \
                          np.delete(np.delete(self.data_2nd, _ind, axis=3), _ind, axis=2),
                          data_1st = np.delete(self.data_1st, _ind, axis=2),
                          data_0th = self.data_0th,
                          dims_cvs = self.dims_cvs - len(args),
                          dims_fls = self.dims_fls)
        else:
            return QGoper(data_2nd = \
                          np.delete(np.delete(self.data_2nd, _ind, axis=1), _ind, axis=0),
                          data_1st = np.delete(self.data_1st, _ind, axis=0),
                          data_0th = self.data_0th,
                          dims_cvs = self.dims_cvs - len(args))
   
    def keep(self, *args) -> QGoper:
        # Keeps CV modes specified in args from self, and return a new QGoper
        # List, array, or tuple of indices passed as args, convert to set to
        # remove repeated indices.
        if not args:
            return QGoper(data_0th = self.data_0th,
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
            return QGoper(data_2nd = \
                          np.take(np.take(self.data_2nd, _ind, axis=3), _ind, axis=2),
                          data_1st = np.take(self.data_1st, _ind, axis=2),
                          data_0th = self.data_0th,
                          dims_cvs = len(args),
                          dims_fls = self.dims_fls)
        else:
            return QGoper(data_2nd = \
                          np.take(np.take(self.data_2nd, _ind, axis=1), _ind, axis=0),
                          data_1st = np.take(self.data_1st, _ind, axis=0),
                          data_0th = self.data_0th,
                          dims_cvs = len(args))

    def mode(self, *args) -> QGoper:
        # Alternate naming for the "keep" method
        return self.keep(*args)
        
    def conj(self) -> QGoper:
        # Complex-conjugate of all elements of the QGoper
        return QGoper(data_2nd = self.data_2nd.conj(),
                      data_1st = self.data_1st.conj(),
                      data_0th = self.data_0th.conj(),
                      dims_cvs = self.dims_cvs,
                      dims_fls = self.dims_fls)

    def trans(self, level = None) -> QGoper:
        # Transpose of arrays within the QGoper. Can specify the level at which
        # it is applied, either "FLS"/"fls" or "CVS"/"cvs", or the entire array
        # if neither is passed.
        if self.isfls:
            if level is None:
                return QGoper(data_2nd = self.data_2nd.transpose([1,0,3,2]),
                              data_1st = self.data_1st.transpose([1,0,2]),
                              data_0th = self.data_0th.transpose([1,0]),
                              dims_fls = [self.dims_fls[1],self.dims_fls[0]],
                              dims_cvs = self.dims_cvs)
            elif level in ('FLS', 'fls'):
                return QGoper(data_2nd = self.data_2nd.transpose([1,0,2,3]),
                              data_1st = self.data_1st.transpose([1,0,2]),
                              data_0th = self.data_0th.transpose([1,0]),
                              dims_fls = [self.dims_fls[1],self.dims_fls[0]],
                              dims_cvs = self.dims_cvs)
            elif level in ('CVS', 'cvs'):
                return QGoper(data_2nd = self.data_2nd.transpose([0,1,3,2]),
                              data_1st = self.data_1st,
                              data_0th = self.data_0th,
                              dims_fls = self.dims_fls,
                              dims_cvs = self.dims_cvs)
            else:
                raise ValueError(f"QGoper transpose passed unsuitable argument '{level}'.")
        else:
            if level in ('CVS', 'cvs') or level is None:
                return QGoper(data_2nd = self.data_2nd.T,
                              data_1st = self.data_1st,
                              data_0th = self.data_0th,
                              dims_cvs = self.dims_cvs)
            elif level in ('FLS', 'fls'):
                return QGoper(self)
            else:
                raise ValueError(f"QGoper transpose passed unsuitable argument '{level}'.")
            
    def dag(self) -> QGoper:
        # Adjoint/complex-conjugate/dagger of QGoper
        if self.isfls:
            return QGoper(data_2nd = self.data_2nd.conj().transpose([1,0,3,2]),
                          data_1st = self.data_1st.conj().transpose([1,0,2]),
                          data_0th = self.data_0th.conj().transpose([1,0]),
                          dims_fls = [self.dims_fls[1], self.dims_fls[0]],
                          dims_cvs = self.dims_cvs)
        else:
            return QGoper(data_2nd = self.data_2nd.conj().T,
                          data_1st = self.data_1st.conj(),
                          data_0th = self.data_0th.conj(),
                          dims_fls = [self.dims_fls[1], self.dims_fls[0]],
                          dims_cvs = self.dims_cvs)

    def tidyup(self, tol: float = None) -> QGoper:
        # Set real/imaginary components with magnitude below tol to zero.
        # Modifies the operator in place and returns it (allows chaining).
        if tol is None:
            tol = qgauss.settings.tidyup_atol
        for data in (self.data_2nd, self.data_1st, self.data_0th):
            data.real[np.abs(data.real) < tol] = 0
            data.imag[np.abs(data.imag) < tol] = 0
        # Cached flags may have changed now that small elements are zero.
        self._invalidate(['is2nd', 'is1st', 'is0th', 'isherm', 'isgauss'])
        return self

    @staticmethod
    def _iszero(data: npt.NDArray) -> bool:
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