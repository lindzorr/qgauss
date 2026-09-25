from __future__ import annotations

import numbers
import operator
import math
import numpy.typing as npt
from functools import cached_property
import qgauss
import numpy as np

__all__ = ['QGsuper']


class QGsuper(object):

    """
    ---- Structure ----
    A class for representing superoperators using a mixed representation for 
    systems comprised of continuous variable system (CVS) quadrature operators 
    up to quadratic/bilinear order and operators acting on finite-level systems 
    (FLS). The FLS component of the superoperator, if it exists, is vectorized 
    in the usual manner. Since the CV component of the operator does not use a 
    Fock state representation, vectorisation is not possible on this part, so 
    coefficients representing left and right multiplication of the state by the 
    quadrature operators are kept separate from each other. Left and right 
    multiplications of the density operator ρ may be represented as matrix 
    multiplcation of the vectorized state |ρ⟩⟩ as:
        · LeftMult[A](ρ) = Aρ   -->  (I ⊗ A)|ρ⟩⟩
        · RightMult[A](ρ) = ρA  -->  (A^T ⊗ I)|ρ⟩⟩
    As a result, combining left and right multiplication yields 
        AρB  -->  (B^T ⊗ A)|ρ⟩⟩. 
    The transposition operation is only applied at the FLS-level of the data 
    structures, with the CVS-level left untouched. The column-stacking procedure 
    for vectorisation used by Qutip and other packages has been used here, which 
    can visualized as follows:
        ρ = [[1,3],  -->  |ρ⟩⟩ = [1,2,3,4]^T
             [2,4]]
    The logic of storing the coefficients works similarly to that of the QGoper 
    class, with data being separated depending on whether it is quadratic/
    bilinear or linear function of the qudrature operators, or whether it is 
    altogether independent. The data structures are meant to account for left 
    and right multiplication by FLS operators, but where left and right 
    multiplcation by quadrature operators of different orders are kept separate. 
    The different data blocks therefore correspond to the following 
    combintations of quadrature operators "r_j" and finite-level operators 
    "A" and "B":
        · data_2nd_l[l,m,j,k] : ½ OL(2)_jk (r_j*r_k*A)*ρ*(B)       (B^T ⊗ A)_lm ⊗ ½ OL(2)_jk
        · data_2nd_r[l,m,j,k] : ½ OR(2)_jk (A)*ρ*(B*r_j*r_k)       (B^T ⊗ A)_lm ⊗ ½ OR(2)_jk 
        · data_2nd_m[l,m,j,k] :   OM(2)_jk (r_j*A)*ρ*(r_k*B)  -->  (B^T ⊗ A)_lm ⊗ OM(2)_jk
        · data_1st_l[l,m,j]   :   OL(1)_j  (r_j*A)*ρ*(B)           (B^T ⊗ A)_lm ⊗ OL(1)_j 
        · data_1st_r[l,m,j]   :   OR(1)_j  (A)*ρ*(r_j*B)           (B^T ⊗ A)_lm ⊗ OR(1)_j
        · data_0th[l,m]       :   O(0)     (A)*ρ*(B)               (B^T ⊗ A)_lm * O(0)   
    The continuous variable component of the superoperator is not restricted to 
    operator coefficients, but also has a Wigner representation motivated by the 
    fact that the QGsuper is equivalent to a partial differential equation (PDE)
    in the Wigner phase space. The individual components of the total density 
    operator may be converted into Wigner quasi-probability distributions (QPDs):
        ρ_T = [[ρ_00, ρ_01, ... , ρ_0M],       W_T = [[W_00(r;t), W_01(r;t), ... , W_0M(r;t)],
               [ρ_10, ρ_11, ... , ρ_1M],  -->         [W_10(r;t), W_11(r;t), ... , W_1M(r;t)],
                //...//                                //...//
               [ρ_L0, ρ_L1, ... , ρ_LM]]              [W_L0(r;t), W_L1(r;t), ... , W_LM(r;t)]]
    The moments of W_T are handled by the QGstate class. The action of the 
    superoperator on a component ρ_jk of ρ_T is equivalent to a partial 
    differential equation acting on the element W_jk(r;t) of W_T, and has the 
    generic form:
        (-G - (∂/∂r).F - r.D + ½*(∂/∂r).C.(∂/∂r) - ½*r.B.r - (∂/∂r).A.r) W_jk(r;t).
    The arrays in the above PDE are stored in the "wigner" proprties of the 
    QGsuper class, and are computed from the "data" arrays. The arrays are 
    defined analagously to the data arrays as:
        · wigner_2nd_rdr[l,m,j,k]  : A_jk (∂/∂r_j)*r_k*(S_l)*ρ*(S_m)
        · wigner_2nd_rr[l,m,j,k]   : B_jk (r_j*r_k)*(S_l)*ρ*(S_m)
        · wigner_2nd_drdr[l,m,j,k] : C_jk (∂/∂r_j)*(∂/∂r_k)*(S_l)*ρ*(S_m)
        · wigner_1st_r[l,m,j]      : D_j  r_j*(S_l)*ρ*(S_m)
        · wigner_1st_dr[l,m,j]     : F_j  (∂/∂r_j)*(S_l)*ρ*(S_m)
        · wigner_0th[l,m]          : G    (S_l)*ρ*(S_m)
    Note, due to this definition, the dynamics of W_jk(r;t) maybe depend on all 
    other components of W_T, depending on the form the superoperator. The 
    resulting superoperator may therefore not yield a Gaussian state when 
    time-evolved.

    ---- Parameters ----
    inpt : QGsuper
        Create a copy of another QGsuper.
    data_2nd_l : array_like
        Data for initialising the operator coefficients corresponding to left 
        multiplication by quadratic/bilinear-order quadrature operators.
    data_2nd_r : array_like
        Data for initialising the operator coefficients corresponding to right 
        multiplication by quadratic/bilinear-order quadrature operators.
    data_2nd_m : array_like
        Data for initialising the operator coefficients corresponding to left 
        and right multiplication by linear-order quadrature operators, resulting 
        in an overall quadratic/bilinear-order term.
    data_1st_l : array_like
        Data for initialising the operator coefficients corresponding to left 
        multiplication by linear-order quadrature operators.
    data_1st_r : array_like
        Data for initialising the operator coefficients corresponding to right 
        multiplication by linear-order quadrature operators.
    data_0th : array_like
        Data for initialising the operator coefficients corresponding to 
        multiplication by operator or constant with no dependance on the 
        quadrature operators. Purely finite-level system operators. 
    dims_cvs : int
        Total number of continuous variable cavity modes.
    dims_fls : array_like
        List of dimensions of the finite level systems, used to keep track of 
        the tensor structure.

    ---- ATtributes ----
    data_2nd_l/data_quad_left : array
        Tensor of 2D arrays containing the coefficients for the quadratic/bilinear 
        operators which multiply the density operator from the left, q_j*q_k*S_l*ρ*S_m.
    data_2nd_r/data_quad_right : array
        Tensor of 2D arrays containing the coefficients for the quadratic/bilinear 
        operators which multiply the density operator from the right, S_l*ρ*S_m*q_j*q_k.
    data_2nd_m/data_quad_jump : array
        Tensor of 2D arrays containing the coefficients for the quadrature 
        operator terms which multiply the density operator from the left and 
        right, ie with the density in the middle, q_j*S_l*ρ*S_m*q_k.
    data_1st_l/data_lin_left : array
        Tensor of 1D arrays containing the coefficients for the linear quadrature 
        operators which multiply the density operator from the left, q_j*S_l*ρ*S_m.
    data_1st_r/data_lin_right : array
        Tensor of 1D arrays containing the coefficients for the linear quadrature 
        operators which multiply the density operator from the right, S_l*ρ*S_m*q_j.
    data_0th/data_const : array
        Tensor of numbers containing the coefficients for the terms which have 
        no dependence on the quadrature operators. This component is the usual 
        vectorized supererator for finite-level systems.
    wigner_2nd_rdr : array
        Tensor of 2D arrays containing PDE coefficients which are first order 
        in the quadrature derivative, (∂/∂r_j), and first order in the 
        quadrature variables, r_k, represented as the array A_jk*S_l*ρ*S_m.
    wigner_2nd_rr : array
        Tensor of 2D arrays containing PDE coefficients which are second order 
        in the quadrature variables, r_j*r_k, represented as the array B_jk*S_l*ρ*S_m.
    wigner_2nd_drdr : array
        Tensor of 2D arrays containing PDE coefficients which are second order 
        in the quadrature derivative (∂/∂r_j)*(∂/∂r_k), represented as the array C_jk*S_l*ρ*S_m.
    wigner_1st_r : array
        Tensor of 1D arrays containing PDE coefficients which are first order 
        in the quadrature variable, r_j, represented as the array D_j*S_l*ρ*S_m.
    wigner_1st_dr : array
        Tensor of 1D arrays containing PDE coefficients which are first order 
        in the quadrature derivatives, (∂/∂r_j), represented as the array F_j*S_l*ρ*S_m.
    wigner_0th : array
        Tensor of numbers containing PDE coefficients which have no dependence 
        on quadrature derivatives or variable.
    dims_cvs : int
        Number of continuous-variable system modes.
    dims_fls : array
        List of dimensions of the finite level systems, used to keep track of 
        the tensor structure.
    shape_2nd : tuple
        Underlying shape of data_2nd_l, data_2nd_r, and data_2nd_m.
    shape_1st : tuple
        Underlying shape of data_1st_l, and data_1st_r.
    shape_0th : tuple
        Underlying shape of data_0th.
    iscvs : bool
        Does QGsuper have a CVS component.
    isfls : bool
        Does QGsuper have an FLS component.
    is2nd : bool
        Does QGsuper have a 2nd-order quadrature component.
    is1st : bool
        Does QGsuper have a 1st-order quadrature component.
    is0th : bool
        Does QGsuper have a 0th-order quadrature component.
    iscoherent : bool
        Does QGsuper correspond to coherent/unitary evolution.
    isgauss : bool
        Does the total dynamics preserve the Gaussian nature of the 
        superposition FLS-CVS state.
    issubgauss : bool
        Does the dynamics preserve the Gaussian nature of a CVS subcomponent of 
        the total FLS-CVS state. Must specify element of the QGsuper in the 
        FLS basis to check, eiher a single index to denote the row, or two 
        number indicating the position on the FLS-level of the total state. The 
        isgauss property uses this to check the preoprty for the total QGsuper.
    symform : array
        Symplectic form, for a system with N = dims_cvs, which has the form: 
            Ω = I_N ⊗ [[0,1],[-1,0]] = ⊕_{j=1}^N [[0,1],[-1,0]].

    ---- Methods ----
    add/sub : (QGsuper, QGsuper) -> QGsuper
        Returns sum/difference of two QGsupers.
    neg : QGsuper -> QGsuper
        Returns negative of QGsuper.
    mult/div : (QGsuper, complex) -> QGsuper
        Multiplication/division of QGsuper by a scaler.
    eq : (QGsuper, QGsuper) -> bool
        Check equality of two QGsupers.
    getitem : (QGsuper, list[int]) -> QGsuper (CVS)
        Extract elements of QGsuper with FLS and CVS component, to create 
        a CV-only QGsuper.
    drop : (QGsuper, int | array[int] | tuple[int]) -> QGsuper
        Remove all specified CVS modes from QGsuper. 
    keep : (QGsuper, int | array[int] | tuple[int]) -> QGsuper
        Keep only the specified CVS modes in QGsuper. 
    mode : (QGsuper, int | array[int] | tuple[int]) -> QGsuper
        Alternate naming for the "keep" method.     
    conj : QGsuper -> QGsuper
        Complex-conjugate of all elements of QGsuper.
    trans : QGsuper -> QGsuper
        Matrix transpose of all elements of QGsuper.
    dag : QGsuper -> QGsuper
        Adjoint (dagger) of QGsuper with respect to Hilbert-Schmidt inner 
        product, tr[(L[ρ])†.X] = tr[(L†[X]).ρ†]. Not equivalent to the
        sequential application of trans and conj.
    tidyup(tol) :
        Removes small elements from QGsuper below some cut-off "tol".

    """

    ### Quantum-Gaussian Superoperator (QGsuper) Initialisation ###
    def __init__(self,
                 inpt: QGsuper = None,
                 data_2nd_l: npt.ArrayLike = None,
                 data_2nd_r: npt.ArrayLike = None,
                 data_2nd_m: npt.ArrayLike = None,
                 data_1st_l: npt.ArrayLike = None,
                 data_1st_r: npt.ArrayLike = None,
                 data_0th: npt.ArrayLike | complex = None,
                 dims_cvs: int = None,
                 dims_fls: list[list[list[int]]] = None
                ):

        # QGsuper as input, copy data
        if isinstance(inpt, QGsuper):
            self._dims_cvs = inpt.dims_cvs
            self._dims_fls = [[list(x) for x in half] for half in inpt.dims_fls]

            self._asym_corr_l = inpt._asym_corr_l.copy()
            self._asym_corr_r = inpt._asym_corr_r.copy()
            self._data_0th = inpt.data_0th.copy()
            self._data_1st_l = inpt.data_1st_l.copy()
            self._data_1st_r = inpt.data_1st_r.copy()
            self._data_2nd_l = inpt.data_2nd_l.copy()
            self._data_2nd_r = inpt.data_2nd_r.copy()
            self._data_2nd_m = inpt.data_2nd_m.copy()

        # In other cases, specific components of QGsuper must be arguments.
        elif inpt is None:
             # Set dimensions of FLS and CV components using specified dimensions.
            self._dims_cvs = QGsuper._set_dims_cvs(dims_cvs)
            self._dims_fls = QGsuper._set_dims_fls(dims_fls)

            # Set data arrays from input data.
            self._asym_corr_l = np.zeros(self.shape_0th, dtype=complex)
            self._asym_corr_r = np.zeros(self.shape_0th, dtype=complex)
            self.data_0th = data_0th
            self.data_1st_l = data_1st_l
            self.data_1st_r = data_1st_r
            self.data_2nd_l = data_2nd_l
            self.data_2nd_r = data_2nd_r
            self.data_2nd_m = data_2nd_m

            if qgauss.settings.auto_tidyup is True:
                self.tidyup()

        else:
            raise TypeError("Input for constructing QGsuper is either " \
            "ill-formatted or of incorrect type.")

    '''
    ------------------
        Properties
    ------------------
    '''
       
    @property
    def data_2nd_l(self) -> npt.NDArray:
        return self._data_2nd_l
    @data_2nd_l.setter
    def data_2nd_l(self, data):
        # Initialize arrays of quadratic/bilinear-order quadrature superoperator 
        # coefficients multiplying from the left.
        if data is None:
            _symm = np.zeros(self.shape_2nd, dtype=complex)
            _corr = np.zeros(self.shape_0th, dtype=complex)
        elif isinstance(data, (np.ndarray, list)):
            if np.shape(data) != self.shape_2nd:
                raise ValueError("Dimensions of data_2nd_l do not agree with stored dimensions.")
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
            raise TypeError("data_2nd_l is not of a supported type: array or list.")
        # Set data. Remove the constant from the old data_2nd, add the new one.
        self._data_0th = self._data_0th - self._asym_corr_l + _corr
        self._asym_corr_l = _corr
        self._data_2nd_l = _symm
        # Invalidate any cached properties
        self._invalidate(self._attr_2nd + self._attr_0th + self._attr_gen)

    @property
    def asym_corr_l(self) -> npt.NDArray:
        return self._asym_corr_l
    
    @property
    def data_2nd_r(self) -> npt.NDArray:
        return self._data_2nd_r
    @data_2nd_r.setter
    def data_2nd_r(self, data):
        if data is None:
            _symm = np.zeros(self.shape_2nd, dtype=complex)
            _corr = np.zeros(self.shape_0th, dtype=complex)
        elif isinstance(data, (np.ndarray, list)):
            if np.shape(data) != self.shape_2nd:
                raise ValueError("Dimensions of data_2nd_r do not agree with stored dimensions.")
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
            raise TypeError("data_2nd_r is not of a supported type: array or list.")
        # Set data. Remove the constant from the old data_2nd, add the new one.
        self._data_0th = self._data_0th - self._asym_corr_r + _corr
        self._asym_corr_r = _corr
        self._data_2nd_r = _symm
        # Invalidate any cached properties
        self._invalidate(self._attr_2nd + self._attr_0th + self._attr_gen)

    @property
    def asym_corr_r(self) -> npt.NDArray:
        return self._asym_corr_r
    
    @property
    def data_2nd_m(self) -> npt.NDArray:
        return self._data_2nd_m
    @data_2nd_m.setter
    def data_2nd_m(self, data):
        # Initialize arrays of quadratic/bilinear-order quadrature superoperator 
        # coefficients multiplying from the left and right.
        if data is None:
            self._data_2nd_m = np.zeros(self.shape_2nd, dtype=complex)
        elif isinstance(data, (np.ndarray, list)):
            if np.shape(data) != self.shape_2nd:
                raise ValueError("Dimensions of data_2nd_m do not agree with stored dimensions.")
            self._data_2nd_m = np.array(data, dtype=complex)
        else:
            raise TypeError("data_2nd_m is not of a supported type: array or list.")
        # Invalidate any cached properties
        self._invalidate(self._attr_2nd + self._attr_0th + self._attr_gen)
    
    @property
    def data_1st_l(self) -> npt.NDArray:
        return self._data_1st_l
    @data_1st_l.setter
    def data_1st_l(self, data):
        # Initialize array of linear-order quadrature operator
        # coefficients multiplying from the left.
        if data is None:
            self._data_1st_l = np.zeros(self.shape_1st, dtype=complex)
        elif isinstance(data, (np.ndarray, list)):
            if np.shape(data) != self.shape_1st:
                raise ValueError("Dimensions of data_1st_l do not agree with stored dimensions.")
            self._data_1st_l = np.array(data, dtype=complex)
        else:
            raise TypeError("data_1st_l is not of a supported type: array or list.")
        # Invalidate any dependent cached properties
        self._invalidate(self._attr_1st + self._attr_gen)
        
    @property
    def data_1st_r(self) -> npt.NDArray:
        return self._data_1st_r
    @data_1st_r.setter
    def data_1st_r(self, data):
        # Initialize arrays of quadratic/bilinear-order quadrature superoperator 
        # coefficients multiplying from the right.
        if data is None:
            self._data_1st_r = np.zeros(self.shape_1st, dtype=complex)
        elif isinstance(data, (np.ndarray, list)):
            if np.shape(data) != self.shape_1st:
                raise ValueError("Dimensions of data_1st_r do not agree with stored dimensions.")
            self._data_1st_r = np.array(data, dtype=complex)
        else:
            raise TypeError("data_1st_r is not of a supported type: array or list.")
        # Invalidate any dependent cached properties
        self._invalidate(self._attr_1st + self._attr_gen)
        
    @property
    def data_0th(self) -> npt.NDArray:
        return self._data_0th
    @data_0th.setter
    def data_0th(self, data):
        # Initialize array of zeroth-order quadrature superoperator coefficients
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
        self._invalidate(self._attr_0th + self._attr_gen)
    
    # Set aliases to access the data arrays that are more human-readable, 
    # along with associated getattr and setattr
    _aliases = {'data_quad_left': 'data_2nd_l', 
                'data_quad_right': 'data_2nd_r', 
                'data_quad_jump': 'data_2nd_m',
                'data_lin_left': 'data_1st_l', 
                'data_lin_right': 'data_1st_r', 
                'data_const': 'data_0th',
                'drift_mat': 'wigner_2nd_rdr',
                'dyn_mat': 'wigner_2nd_rdr',
                'disp_mat': 'wigner_2nd_rr',
                'diff_mat': 'wigner_2nd_drdr',
                'noise_mat': 'wigner_2nd_drdr',
                'longit_vec': 'wigner_1st_r',
                'drive_vec': 'wigner_1st_dr',
                'source_term': 'wigner_0th'
                }
    
    def __getattr__(self, name):
        if name in self._aliases:
            return getattr(self, self._aliases[name])
        else:
            raise AttributeError(f"QGsuper has no attribute '{name}'.")

    def __setattr__(self, name, data):
        if name in self._aliases:
            return setattr(self, self._aliases[name], data)
        else:
            super().__setattr__(name, data)

    @cached_property
    def wigner_2nd_rdr(self) -> npt.NDArray:
        # Drift matrix, system dynamics matrix
        if self.isfls:
            _data = ((1/2)*(self.data_2nd_l + self.data_2nd_l.transpose([0,1,3,2]))
                     - (1/2)*(self.data_2nd_r + self.data_2nd_r.transpose([0,1,3,2]))
                     + (self.data_2nd_m - self.data_2nd_m.transpose([0,1,3,2])))
            return np.einsum("jk,lmkn->lmjn",
                             (1j/2)*self.symform,
                             _data)
        else:
            _data = ((1/2)*(self.data_2nd_l + self.data_2nd_l.T)
                     - (1/2)*(self.data_2nd_r + self.data_2nd_r.T)
                     + (self.data_2nd_m - self.data_2nd_m.T))
            return (1j/2)*self.symform @ _data
    
    @cached_property
    def wigner_2nd_rr(self) -> npt.NDArray:
        # Riccati coupling/feedback matrix, dispersive coupling matrix
        if self.isfls:
            return (-(1/2)*(self.data_2nd_l + self.data_2nd_l.transpose([0,1,3,2]))
                    -(1/2)*(self.data_2nd_r + self.data_2nd_r.transpose([0,1,3,2]))
                    -(self.data_2nd_m + self.data_2nd_m.transpose([0,1,3,2])))
        else:
            return (-(1/2)*(self.data_2nd_l + self.data_2nd_l.T)
                    -(1/2)*(self.data_2nd_r + self.data_2nd_r.T)
                    -(self.data_2nd_m + self.data_2nd_m.T))
        
    @cached_property
    def wigner_2nd_drdr(self) -> npt.NDArray:
        # Diffusion matrix, noise matrix
        if self.isfls:
            _data = ((1/2)*(self.data_2nd_l + self.data_2nd_l.transpose([0,1,3,2]))
                     + (1/2)*(self.data_2nd_r + self.data_2nd_r.transpose([0,1,3,2]))
                     - (self.data_2nd_m + self.data_2nd_m.transpose([0,1,3,2])))
            return np.einsum("jk,lmkn,np->lmjp",
                             (1/4)*self.symform,
                             _data,
                             self.symform)
        else:
            _data = ((1/2)*(self.data_2nd_l + self.data_2nd_l.T)
                     + (1/2)*(self.data_2nd_r + self.data_2nd_r.T)
                     - (self.data_2nd_m + self.data_2nd_m.T))
            return (1/4)*self.symform @ _data @ self.symform
    
    @cached_property
    def wigner_1st_r(self) -> npt.NDArray:
        # Restoring‑force vector, linear-damping vector, longitudinal vector
        return -(self.data_1st_l + self.data_1st_r)

    @cached_property
    def wigner_1st_dr(self) -> npt.NDArray:
        # Forcing vector, driving vector, displacement vector
        _data = self.data_1st_l - self.data_1st_r 
        if self.isfls:  
            return np.einsum("jk,lmk->lmj",
                             (1j/2)*self.symform,
                             _data)
        else:   
            return (1j/2)*self.symform @ _data                                  
        
    @cached_property
    def wigner_0th(self) -> npt.NDArray:
        # Decay term, source/sink term, decay rate of the state norm, loss rate
        _data = (1/2)*(self.data_2nd_l + self.data_2nd_r) - self.data_2nd_m
        if self.isfls:
            return (-self.data_0th 
                    + (1j/2)*np.einsum("jk,lmkj->lm",
                                       self.symform,
                                       _data))
        else:
            return (-self.data_0th 
                    + (1j/2)*np.array(np.trace(self.symform @ _data)))
                
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
    def dims_fls(self) -> list[list[list[int]]]:
        return self._dims_fls
    @staticmethod
    def _set_dims_fls(dims) -> list[list[list[int]]]:
        # Normalise to a [[R_rows,R_cols], [C_rows,C_cols]] list of ints.
        if dims is None:
            return [[[],[]],[[],[]]]
        try:
            [[R_rows,R_cols], [C_rows,C_cols]] = dims
            R_rows = [operator.index(x) for x in R_rows]
            R_cols = [operator.index(x) for x in R_cols]
            C_rows = [operator.index(x) for x in C_rows]
            C_cols = [operator.index(x) for x in C_cols]
        except (TypeError, ValueError):
            raise TypeError("dims_fls must have the form " \
                            "[[R_rows,R_cols], [C_rows,C_cols]] " \
                            "with integer entries.") from None
        if not R_rows and not R_cols and not C_rows and not C_cols:
            return [[[],[]],[[],[]]]
        if not R_rows or not R_cols or not C_rows or not C_cols:
            raise ValueError("dims_fls row and column dims must both be empty "
                             "or both be non-empty.")
        if min(R_rows + R_cols + C_rows + C_cols) < 1:
            raise ValueError("dims_fls entries must be positive integers.")
        return [[R_rows,R_cols], [C_rows,C_cols]]

    @property
    def shape_2nd(self) -> tuple[int,int,int,int] | tuple[int,int]:
        if self.isfls:
            return (math.prod(self.dims_fls[0][0]) * math.prod(self.dims_fls[0][1]),
                    math.prod(self.dims_fls[1][0]) * math.prod(self.dims_fls[1][1]),
                    2*self.dims_cvs, 
                    2*self.dims_cvs)
        else:
            return (2*self.dims_cvs, 
                    2*self.dims_cvs)
        
    @property
    def shape_1st(self) -> tuple[int,int,int] | tuple[int]:
        if self.isfls:
            return (math.prod(self.dims_fls[0][0]) * math.prod(self.dims_fls[0][1]),
                    math.prod(self.dims_fls[1][0]) * math.prod(self.dims_fls[1][1]),
                    2*self.dims_cvs)
        else:
            return (2*self.dims_cvs,)
        
    @property
    def shape_0th(self) -> tuple[int,int] | tuple[int]:
        if self.isfls:
            return (math.prod(self.dims_fls[0][0]) * math.prod(self.dims_fls[0][1]),
                    math.prod(self.dims_fls[1][0]) * math.prod(self.dims_fls[1][1]))
        else:
            return (1,)
    
    @cached_property
    def iscvs(self) -> bool:
        return self._dims_cvs > 0
    
    @cached_property
    def isfls(self) -> bool:
        return (len(self._dims_fls[0][0]) > 0 and 
                len(self._dims_fls[0][1]) > 0 and
                len(self._dims_fls[1][0]) > 0 and 
                len(self._dims_fls[1][1]) > 0)
 
    @cached_property
    def is2nd(self) -> bool:
        if ((QGsuper._iszero(self.data_2nd_l) and
             QGsuper._iszero(self.data_2nd_r) and
             QGsuper._iszero(self.data_2nd_m))
             or
            (self.data_2nd_l.size == 0 and
             self.data_2nd_r.size == 0 and
             self.data_2nd_m.size == 0)
           ):
            return False
        else:
            return True
    
    @cached_property
    def is1st(self) -> bool:
        if ((QGsuper._iszero(self.data_1st_l) and
             QGsuper._iszero(self.data_1st_r))
             or
            (self.data_1st_l.size == 0 and
             self.data_1st_r.size == 0)
           ):
            return False
        else:
            return True
    
    @cached_property
    def is0th(self) -> bool:
        if (QGsuper._iszero(self.data_0th) or
            self.data_0th.size == 0
           ):
            return False
        else:
            return True

    @cached_property
    def iscoherent(self) -> bool:
        if self == -self.dag():
            return True
        else:
            return False
        
    @cached_property
    def isgauss(self) -> bool: 
        if not self.isfls:
            return True
        else:
            return all([self.issubgauss(j) 
                        for j in range(math.prod(self.dims_fls[0][0])
                                       *math.prod(self.dims_fls[0][1]))])

    def issubgauss(self, row: int, col: int = None) -> bool:
        """ Checks that the dynamics of a subcomponent of the QGsuper is
        Gaussian by ensuring that there is no coupling to other elements of the 
        qubit-density operator. If only row is specified, then the row in the 
        vectorised superoperator is to be checked. If row and col are specified, 
        then the dynamics acting on the [row,col] component of a QGstate is to 
        be checked. """
        rank = math.prod(self.dims_fls[1][0]) * math.prod(self.dims_fls[1][1])
        if col is None:
            j = row
        else:
            m = math.prod(self.dims_fls[0][0])
            j = m*col + row

        if (all([QGsuper._iszero(self.data_2nd_l[j,k]) for k in range(rank) if k != j]) and
            all([QGsuper._iszero(self.data_2nd_r[j,k]) for k in range(rank) if k != j]) and
            all([QGsuper._iszero(self.data_2nd_m[j,k]) for k in range(rank) if k != j]) and
            all([QGsuper._iszero(self.data_1st_l[j,k]) for k in range(rank) if k != j]) and
            all([QGsuper._iszero(self.data_1st_r[j,k]) for k in range(rank) if k != j]) and
            all([QGsuper._iszero(self.data_0th[j,k]) for k in range(rank) if k != j])
           ):
            return True
        else:
            return False

    @cached_property
    def symform(self) -> npt.NDArray:
        return np.kron(np.identity(self.dims_cvs), np.array([[0,1],[-1,0]]))
    
    # Set lists of attribute names that need to be invalidated together
    # when data arrays are updated.
    _attr_2nd = ['is2nd','wigner_2nd_rdr','wigner_2nd_rr','wigner_2nd_drdr']
    _attr_1st = ['is1st','wigner_1st_dr','wigner_1st_r']
    _attr_0th = ['is0th','wigner_0th']
    _attr_gen = ['iscoherent','isgauss']

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

    ### Addition and subtraction of QGsupers ###

    def __add__(self, other: QGsuper) -> QGsuper:
        # Addition with self.QGsuper on the left
        if isinstance(other, QGsuper):
            if ((self.dims_cvs == other.dims_cvs) and 
                (self.dims_fls == other.dims_fls)
                ):
                return QGsuper(data_2nd_l = self.data_2nd_l + other.data_2nd_l,
                               data_2nd_r = self.data_2nd_r + other.data_2nd_r,
                               data_2nd_m = self.data_2nd_m + other.data_2nd_m,
                               data_1st_l = self.data_1st_l + other.data_1st_l,
                               data_1st_r = self.data_1st_r + other.data_1st_r,
                               data_0th = self.data_0th + other.data_0th,
                               dims_cvs = self.dims_cvs,
                               dims_fls = self.dims_fls)
            else:
                raise ValueError("Cannot perform addition operation between " \
                "QGsupers with different dimensions.")
        elif isinstance(other, (numbers.Number, np.number)) and other == 0:
            return QGsuper(self)
        else:
            return NotImplemented

    def __radd__(self, other: QGsuper) -> QGsuper:
        # Addition with the self.QGsuper on the right
        return self.__add__(other)

    def __sub__(self, other: QGsuper) -> QGsuper:
        # Subtraction with self.QGsuper on the left
        if isinstance(other, (QGsuper, numbers.Number, np.number)):
            return self.__add__(other.__neg__())
        else:
            return NotImplemented

    def __rsub__(self, other: QGsuper) -> QGsuper:
        # Subtraction with self.QGsuper on the right
        return (self.__neg__()).__add__(other)

    def __neg__(self) -> QGsuper:
        # Negation of self.QGoper
        return QGsuper(data_2nd_l = -self.data_2nd_l,
                       data_2nd_r = -self.data_2nd_r,
                       data_2nd_m = -self.data_2nd_m,
                       data_1st_l = -self.data_1st_l,
                       data_1st_r = -self.data_1st_r,
                       data_0th = -self.data_0th,
                       dims_cvs = self.dims_cvs,
                       dims_fls = self.dims_fls)

    ### Multiplication and division of QGosupers ###

    def __mul__(self, other: complex) -> QGsuper:
        # Multiplication of number with self.QGsuper on the left
        if isinstance(other, (numbers.Number, np.number)):
            return QGsuper(data_2nd_l = other*self.data_2nd_l,
                           data_2nd_r = other*self.data_2nd_r,
                           data_2nd_m = other*self.data_2nd_m,
                           data_1st_l = other*self.data_1st_l,
                           data_1st_r = other*self.data_1st_r,
                           data_0th = other*self.data_0th,
                           dims_cvs = self.dims_cvs,
                           dims_fls = self.dims_fls)
        else:
            return NotImplemented

    def __rmul__(self, other: complex) -> QGsuper:
        # Multiplication with self.QGsuper on the right
        if isinstance(other, (numbers.Number, np.number)):
            return self.__mul__(other)
        else:
            return NotImplemented

    def __truediv__(self, other: complex) -> QGsuper:
        # Division of self.QGsuper by number
        if isinstance(other, (numbers.Number,np.number)):
            if other == 0:
                raise ZeroDivisionError
            return QGsuper(data_2nd_l = self.data_2nd_l/other,
                           data_2nd_r = self.data_2nd_r/other,
                           data_2nd_m = self.data_2nd_m/other,
                           data_1st_l = self.data_1st_l/other,
                           data_1st_r = self.data_1st_r/other,
                           data_0th = self.data_0th/other,
                           dims_cvs = self.dims_cvs,
                           dims_fls = self.dims_fls)
        else:
            return NotImplemented

    ### Assorted Methods ###

    def __eq__(self, other: QGsuper) -> bool:
        # Check equality of QGsupers #
        if isinstance(other, QGsuper):
            same_dims = (self.dims_fls == other.dims_fls and
                         self.dims_cvs == other.dims_cvs)
            if same_dims:
                same_elems = \
                (QGsuper._allclose(self.data_2nd_l, other.data_2nd_l) and
                 QGsuper._allclose(self.data_2nd_r, other.data_2nd_r) and
                 QGsuper._allclose(self.data_2nd_m, other.data_2nd_m) and
                 QGsuper._allclose(self.data_1st_l, other.data_1st_l) and
                 QGsuper._allclose(self.data_1st_r, other.data_1st_r) and
                 QGsuper._allclose(self.data_0th, other.data_0th))
                if same_elems: return True
                else: return False
            else: return False
        else: return False

    def __getitem__(self, index) -> QGsuper:
        # Grab CV elements from self at index in the FLS component
        # and return a QGsuper with a CV component only
        if self.isfls and self.iscvs:
            return QGsuper(data_2nd_l = self.data_2nd_l[index],
                           data_2nd_r = self.data_2nd_r[index],
                           data_2nd_m = self.data_2nd_m[index],
                           data_1st_l = self.data_1st_l[index],
                           data_1st_r = self.data_1st_r[index],
                           data_0th = self.data_0th[index],
                           dims_cvs = self.dims_cvs)
        else:
            raise ValueError("QGsuper requires an FLS and CV component to " \
            "use this method. Access QGsuper data arrays individually if " \
            "specific elements are required.")

    def drop(self, *args) -> QGsuper:
        # Removes CV modes specified in args from self, and return a new QGsuper.
        # List, array, or tuple of indices passed as args, convert to set to
        # remove repeated indices.
        if not args:
            return QGsuper(self) 
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple, set)):
            args = set(args[0])
        else:
            args = set(args)
        if min(args) < 1 or max(args) > self.dims_cvs:
            raise ValueError(f"Mode indices must be integers in the range [1,{self.dims_cvs}].")
        # Generate indices to remove from CVS part
        _ind = [n for x in sorted(args) for n in (2*x-2, 2*x-1)]

        if self.isfls:
            return QGsuper(data_2nd_l = \
                           np.delete(np.delete(self.data_2nd_l, _ind, axis=3), _ind, axis=2),
                           data_2nd_r = \
                           np.delete(np.delete(self.data_2nd_r, _ind, axis=3), _ind, axis=2),
                           data_2nd_m = \
                           np.delete(np.delete(self.data_2nd_m, _ind, axis=3), _ind, axis=2),
                           data_1st_l = np.delete(self.data_1st_l, _ind, axis=2),
                           data_1st_r = np.delete(self.data_1st_r, _ind, axis=2),
                           data_0th = self.data_0th,
                           dims_fls = self.dims_fls,
                           dims_cvs = self.dims_cvs - len(args))
        else:
            return QGsuper(data_2nd_l = \
                           np.delete(np.delete(self.data_2nd_l, _ind, axis=1), _ind, axis=0),
                           data_2nd_r = \
                           np.delete(np.delete(self.data_2nd_r, _ind, axis=1), _ind, axis=0),
                           data_2nd_m = \
                           np.delete(np.delete(self.data_2nd_m, _ind, axis=1), _ind, axis=0),
                           data_1st_l = np.delete(self.data_1st_l, _ind, axis=0),
                           data_1st_r = np.delete(self.data_1st_r, _ind, axis=0),
                           data_0th = self.data_0th,
                           dims_cvs = self.dims_cvs - len(args))
    
    def keep(self, *args) -> QGsuper:
        # Keeps CV modes specified in args from self, and return a new QGsuper.
        # List, array, or tuple of indices passed as args, convert to set to
        # remove repeated indices.
        if not args:
            return QGsuper(data_0th = self.data_0th,
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
            return QGsuper(data_2nd_l = \
                           np.take(np.take(self.data_2nd_l, _ind, axis=3), _ind, axis=2),
                           data_2nd_r = \
                           np.take(np.take(self.data_2nd_r, _ind, axis=3), _ind, axis=2),
                           data_2nd_m = \
                           np.take(np.take(self.data_2nd_m, _ind, axis=3), _ind, axis=2),
                           data_1st_l = np.take(self.data_1st_l, _ind, axis=2),
                           data_1st_r = np.take(self.data_1st_r, _ind, axis=2),
                           data_0th = self.data_0th,
                           dims_cvs = len(args),
                           dims_fls = self.dims_fls)
        else:
            return QGsuper(data_2nd_l = \
                           np.take(np.take(self.data_2nd_l, _ind, axis=1), _ind, axis=0),
                           data_2nd_r = \
                           np.take(np.take(self.data_2nd_r, _ind, axis=1), _ind, axis=0),
                           data_2nd_m = \
                           np.take(np.take(self.data_2nd_m, _ind, axis=1), _ind, axis=0),
                           data_1st_l = np.take(self.data_1st_l, _ind, axis=0),
                           data_1st_r = np.take(self.data_1st_r, _ind, axis=0),
                           data_0th = self.data_0th,
                           dims_cvs = len(args))

    def mode(self, *args) -> QGsuper:
        # Alternate naming for the "keep" method
        return self.keep(*args)
    
    def conj(self) -> QGsuper:
        # Complex-conjugate of all elements of the QGsuper
        return QGsuper(data_2nd_l = self.data_2nd_l.conj(),
                       data_2nd_r = self.data_2nd_r.conj(),
                       data_2nd_m = self.data_2nd_m.conj(),
                       data_1st_l = self.data_1st_l.conj(),
                       data_1st_r = self.data_1st_r.conj(),
                       data_0th = self.data_0th.conj(),
                       dims_cvs = self.dims_cvs,
                       dims_fls = self.dims_fls)

    def trans(self, level = None) -> QGsuper:
        # Transpose of arrays within the QGsuper. Can specify the level at 
        # which it is applied, either "FLS"/"fls" or "CVS"/"cvs", or the entire
        # array if none are passed. 
        if self.isfls:
            if level is None:
                return QGsuper(data_2nd_l = self.data_2nd_l.transpose([1,0,3,2]),
                               data_2nd_r = self.data_2nd_r.transpose([1,0,3,2]),
                               data_2nd_m = self.data_2nd_m.transpose([1,0,3,2]),
                               data_1st_l = self.data_1st_l.transpose([1,0,2]),
                               data_1st_r = self.data_1st_r.transpose([1,0,2]),
                               data_0th = self.data_0th.transpose([1,0]),
                               dims_fls = [self.dims_fls[1], self.dims_fls[0]],
                               dims_cvs = self.dims_cvs)
            elif level in ('FLS', 'fls'):
                return QGsuper(data_2nd_l = self.data_2nd_l.transpose([1,0,2,3]),
                               data_2nd_r = self.data_2nd_r.transpose([1,0,2,3]),
                               data_2nd_m = self.data_2nd_m.transpose([1,0,2,3]),
                               data_1st_l = self.data_1st_l.transpose([1,0,2]),
                               data_1st_r = self.data_1st_r.transpose([1,0,2]),
                               data_0th = self.data_0th.transpose([1,0]),
                               dims_fls = [self.dims_fls[1], self.dims_fls[0]],
                               dims_cvs = self.dims_cvs)
            elif level in ('CVS', 'cvs'):
                return QGsuper(data_2nd_l = self.data_2nd_l.transpose([0,1,3,2]),
                               data_2nd_r = self.data_2nd_r.transpose([0,1,3,2]),
                               data_2nd_m = self.data_2nd_m.transpose([0,1,3,2]),
                               data_1st_l = self.data_1st_l,
                               data_1st_r = self.data_1st_r,
                               data_0th = self.data_0th,
                               dims_fls = self.dims_fls,
                               dims_cvs = self.dims_cvs)
            else:
                raise ValueError(f"QGsuper transpose passed unsuitable argument '{level}'.")
        else:
            if level in ('CVS', 'cvs') or level is None:
                return QGsuper(data_2nd_l = self.data_2nd_l.T,
                               data_2nd_r = self.data_2nd_r.T,
                               data_2nd_m = self.data_2nd_m.T,
                               data_1st_l = self.data_1st_l.T,
                               data_1st_r = self.data_1st_r.T,
                               data_0th = self.data_0th.T,
                               dims_cvs = self.dims_cvs)
            elif level in ('FLS', 'fls'):
                return self
            else:
                raise ValueError(f"QGsuper transpose passed unsuitable argument '{level}'.")

    def dag(self) -> QGsuper:
        # Adjoint/dagger of QGsuper with respect to Hilbert-Schmidt inner 
        # product: tr[(L[ρ])†.X] = tr[(L†[X]).ρ†].
        # For right and left multiplication by some operators A and B, the
        # dagger is then: tr[(A.ρ.B)†.X] = tr[(A†.X.B†).ρ†].
        # Note: this operation is not equal to the application of trans and conj
        # since the jump-terms are not transposed at the CVS level.
        #   L.dag() != L.trans().conj() == L.conj().trans()
        if self.isfls:
            return QGsuper(data_2nd_l = self.data_2nd_l.conj().transpose([1,0,3,2]),
                           data_2nd_r = self.data_2nd_r.conj().transpose([1,0,3,2]),
                           data_2nd_m = self.data_2nd_m.conj().transpose([1,0,2,3]),
                           data_1st_l = self.data_1st_l.conj().transpose([1,0,2]),
                           data_1st_r = self.data_1st_r.conj().transpose([1,0,2]),
                           data_0th = self.data_0th.conj().transpose([1,0]),
                           dims_fls = [self.dims_fls[1], self.dims_fls[0]],
                           dims_cvs = self.dims_cvs)
        else:
            return QGsuper(data_2nd_l = self.data_2nd_l.conj().T,
                           data_2nd_r = self.data_2nd_r.conj().T,
                           data_2nd_m = self.data_2nd_m.conj(),
                           data_1st_l = self.data_1st_l.conj().T,
                           data_1st_r = self.data_1st_r.conj().T,
                           data_0th = self.data_0th.conj(),
                           dims_cvs = self.dims_cvs)

    def tidyup(self, tol: float = None) -> QGsuper:
        # Set real/imaginary components with magnitude below tol to zero.
        # Modifies the operator in place and returns it (allows chaining).
        if tol is None:
            tol = qgauss.settings.tidyup_atol
        for data in (self.data_2nd_l, self.data_2nd_r, self.data_2nd_m,
                     self.data_1st_l, self.data_1st_r, self.data_0th):
            data.real[np.abs(data.real) < tol] = 0
            data.imag[np.abs(data.imag) < tol] = 0
        # Cached flags may have changed now that small elements are zero.
        self._invalidate(self._attr_2nd + self._attr_1st
                         + self._attr_0th + self._attr_gen)
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