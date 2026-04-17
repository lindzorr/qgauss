from __future__ import annotations

import numbers
import numpy.typing as npt
from functools import cached_property
import qgauss
import numpy as np

__all__ = ['QGsuper']


class QGsuper(object):

    """
    ---- Structure ----
    A class for representing superoperators using a mixed representation for systems comprised of continuous variable 
    (CV) quadrature operators up to bilinear order and operators acting on finite-level systems (FLS). The FLS 
    component of the superoperator, if it exists, is vectorized in the usual manner. Since the CV component of the 
    operator does not use a Fock state representation, vectorisation is not possible on this part, so coefficients 
    representing left and right multiplication of the state by the quadrature operators are kept separate from each 
    other. Left and right multiplications of the density operator ρ may be represented as matrix multiplcation of the 
    vectorized state |ρ⟩⟩ as:
        · L[A](ρ) = Aρ  -->  (I ⊗ A)|ρ⟩⟩
        · R[A](ρ) = ρA  -->  (A^T ⊗ I)|ρ⟩⟩
    As a result, combining left and right multiplication yields AρB  -->  (B^T ⊗ A)|ρ⟩⟩. The transposition operation is 
    only applied at the FLS-level of the data structures, with the CV-level left untouched. The column-stacking 
    procedure for vectorisation used by Qutip and other packages has been used here, which can visualized as follows:
        ρ = [[1,3],  -->  |ρ⟩⟩ = [1,2,3,4]^T
             [2,4]]
    The logic of storing the coefficients works similarly to that of the QGoper class, with data being separated 
    depending on whether it is bilinear or linear function of the qudrature operators, or whether it is altogether
    independent. The data structures are meant to account for left and right multiplication by FLS operators, but 
    where left and right multiplcation by quadrature operators of different orders are kept separate. The different 
    data blocks therefore correspond to the following combintations of quadrature operators "r_j" and finite-level 
    operators "A" and "B":
        · data_2nd_l[l,m,j,k] : ½ OL(2)_jk (r_j*r_k*A)*ρ*(B)       (B^T ⊗ A)_lm ⊗ ½ OL(2)_jk
        · data_2nd_r[l,m,j,k] : ½ OR(2)_jk (A)*ρ*(B*r_j*r_k)       (B^T ⊗ A)_lm ⊗ ½ OL(2)_jk 
        · data_2nd_m[l,m,j,k] :   OM(2)_jk (r_j*A)*ρ*(r_k*B)  -->  (B^T ⊗ A)_lm ⊗ OM(2)_jk
        · data_1st_l[l,m,j]   :   OL(1)_j  (r_j*A)*ρ*(B)           (B^T ⊗ A)_lm ⊗ OL(1)_j 
        · data_1st_r[l,m,j]   :   OR(1)_j  (A)*ρ*(r_j*B)           (B^T ⊗ A)_lm ⊗ OR(1)_j
        · data_0th[l,m]       :   O(0)     (A)*ρ*(B)               (B^T ⊗ A)_lm * O(0)   
    The continuous variable component of the superoperator is not restricted to operator coefficients, but also has a
    Wigner representation motivated by the fact that the QGsuper is equivalent to a partial differential equation (PDE)
    in the Wigner phase space. The individual components of the total density operator may be converted into Wigner 
    quasi-probability distributions (QPDs):
        ρ_T = [[ρ_00, ρ_01, ... , ρ_0M],       W_T = [[W_00(r;t), W_01(r;t), ... , W_0M(r;t)],
               [ρ_10, ρ_11, ... , ρ_1M],  -->         [W_10(r;t), W_11(r;t), ... , W_1M(r;t)],
                //...//                                //...//
               [ρ_L0, ρ_L1, ... , ρ_LM]]              [W_L0(r;t), W_L1(r;t), ... , W_LM(r;t)]]
    The moments of W_T are handled by the QGstate class. The action of the superoperator on a component ρ_jk of ρ_T is 
    equivalent to a partial differential equation acting on the element W_jk(r;t) of W_T, and has the generic form:
        (-G - (∂/∂r).F - r.D + ½*(∂/∂r).C.(∂/∂r) - ½*r.B.r - (∂/∂r).A.r) W_jk(r;t).
    The arrays in the above PDE are stored in the "wigner" proprties of the QGsuper class, and are computed from the 
    "data" arrays. The arrays are defined analagously to the data arrays as:
        · wigner_2nd_deriv_var[l,m,j,k] : A_jk (∂/∂r_j)*r_k*(S_l)*ρ*(S_m)
        · wigner_2nd_var[l,m,j,k]       : B_jk (r_j*r_k)*(S_l)*ρ*(S_m)
        · wigner_2nd_deriv[l,m,j,k]     : C_jk (∂/∂r_j)*(∂/∂r_k)*(S_l)*ρ*(S_m)
        · wigner_1st_var[l,m,j]         : D_j  r_j*(S_l)*ρ*(S_m)
        · wigner_1st_deriv[l,m,j]       : F_j  (∂/∂r_j)*(S_l)*ρ*(S_m)
        · wigner_0th[l,m]               : G    (S_l)*ρ*(S_m)
    Note, due to this definition, the dynamics of W_jk(r;t) maybe depend on all other components of W_T, depending on
    the form the superoperator. The resulting superoperator may therefore not yield a Gaussian state when time-evolved.

    ---- Parameters ----
    inpt : QGsuper
        Create a copy of another QGsuper.
    data_2nd_l : array_like
        Data for initialising the operator coefficients corresponding to left multiplication 
        by bilinear-order quadrature operators.
    data_2nd_r : array_like
        Data for initialising the operator coefficients corresponding to right multiplication 
        by bilinear-order quadrature operators.
    data_2nd_m : array_like
        Data for initialising the operator coefficients corresponding to left and right multiplication 
        by linear-order quadrature operators, resulting in an overall bilinear-order term.
    data_1st_l : array_like
        Data for initialising the operator coefficients corresponding to left multiplication 
        by linear-order quadrature operators.
    data_1st_r : array_like
        Data for initialising the operator coefficients corresponding to right multiplication 
        by linear-order quadrature operators.
    data_0th : array_like
        Data for initialising the operator coefficients corresponding to multiplication by operator or
        constant with no dependance on the quadrature operators. Purely finite-level system operators. 
    dims_cvs : int
        Total number of continuous variable cavity modes.
    dims_fls : array_like
        List of dimensions of the finite level systems, used to keep track of the tensor structure.

    ---- ATtributes ----
    data_2nd_l : array
        Tensor of 2D arrays containing the coefficients for the bilinear operators which multiply
        the density operator from the left, q_j*q_k*S_l*ρ*S_m.
    data_2nd_r : array
        Tensor of 2D arrays containing the coefficients for the bilinear operators which multiply
        the density operator from the right, S_l*ρ*S_m*q_j*q_k.
    data_2nd_m : array
        Tensor of 2D arrays containing the coefficients for the quadrature operator terms which multiply
        the density operator from the left and right, ie with the density in the middle, q_j*S_l*ρ*S_m*q_k.
    data_1st_l : array
        Tensor of 1D arrays containing the coefficients for the linear quadrature operators which multiply
        the density operator from the left, q_j*S_l*ρ*S_m.
    data_1st_r : array
        Tensor of 1D arrays containing the coefficients for the linear quadrature operators which multiply
        the density operator from the right, S_l*ρ*S_m*q_j.
    data_0th : array
        Tensor of numbers containing the coefficients for the terms which have no dependence on the
        quadrature operators. This component is the usual vectorized supererator for finite-level systems.
    wigner_2nd_deriv_Var : array
        Tensor of 2D arrays containing PDE coefficients which are first order in the quadrature derivative, (∂/∂r_j), 
        and first order in the quadrature variables, r_k, represented as the array A_jk*S_l*ρ*S_m.
    wigner_2nd_var : array
        Tensor of 2D arrays containing PDE coefficients which are second order in the quadrature variables, r_j*r_k,
        represented as the array B_jk*S_l*ρ*S_m.
    wigner_2nd_deriv : array
        Tensor of 2D arrays containing PDE coefficients which are second order in the quadrature derivative 
        (∂/∂r_j)*(∂/∂r_k), represented as the array C_jk*S_l*ρ*S_m.
    wigner_1st_var : array
        Tensor of 1D arrays containing PDE coefficients which are first order in the quadrature variable, r_j,
        represented as the array D_j*S_l*ρ*S_m.
    wigner_1st_r : array
        Tensor of 1D arrays containing PDE coefficients which are first order in the quadrature derivatives, (∂/∂r_j),
        represented as the array F_j*S_l*ρ*S_m.
    wigner_0th : array
        Tensor of numbers containing PDE coefficients which have no dependence on quadrature derivatives or variable.
    dims_cvs : int
        Number of continuous-variable system modes.
    dims_fls : array
        List of dimensions of the finite level systems, used to keep track of the tensor structure.
    shape_2nd : tuple
        Underlying shape of data_2nd_l, data_2nd_r, and data_2nd_m.
    shape_1st : tuple
        Underlying shape of data_1st_l, and data_1st_r.
    shape_0th : tuple
        Underlying shape of data_0th.
    iscvs : bool
        Does QGsuper have a CV component.
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
        Does the total dynamics preserve the Gaussian nature the superposition FLS-CVS state.
    issubgauss : bool
        Does the dynamics preserve the Gaussian nature of a CVS subcomponent of the total FLS-CVS state. Must specify 
        element of the QGsuper in the FLS basis to check, eiher a single index to denote the row, or two number
        indicating the position on the FLS-level of the total state. The isgauss property uses this to check the 
        preoprty for the total QGsuper.
    symform : array
        Symplectic form, for a system with N = dims_cvs this has the form: Ω = ⊗_{j=1}^N [[0,1],[-1,0]].

    ---- Methods ----
    add/sub : (QGsuper, QGsuper) -> QGsuper
        Returns sum/difference of two QGsupers.
    neg : QGsuper -> QGsuper
        Returns negative of QGsuper.
    mult/div : (QGsuper, complex) -> QGsuper
        Multiplication/division of QGsuper by a scaler.
    eq : (QGsuper, QGsuper) -> bool
        Check equality of two QGsupers.
    getitem : (QGsuper, list[int]) -> QGsuper (CV)
        Extract elements of QGsuper with FLS and CV component, to create a CV-only QGsuper.
    drop : (QGsuper, int | array[int] | tuple[int]) -> QGsuper
        Remove all specified CV modes from QGoper. 
    keep : (QGsuper, int | array[int] | tuple[int]) -> QGsuper
        Keep only the specified CV modes in QGsuper.     
    conj : QGsuper -> QGsuper
        Complex-conjugate of all elements of QGsuper.
    trans : QGsuper -> QGsuper
        Transpose of all elements of QGsuper.
    dag : QGsuper -> QGsuper
        Adjoint (dagger) of QGsuper.
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
            self._dims_fls = inpt.dims_fls
            self._iscvs = inpt.iscvs
            self._isfls = inpt.isfls

            self._data_0th = inpt.data_0th
            self._data_1st_l = inpt.data_1st_l
            self._data_1st_r = inpt.data_1st_r
            self._data_2nd_l = inpt.data_2nd_l
            self._data_2nd_r = inpt.data_2nd_r
            self._data_2nd_m = inpt.data_2nd_m

        # In other cases, specific components of QGsuper must be included as arguments.
        elif inpt is None:
            # Set dimensions of FLS and CV components from input data. Also sets the isfls and iscvs properties.
            self.dims_cvs = dims_cvs
            self.dims_fls = dims_fls

            # Set data arrays from input data.
            self.data_0th = data_0th
            self.data_1st_l = data_1st_l
            self.data_1st_r = data_1st_r
            self.data_2nd_l = data_2nd_l
            self.data_2nd_r = data_2nd_r
            self.data_2nd_m = data_2nd_m

            if qgauss.settings.auto_tidyup == True:
                self.tidyup()

        else:
            raise TypeError("Input for constructing QGsuper is either ill-formatted or of incorrect type.")

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
        # Initialize arrays of bilinear-order quadrature superoperator 
        # coefficients multiplying from the left.
        if isinstance(data, (np.ndarray, list)):
            if np.shape(data) == self.shape_2nd:
                self._data_2nd_l = np.asarray(data, dtype = complex)
            else:
                raise ValueError("Dimensions of data_2nd_l do not agree with stored dimensions.")
        elif data is None:
            self._data_2nd_l = np.zeros(self.shape_2nd, dtype = complex)
        else:
            raise TypeError("Input of data_2nd_l is not of a supported type: np.ndarray or list.")
        self._invalidate_wigner('2nd')
        self._invalidate_order('2nd')

    @property
    def data_2nd_r(self) -> npt.NDArray:
        return self._data_2nd_r
    @data_2nd_r.setter
    def data_2nd_r(self, data):
        # Initialize arrays of bilinear-order quadrature superoperator 
        # coefficients multiplying from the right.
        if isinstance(data, (np.ndarray, list)):
            if np.shape(data) == self.shape_2nd:
                self._data_2nd_r = np.asarray(data, dtype = complex)
            else:
                raise ValueError("Dimensions of data_2nd_r do not agree with stored dimensions.")
        elif data is None:
            self._data_2nd_r = np.zeros(self.shape_2nd, dtype = complex)
        else:
            raise TypeError("Input of data_2nd_r is not of a supported type: np.ndarray or list.")
        self._invalidate_wigner('2nd')
        self._invalidate_order('2nd')

    @property
    def data_2nd_m(self) -> npt.NDArray:
        return self._data_2nd_m
    @data_2nd_m.setter
    def data_2nd_m(self, data):
        # Initialize arrays of bilinear-order quadrature superoperator 
        # coefficients multiplying from the left and right.
        if isinstance(data, (np.ndarray, list)):
            if np.shape(data) == self.shape_2nd:
                self._data_2nd_m = np.asarray(data, dtype = complex)
            else:
                raise ValueError("Dimensions of data_2nd_m do not agree with stored dimensions.")
        elif data is None:
            self._data_2nd_m = np.zeros(self.shape_2nd, dtype = complex)
        else:
            raise TypeError("Input of data_2nd_m is not of a supported type: np.ndarray or list.")
        self._invalidate_wigner('2nd')
        self._invalidate_order('2nd')
    
    @property
    def data_1st_l(self) -> npt.NDArray:
        return self._data_1st_l
    @data_1st_l.setter
    def data_1st_l(self, data):
        # Initialize array of linear-order quadrature operator
        # coefficients multiplying from the left.
        if isinstance(data, (np.ndarray, list)):
            if np.shape(data) == self.shape_1st:
                self._data_1st_l = np.asarray(data, dtype = complex)
            else:
                raise ValueError("Dimensions of data_1st_l do not agree with stored dimensions.")
        elif data is None:
            self._data_1st_l = np.zeros(self.shape_1st, dtype = complex)
        else:
            raise TypeError("Input of data_1st_l is not of a supported type: np.ndarray or list.")
        self._invalidate_wigner('1st')
        self._invalidate_order('1st')
        
    @property
    def data_1st_r(self) -> npt.NDArray:
        return self._data_1st_r
    @data_1st_r.setter
    def data_1st_r(self, data):
        # Initialize arrays of bilinear-order quadrature superoperator 
        # coefficients multiplying from the right.
        if isinstance(data, (np.ndarray, list)):
            if np.shape(data) == self.shape_1st:
                self._data_1st_r = np.asarray(data, dtype = complex)
            else:
                raise ValueError("Dimensions of data_1st_r do not agree with stored dimensions.")
        elif data is None:
            self._data_1st_r = np.zeros(self.shape_1st, dtype = complex)
        else:
            raise TypeError("Input of data_1st_r is not of a supported type: np.ndarray or list.")
        self._invalidate_wigner('1st')
        self._invalidate_order('1st')
        
    @property
    def data_0th(self) -> npt.NDArray:
        return self._data_0th
    @data_0th.setter
    def data_0th(self, data):
        # Initialize array of zeroth-order quadrature superoperator coefficients.
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
            raise TypeError("Input of data_0th is not of a supported type: np.ndarray, list, or number.")
        self._invalidate_wigner('0th')
        self._invalidate_order('0th')
    
    @cached_property
    def wigner_2nd_deriv_var(self) -> npt.NDArray:
        # Drift matrix, system dynamics matrix
        if self.isfls == True:
            return np.einsum("jk,lmkn->lmjn",
                             0.5j*self.symform,
                             (0.5*(self.data_2nd_l + np.transpose(self.data_2nd_l,[0,1,3,2]))
                              - 0.5*(self.data_2nd_r + np.transpose(self.data_2nd_r,[0,1,3,2]))
                              + (self.data_2nd_m - np.transpose(self.data_2nd_m,[0,1,3,2])))
                              )
        else:
            return 0.5j*self.symform@(0.5*(self.data_2nd_l + np.transpose(self.data_2nd_l))
                                      - 0.5*(self.data_2nd_r + np.transpose(self.data_2nd_r))
                                      + (self.data_2nd_m - np.transpose(self.data_2nd_m)))
    
    @cached_property
    def wigner_2nd_var(self) -> npt.NDArray:
        # Riccati coupling/feedback matrix, information-gain‑matrix, measurement‑error‑weight, trap/potential stiffness
        if self.isfls == True:
            return (- 0.5*(self.data_2nd_l + np.transpose(self.data_2nd_l,[0,1,3,2]))
                    - 0.5*(self.data_2nd_r + np.transpose(self.data_2nd_r,[0,1,3,2]))
                    - (self.data_2nd_m + np.transpose(self.data_2nd_m,[0,1,3,2])))
        else:
            return (- 0.5*(self.data_2nd_l + np.transpose(self.data_2nd_l))
                    - 0.5*(self.data_2nd_r + np.transpose(self.data_2nd_r))
                    - (self.data_2nd_m + np.transpose(self.data_2nd_m)))
        
    @cached_property
    def wigner_2nd_deriv(self) -> npt.NDArray:
        # Diffusion matrix, additive-noise matrix
        if self.isfls == True:
            return np.einsum("jk,lmkn,np->lmjp",
                             0.25*self.symform,
                             (0.5*(self.data_2nd_l + np.transpose(self.data_2nd_l,[0,1,3,2]))
                              + 0.5*(self.data_2nd_r + np.transpose(self.data_2nd_r,[0,1,3,2]))
                              - (self.data_2nd_m + np.transpose(self.data_2nd_m,[0,1,3,2]))),
                              self.symform
                              )
        else:   
            return 0.25*self.symform@(0.5*(self.data_2nd_l + np.transpose(self.data_2nd_l))
                                      + 0.5*(self.data_2nd_r + np.transpose(self.data_2nd_r))
                                      - (self.data_2nd_m + np.transpose(self.data_2nd_m))
                                      )@self.symform
    
    @cached_property
    def wigner_1st_var(self) -> npt.NDArray:
        # Restoring‑force coefficient vector, linear-damping/decay rate vector
        return -(self.data_1st_l + self.data_1st_r)

    @cached_property
    def wigner_1st_deriv(self) -> npt.NDArray:
        # Friction vector, drift‑shift coefficient vector
        if self.isfls == True:  
            return np.einsum("jk,lmj->lmk",
                             0.5j*self.symform,
                             self.data_1st_l - self.data_1st_r
                             )
        else:   
            return 0.5j*self.symform@(self.data_1st_l - self.data_1st_r)                                          
        
    @cached_property
    def wigner_0th(self) -> npt.NDArray:
        # Decay term, sink term, decay rate of the state norm, loss rate
        if self.isfls == True:
            return -self.data_0th + 0.5j*np.einsum("jk,lmkj->lm",
                                                   self.symform,
                                                   0.5*(self.data_2nd_l + self.data_2nd_r) - self.data_2nd_m
                                                   )
        else:
            return -self.data_0th + 0.5j*np.array(
                np.trace(self.symform@(0.5*(self.data_2nd_l + self.data_2nd_r) - self.data_2nd_m))
                )

    def _invalidate_wigner(self, order):
        # Remove chached properties storing Wigner arrays when updating data matrices.
        if order == '2nd':
            if hasattr(self, 'wigner_2nd_deriv_var'):
                delattr(self, 'wigner_2nd_deriv_var')
            if hasattr(self, 'wigner_2nd_var'):
                delattr(self, 'wigner_2nd_var')
            if hasattr(self, 'wigner_2nd_deriv'):
                delattr(self, 'wigner_2nd_deriv')
            if hasattr(self, 'wigner_0th'):
                delattr(self, 'wigner_0th')
        elif order == '1st':
            if hasattr(self, 'wigner_1st_deriv'):
                delattr(self, 'wigner_1st_deriv')
            if hasattr(self, 'wigner_1st_var'):
                delattr(self, 'wigner_1st_var')
        elif order == '0th':
            if hasattr(self, 'wigner_0th'):
                delattr(self, 'wigner_0th')
                
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
        # Set iscvs propert
        self.iscvs = dims
        
    @property
    def dims_fls(self) -> list[list[list[int]]]:
        return self._dims_fls
    @dims_fls.setter
    def dims_fls(self, dims):
        if isinstance(dims, (np.ndarray, list)):
            self._dims_fls = list(dims)
        elif dims is None:
            self._dims_fls = [[[],[]],[[],[]]]
        else:
            raise TypeError("Input to dims_fls is not of a supported type: np.ndarray or list.")
        # Set isfls propert
        self.isfls = dims

    @property
    def shape_2nd(self) -> tuple[int,int,int,int] | tuple[int,int]:
        if self.isfls:
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
        if self.isfls:
            return (np.prod(self.dims_fls[0]).item(),
                    np.prod(self.dims_fls[1]).item(),
                    2*self.dims_cvs
                    )
        else:
            return (2*self.dims_cvs,)
        
    @property
    def shape_0th(self) -> tuple[int,int] | tuple[int]:
        if self.isfls:
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
        if dims == [[[],[]],[[],[]]] or dims == None:
            self._isfls = False
        else:
            self._isfls = True
 
    @cached_property
    def is2nd(self) -> bool:
        if ((np.all(np.abs(self.data_2nd_l) < qgauss.settings.atol) or
             np.all(np.abs(self.data_2nd_r) < qgauss.settings.atol) or
             np.all(np.abs(self.data_2nd_m) < qgauss.settings.atol))
             or
            (self.data_2nd_l.size == 0 or
             self.data_2nd_r.size == 0 or
             self.data_2nd_m.size == 0)
            ):
            return False
        else:
            return True
    
    @cached_property
    def is1st(self) -> bool:
        if ((np.all(np.abs(self.data_1st_l) < qgauss.settings.atol) or
             np.all(np.abs(self.data_1st_r) < qgauss.settings.atol))
             or
            (self.data_1st_l.size == 0 or
             self.data_1st_r.size == 0)
            ):
            return False
        else:
            return True
    
    @cached_property
    def is0th(self) -> bool:
        if (np.all(np.abs(self.data_0th) < qgauss.settings.atol) or
            self.data_0th.size == 0
            ):
            return False
        else:
            return True

    def _invalidate_order(self, order):
        # Remove chached properties when updating data matrices.
        if order == '2nd':
            if hasattr(self, 'is2nd'):
                delattr(self, 'is2nd')
            if hasattr(self, 'is0th'):
                delattr(self, 'is0th')
        elif order == '1st':
            if hasattr(self, 'is1st'):
                delattr(self, 'is1st')
        elif order == '0th':
            if hasattr(self, 'is0th'):
                delattr(self, 'is0th')

    @property
    def iscoherent(self) -> bool:
        if self == -self.dag():
            return True
        else:
            return False
        
    @property 
    def isgauss(self) -> bool: 
        if self.iscvs and not self.isfls:
            return True
        else:
            return all([self.issubgauss(j) for j in range(np.prod(self.dims_fls[1]))])

    def issubgauss(self, row: int, col: int = None) -> bool:
        """ Checks that the dynamics of CV-component of input QGsuper are Gaussian by ensuring that there is no coupling 
        to other elements of the qubit-density operator. If only row is specified, this is taken to mean that the that
        row in the vectorised superoperator is to be checked. If rol and col are specified, then this is taken to mean
        that the dynamics acting on the [row,col] component of a QGstate is Gaussian. """
        rank = np.prod(self.dims_fls[1])
        if col is None:
            j = row
        else:
            j = row*np.prod(self.dims_fls[0][0]) + col

        if (all([np.any(np.abs(self.data_2nd_l[j,k]) < qgauss.settings.atol) for k in range(rank) if k != j]) and
            all([np.any(np.abs(self.data_2nd_r[j,k]) < qgauss.settings.atol) for k in range(rank) if k != j]) and
            all([np.any(np.abs(self.data_2nd_m[j,k]) < qgauss.settings.atol) for k in range(rank) if k != j]) and
            all([np.any(np.abs(self.data_1st_l[j,k]) < qgauss.settings.atol) for k in range(rank) if k != j]) and
            all([np.any(np.abs(self.data_1st_r[j,k]) < qgauss.settings.atol) for k in range(rank) if k != j]) and
            all([np.any(np.abs(self.data_0th[j,k]) < qgauss.settings.atol) for k in range(rank) if k != j])
            ):
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

    ### Addition and subtraction of QGsupers ###

    def __add__(self, other: QGsuper) -> QGsuper:
        # Addition with self.QGsuper on the left
        if isinstance(other, QGsuper):
            if ((self.dims_cvs == other.dims_cvs) and (self.dims_fls == other.dims_fls)):
                return QGsuper(data_2nd_l = self.data_2nd_l + other.data_2nd_l,
                               data_2nd_r = self.data_2nd_r + other.data_2nd_r,
                               data_2nd_m = self.data_2nd_m + other.data_2nd_m,
                               data_1st_l = self.data_1st_l + other.data_1st_l,
                               data_1st_r = self.data_1st_r + other.data_1st_r,
                               data_0th = self.data_0th + other.data_0th,
                               dims_cvs = self.dims_cvs,
                               dims_fls = self.dims_fls
                              )
            else:
                raise ValueError("Cannot perform addition operation between QGsupers with different dimensions.")
        elif other == 0:
            return QGsuper(self)
        else:
            raise TypeError("Cannot perform addition operation between the types QGsuper and " 
                            + type(other).__name__ + ".")

    def __radd__(self, other: QGsuper) -> QGsuper:
        # Addition with the self.QGsuper on the right
        return self.__add__(other)

    def __sub__(self, other: QGsuper) -> QGsuper:
        # Subtraction with self.QGsuper on the left
        return self.__add__(other.__neg__())

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
                       dims_fls = self.dims_fls
                      )

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
                           dims_fls = self.dims_fls
                          )
        else:
            raise TypeError("Cannot perform multiplication operation between the types QGsuper and " 
                            + type(other).__name__ + ".")

    def __rmul__(self, other: complex) -> QGsuper:
        # Multiplication with self.QGsuper on the right
        if isinstance(other, (numbers.Number, np.number)):
            return self.__mul__(other)
        else:
            raise TypeError("Cannot perform multiplication operation between the types QGsuper and " 
                            + type(other).__name__ + ".")

    def __truediv__(self, other: complex) -> QGsuper:
        # Division of self.QGsuper by number
        if isinstance(other, (numbers.Number,np.number)):
            return QGsuper(data_2nd_l = self.data_2nd_l/other,
                           data_2nd_r = self.data_2nd_r/other,
                           data_2nd_m = self.data_2nd_m/other,
                           data_1st_l = self.data_1st_l/other,
                           data_1st_r = self.data_1st_r/other,
                           data_0th = self.data_0th/other,
                           dims_cvs = self.dims_cvs,
                           dims_fls = self.dims_fls
                          )
        else:
            raise TypeError("Cannot perform division operation between the types QGsuper and " 
                            + type(other).__name__ + ".")

    ### Assorted Methods ###

    def __eq__(self, other: QGsuper) -> bool:
        # Check equality of QGsupers #
        if (isinstance(other, QGsuper) and
            (self.dims_fls == other.dims_fls) and
            (self.dims_cvs == other.dims_cvs) and
            np.all(np.abs(self.data_2nd_l - other.data_2nd_l) < qgauss.settings.atol) and
            np.all(np.abs(self.data_2nd_r - other.data_2nd_r) < qgauss.settings.atol) and
            np.all(np.abs(self.data_2nd_m - other.data_2nd_m) < qgauss.settings.atol) and
            np.all(np.abs(self.data_1st_l - other.data_1st_l) < qgauss.settings.atol) and
            np.all(np.abs(self.data_1st_r - other.data_1st_r) < qgauss.settings.atol) and
            np.all(np.abs(self.data_0th - other.data_0th) < qgauss.settings.atol)
            ):
            return True
        else:
            return False

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
                           dims_cvs = self.dims_cvs
                          )
        else:
            raise ValueError("QGsuper requires an FLS and CV component to use this method. "
                             + "Access QGsuper data arrays individually if specific elements are required.")

    def drop(self, *args) -> QGsuper:
        # Removes CV modes specified in args from self, and return a new QGsuper
        # List, array, or tuple of indices passed as args, convert to tuple
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple)):
            args = tuple(args[0])
        # Generate indices to remove from CVS part
        ind = [n for x in args for n in (2*x-2, 2*x-1)]

        if self.isfls == False:
            return QGsuper(data_2nd_l = np.delete(np.delete(self.data_2nd_l, ind, axis=1), ind, axis=0),
                           data_2nd_r = np.delete(np.delete(self.data_2nd_r, ind, axis=1), ind, axis=0),
                           data_2nd_m = np.delete(np.delete(self.data_2nd_m, ind, axis=1), ind, axis=0),
                           data_1st_l = np.delete(self.data_1st_l, ind, axis=0),
                           data_1st_r = np.delete(self.data_1st_r, ind, axis=0),
                           data_0th = self.data_0th,
                           dims_cvs = self.dims_cvs - len(args)
                           )
        else:
            return QGsuper(data_2nd_l = np.delete(np.delete(self.data_2nd_l, ind, axis=3), ind, axis=2),
                           data_2nd_r = np.delete(np.delete(self.data_2nd_r, ind, axis=3), ind, axis=2),
                           data_2nd_m = np.delete(np.delete(self.data_2nd_m, ind, axis=3), ind, axis=2),
                           data_1st_l = np.delete(self.data_1st_l, ind, axis=2),
                           data_1st_r = np.delete(self.data_1st_r, ind, axis=2),
                           data_0th = self.data_0th,
                           dims_fls = self.dims_fls,
                           dims_cvs = self.dims_cvs - len(args)
                           )
   
    def keep(self, *args) -> QGsuper:
        # Keeps CV modes specified in args from self, and return a new QGsuper
        # List, array, or tuple of indices passed as args, convert to tuple
        if len(args) == 1 and isinstance(args[0], (np.ndarray, list, tuple)):
            args = tuple(args[0])

        # Generate list of modes to remove from CVS part by taking difference with set of all modes
        ind = list(set(range(1,self.dims_cvs+1)) - set(args))
        return self.drop(ind)

    def conj(self) -> QGsuper:
        # Complex-conjugate of all elements of the QGsuper
        return QGsuper(data_2nd_l = np.conj(self.data_2nd_l),
                       data_2nd_r = np.conj(self.data_2nd_r),
                       data_2nd_m = np.conj(self.data_2nd_m),
                       data_1st_l = np.conj(self.data_1st_l),
                       data_1st_r = np.conj(self.data_1st_r),
                       data_0th = np.conj(self.data_0th),
                       dims_cvs = self.dims_cvs,
                       dims_fls = self.dims_fls
                       )

    def trans(self, level = None) -> QGsuper:
        # Transpose of arrays within the QGsuper. Can specify the level at which it is
        # applied, either "FLS" or "CVS", or the entire array if none is passed. 
        if self.isfls:
            if level is None:
                return QGsuper(data_2nd_l = np.transpose(self.data_2nd_l,[1,0,3,2]),
                               data_2nd_r = np.transpose(self.data_2nd_r,[1,0,3,2]),
                               data_2nd_m = np.transpose(self.data_2nd_m,[1,0,3,2]),
                               data_1st_l = np.transpose(self.data_1st_l,[1,0,2]),
                               data_1st_r = np.transpose(self.data_1st_r,[1,0,2]),
                               data_0th = np.transpose(self.data_0th,[1,0]),
                               dims_fls = [self.dims_fls[1], self.dims_fls[0]],
                               dims_cvs = self.dims_cvs
                               )
            elif level == 'FLS':
                return QGsuper(data_2nd_l = np.transpose(self.data_2nd_l,[1,0,2,3]),
                               data_2nd_r = np.transpose(self.data_2nd_r,[1,0,2,3]),
                               data_2nd_m = np.transpose(self.data_2nd_m,[1,0,2,3]),
                               data_1st_l = np.transpose(self.data_1st_l,[1,0,2]),
                               data_1st_r = np.transpose(self.data_1st_r,[1,0,2]),
                               data_0th = np.transpose(self.data_0th,[1,0]),
                               dims_fls = [self.dims_fls[1], self.dims_fls[0]],
                               dims_cvs = self.dims_cvs
                               )
            elif level == 'CVS':
                return QGsuper(data_2nd_l = np.transpose(self.data_2nd_l,[0,1,3,2]),
                               data_2nd_r = np.transpose(self.data_2nd_r,[0,1,3,2]),
                               data_2nd_m = np.transpose(self.data_2nd_m,[0,1,3,2]),
                               data_1st_l = self.data_1st_l,
                               data_1st_r = self.data_1st_r,
                               data_0th = self.data_0th,
                               dims_fls = self.dims_fls,
                               dims_cvs = self.dims_cvs
                               )
        else:
            if level == 'CVS' or level is None:
                return QGsuper(data_2nd_l = np.transpose(self.data_2nd_l),
                               data_2nd_r = np.transpose(self.data_2nd_r),
                               data_2nd_m = np.transpose(self.data_2nd_m),
                               data_1st_l = np.transpose(self.data_1st_l),
                               data_1st_r = np.transpose(self.data_1st_r),
                               data_0th = np.transpose(self.data_0th),
                               dims_cvs = self.dims_cvs
                               )
            elif level == 'FLS':
                return self

    def dag(self) -> QGsuper:
        # Adjoint/complex-conjugate/dagger of QGsuper
        # Right and left-multiplication are switched
        if self.isfls:
            return QGsuper(data_2nd_l = np.transpose(np.conj(self.data_2nd_l),[1,0,3,2]),
                           data_2nd_r = np.transpose(np.conj(self.data_2nd_r),[1,0,3,2]),
                           data_2nd_m = np.transpose(np.conj(self.data_2nd_m),[1,0,3,2]),
                           data_1st_l = np.transpose(np.conj(self.data_1st_l),[1,0,2]),
                           data_1st_r = np.transpose(np.conj(self.data_1st_r),[1,0,2]),
                           data_0th = np.transpose(np.conj(self.data_0th),[1,0]),
                           dims_fls = [self.dims_fls[1], self.dims_fls[0]],
                           dims_cvs = self.dims_cvs
                           )
        else:
            return QGsuper(data_2nd_l = np.transpose(np.conj(self.data_2nd_l)),
                           data_2nd_r = np.transpose(np.conj(self.data_2nd_r)),
                           data_2nd_m = np.transpose(np.conj(self.data_2nd_m)),
                           data_1st_l = np.transpose(np.conj(self.data_1st_l)),
                           data_1st_r = np.transpose(np.conj(self.data_1st_r)),
                           data_0th = np.transpose(np.conj(self.data_0th)),
                           dims_fls = [self.dims_fls[1], self.dims_fls[0]],
                           dims_cvs = self.dims_cvs
                           )
        
    def tidyup(self, tol: float = qgauss.settings.tidyup_atol) -> QGsuper:
        # Private void function to remove small magnitude elements from data arrays
        np.real(self.data_2nd_l)[np.abs(np.real(self.data_2nd_l)) < tol] = 0
        np.imag(self.data_2nd_l)[np.abs(np.imag(self.data_2nd_l)) < tol] = 0

        np.real(self.data_2nd_r)[np.abs(np.real(self.data_2nd_r)) < tol] = 0
        np.imag(self.data_2nd_r)[np.abs(np.imag(self.data_2nd_r)) < tol] = 0

        np.real(self.data_2nd_m)[np.abs(np.real(self.data_2nd_m)) < tol] = 0
        np.imag(self.data_2nd_m)[np.abs(np.imag(self.data_2nd_m)) < tol] = 0

        np.real(self.data_1st_l)[np.abs(np.real(self.data_1st_l)) < tol] = 0
        np.imag(self.data_1st_l)[np.abs(np.imag(self.data_1st_l)) < tol] = 0

        np.real(self.data_1st_r)[np.abs(np.real(self.data_1st_r)) < tol] = 0
        np.imag(self.data_1st_r)[np.abs(np.imag(self.data_1st_r)) < tol] = 0

        np.real(self.data_0th)[np.abs(np.real(self.data_0th)) < tol] = 0
        np.imag(self.data_0th)[np.abs(np.imag(self.data_0th)) < tol] = 0