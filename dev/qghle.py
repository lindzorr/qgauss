from __future__ import annotations

import warnings
import numbers
import numpy.typing as npt
from functools import cached_property
import qgauss
import numpy as np

from ..core.qgstate import QGstate
from ..core.qgoper import QGoper
from ..core.qgsuper import QGsuper
from ..calc.utilities import trim,trim_abs

__all__ = ['QGhle']


class QGhle(object):

    """
    ---- Structure ----
    Write this and add more comments to code.
    
    This function follows the theory developed in the following paper to 
    describe the system-environment/bath/reservoir coupling and input-ouput theory:
        Gardiner & Collett, Phys. Rev. A 31, 3761 (1985).
    For the multimode case, supporting expressions can be found in Appendices A3 
    and A4 in Miller and Orr et al, arXiv:2603.12312 (2026).
    To summarize, we 
    represent the system quadratures as
        r_sys = (q1,p1,...,qN,pN),
    while the bath quadratures are implicitly frequency dependent and are 
    expressed as
        r_bath = (q1(ω),p1(ω),...,qM(ω),pM(ω)).
    In representing the system-bath Hamiltonian, the corresponding form, 
    following Gardiner and Collett, will be
        (1/√2π)*Integral[ (r_sys,r_bath)^T.H_system_bath.(r_sys,r_bath) dω],
    where "H_system_bath" is the matrix of coefficients for the bilinear 
    system-bath Hamiltonian. When constructing this Hamiltonian, omit the 
    integral and instead construct the QGoper using
        (r_sys,r_bath)^T.H_system_bath.(r_sys,r_bath)
    where the bath modes must come after the system modes in the tensor product.
    The data in the system-bath Hamiltonian will have the form 
        H_system_bath.data_2nd = [[0,Hsb],[Hsb^T,0]]. 
    If we represent the input states, ρ_in, as a QGstate, then the corresponding 
    dissipation component in the Lindbladian will be:
        Γ = Hsb @ (ρ_in.data_2nd + iΩ_2M/2) @ Hsb^T 
        where Ω_2M is the 2Mx2M symplectic form.
    This deomnstrates why this function does not use the QGsuper as its input, 
    since Hsb is required to formulate the input-ouput theory for the 
    measurement rate, but knowledge of Γ and ρ_in is insufficient to uniquqly 
    determine the system-bath coupling Hsb.


    ---- Parameters ----
    inpt : QGhle
        Create a copy of another QGhle.
    h_sys : QGoper
        System Hamiltonian, containing interactions between the CVS and FLS 
        parts of the main system. If system operators are found to be tensored 
        with bath operators, the bath modes will be dropped.
    h_sys_env : QGoper
        System-bath Hamiltonian, representing coupling between the system and a
        Markovian bath. The Hamiltonian coupling is therefore independent of the
        frequency of the bath modes.
    input_env : QGstate
        Input-field state of the input bath operators, representing the 
        correlations of the Markovian noise from the bath. Delta-correlations
        part of the covariances are ignored.
    dims_cvs : int
        Number of continuous variable system (CVS) modes.
    dims_fls : array_like
        Dimensions of the finite-level component of the system (FLS).
    dims_env : int
        Number of continuous variable environments/baths/reservoirs, each of 
        which is comprised of an infinite number of bosonic modes.
    input_coupling_mat : array_like
        Alternative to h_sys_env. Represents the coupling to the input-fields
        in the Heisenberg-Langevin equations:
            dr/dt = A.r - sqrt(k).r_in
        The argument input_coupling_mat corresponds to the array sqrt(k), from which
        h_sys_env is then constructed. Cannot be used for systems where 
        dissipation is dependent on the FLS system, 
            
    ---- Attributes ----
    h_sys : QGoper
        System Hamiltonian, containing interactions between the CVS and FLS 
        parts of the main system.
    h_sys_env : QGoper
        System-environment Hamiltonian, representing the coupling of the system
        to a Markovian bath/reservoir.
    h_se : array
        Internal data representing the non-zero coupling component from
        h_sys_env. Specifically, when writing h_sys_env in block form, h_se
        represents the following array:
            h_sys_env.data_2nd = [[0, h_se], [0, h_se^T]].
        Since h_se is used in many calculations using the HLEs, it is beneficial
        to store this array as a class property.
    input_env : QGstate
        In-field state of the input bath operators, representing the 
        correlations of the Markovian noise from the bath.
    lme : QGsuper
        Lindblad master equation equivalent to the Heisenberg-Langevin equations
        for the internal modes of the system. Both h_sys and h_sys_env must be
        Hermitian for the Lindbladian to be CPTP, and both must be diagonal in 
        any FLS-basis, or else no Gaussian presrving superoperator will be
        constructed.
    dims_cvs : int
        Number of continuous variable system (CVS) modes of the main system.
    dims_fls : list[list[int]]
        Dimensions of the finite-level component of the main system (FLS).
    dims_env : int
        Number of continuous variable environments, each of which is comprised
        of an infinite number of bosonic modes.
    isfls : bool
    isherm : bool
    isgauss : bool
    iscoherent : bool
    symform_sys : array
        Symplectic form acting on either the system CVS modes. The
        generated matrix will have the form:
            Ω = ⊗_{j=1}^dims_cvs [[0,1],[-1,0]]
    symform_env : array
        Symplectic form acting on either the bath degrees of freedom. The
        generated matrix will have the form:
            Ω = ⊗_{j=1}^dims_env [[0,1],[-1,0]]
    lme : QGsuper
        Lindbladian of the main system which is equivalent to the main-system
        dynamics described by the Heisenberg-Langevin equation. Corresponds to
        a CPTP Gaussian-quantum channel, and can only be defined if both h_sys
        and h_sys_env are Hermitian Gaussian-state preserving Hamiltonians, and
        if input_env corresponds to a true quantum-state.

    ---- Methods ----
    scattering_matrix : (QGhle, float, str) -> array
    transfer_matrix : (QGhle, float, str) -> array
    output_env : (QGhle, float, str) -> QGstate
    eq : (QGhle, QGhle) -> bool
        Check equality of two QGhles.
    
    """

    ### Quantum-Gaussian Heisenberg-Langevin Equation (QGhle) Initialisation ###
    def __init__(self, 
                 inpt: QGhle = None,
                 h_sys: QGoper = None,
                 h_sys_env: QGoper = None,
                 input_env: QGstate = None,
                 dims_cvs: int = None,
                 dims_fls: list[list[int]] = None,
                 dims_env: int = None,
                 input_coupling_mat: npt.ArrayLike = None
                ):

        # QGhle as input, copy data.
        if isinstance(inpt, QGoper):
            self._dims_cvs = inpt.dims_cvs
            self._dims_fls = inpt.dims_fls
            self._dims_env = inpt.dims_env

            self._h_sys = inpt.h_sys
            self._h_sys_env = inpt.h_sys_env
            self._input_env = inpt.input_env

        # In other cases, specific components of QGoper must be passed as
        # arguments. The exceptions are h_sys_env and input_coupling_mat, where
        # only one need be provided.
        elif inpt is None:
            # Set dimensions of FLS and CVS components, along with the number
            # of evironmental degrees of freedom. Currently, the dims_cvs and 
            # dims_env  parameters must be specified for initialization to 
            # proceed, and are not inferred from the data.

            # Set dims_fls by successively checking each argument.
            if dims_fls is not None:
                self.dims_fls = dims_fls
            elif h_sys is not None:
                self.dims_fls = h_sys.dims_fls
            elif h_sys_env is not None:
                self.dims_fls = h_sys_env.dims_fls
            else:
                self.dims_fls = dims_fls
            # Set dims_cvs and dims_env
            self.dims_cvs = dims_cvs
            self.dims_env = dims_env

            # Set data structures from inputs
            self.h_sys = h_sys
            self.input_env = input_env

            # input_coupling_mat is a convenience argument to pass as an alternative
            # to constructing the system-environment Hamiltonian h_sys_env explicitly,
            # and will only be used if no argument for h_sys_env is passed.
            if h_sys_env is not None:
                self.h_sys_env = h_sys_env
            else:
                self.h_sys_env = input_coupling_mat

            if qgauss.settings.auto_tidyup == True: 
                self.tidyup()

        else:
            raise TypeError("Input for constructing QGhle is ill-formatted or of incorrect type.")

    '''
    ------------------
        Properties
    ------------------
    '''

    @property
    def h_sys(self) -> QGoper:
        return self._h_sys
    @h_sys.setter
    def h_sys(self, data):
        # Initialize QGoper which represents the system Hamiltonian.
        if isinstance(data, QGoper):
            # h_sys must have the dimensions speficied by self.dims_cvs and 
            # self.dims_fls. Since it may be more convenient when creating
            # h_sys to initially tensor the system and environment operators, 
            # the dimensions self.dims_cvs + self.dims_env is also accepted. 
            # In this case, the environment modes will be dropped from h_sys.
            if (data.dims_fls == self.dims_fls and 
                data.dims_cvs == self.dims_cvs
               ):
                self._h_sys = data
            elif (data.dims_fls == self.dims_fls and
                  data.dims_cvs == self.dims_tot
                 ):
                self._h_sys = data.drop(tuple(range(self.dims_cvs + 1, 
                                                    self.dims_tot + 1)))
            else:
                raise ValueError("Dimensions of h_sys do not agree with stored dimensions.")
        elif data is None:
            # No argument passed for h_sys, create a QGoper of all zeroes
            self._h_sys = QGoper(dims_cvs = self.dims_cvs,
                                 dims_fls = self.dims_fls)
        else:
            raise TypeError("Input for h_sys is not of a supported type: QGoper.")
    
    @property
    def h_sys_env(self) -> QGoper:
        return self._h_sys_env
    @h_sys_env.setter
    def h_sys_env(self, data) -> QGoper:
        if isinstance(data, QGoper):
            # If a QGoper is passed, assume this represents a system-environment Hamiltonian.
            # Note: The first check requires calculating cached properties, 
            # which may be slow if we are repeadedly creating instances of 
            # QGhle. Find a way around this, or else find out how such
            # couplings can be incorporated into the construction of the class.

            # First, check that only bilinear couplings are present. Couplings
            # between pairs of system modes and pairs of environment modes are 
            # currently not checked, and will simply be ignored.
            if data.is1st or data.is0th:
                raise ValueError("h_sys_env can only contain couplings " \
                "between system and environment operators.")
            
            # Next, check that the dimensions of data agree with combined
            # system-CVS and environment dimensions of self.
            # FLS dimensions of data must either agree with those of self, 
            # or else the system-environment Hamiltonian is not coupled to an
            # FLS. In this case, data is tensored with an identity QGoper.
            if (data.dims_fls == self.dims_fls and
                data.dims_cvs == self.dims_tot
               ):
                self._h_sys_env = data
            elif (data.dims_cvs == self.dims_tot and
                  self._isfls and not data.isfls
                 ):
                self._h_sys_env = \
                qgauss.tensor(data,
                              QGoper(data_0th = np.identity(np.prod(self.dims_fls[0])),
                                     dims_fls = self.dims_fls))
            else:
                raise ValueError("Dimensions of h_sys_env do not agree with stored dimensions.")     
        elif isinstance(data, (np.ndarray, list)):
            # If an array-like structure is passed, assume this represents the
            # decay-rate coupling matrix between the system and input-fields
            _data = self.symform_sys @ np.asarray(data)
            if (_data.shape[0] == 2*self.dims_cvs and
                _data.shape[1] == 2*self.dims_env and
                _data.ndim == 2
               ):
                _data_2nd = np.block([[np.zeros((2*self.dims_cvs, 2*self.dims_cvs)),
                                       _data],
                                      [_data.T,
                                       np.zeros((2*self.dims_env, 2*self.dims_env))]])
                if self.isfls:
                    self._h_sys_env = \
                    qgauss.tensor(QGoper(data_2nd = _data_2nd,
                                         dims_cvs = self.dims_tot),
                                  QGoper(data_0th = np.identity(np.prod(self.dims_fls[0])),
                                         dims_fls = self.dims_fls))
                else:
                    self._h_sys_env = \
                    QGoper(data_2nd = _data_2nd,
                           dims_cvs = self.dims_tot)
            else:
                raise ValueError("Dimensions of input_coupling_mat cannot be " \
                "brought into agreement with stored dimensions.")
        elif data is None:
            # No argument passed, create an empty QGoper
            self._h_sys_env = QGoper(dims_fls = self.dims_fls,
                                     dims_cvs = self.dims_tot)
        else:
           raise TypeError("Input is not of a supported type: QGoper or array-like.")
        
        # Set h_se property immediately.
        if self.isfls == True:
            self._h_se = \
            np.asarray([[self._h_sys_env[x,y].data_2nd[0:2*self.dims_cvs,
                                                        2*self.dims_cvs:2*self.dims_tot]
                         for x in range(0,np.prod(self.dims_fls[0]))]
                         for y in range(0,np.prod(self.dims_fls[1]))])
        else:
            self._h_se = \
            self._h_sys_env.data_2nd[0:2*self.dims_cvs,
                                     2*self.dims_cvs:2*self.dims_tot]
             
    @property
    def h_se(self) -> npt.NDArray:
        return self._h_se
                   
    @property
    def input_env(self) -> QGstate:
        return self._input_env
    @input_env.setter
    def input_env(self, data):
        # Initialize QGstate which represents the input state of the environment.
        if isinstance(data, QGstate):
            # input_env must have the dimensions speficied by self.dims_env.
            if data.isfls is True:
                raise ValueError("Input-field state can only have a CVS component.")
            elif data.dims_cvs == self.dims_env:
                self._input_env = data
            else:
                raise ValueError("Dimensions of input-field state do not agree with stored dimensions.")
        elif data is None:
            # No argument passed for input_env, create a vacuum QGstate.
            self._input_env = QGstate(data_2nd = (1/2)*np.identity(2*self.dims_env), 
                                      dims_cvs = self.dims_env)
        else:
            raise TypeError("Input-field state is not of a supported type: QGstate.")

    @property
    def dims_cvs(self) -> int:
        return self._dims_cvs
    @dims_cvs.setter
    def dims_cvs(self, dims):
        if isinstance(dims, numbers.Integral):
            self._dims_cvs = int(dims)
        else:
            raise TypeError("dims_cvs is not of a supported type: number.")

    @property
    def dims_env(self) -> int:
        return self._dims_env
    @dims_env.setter
    def dims_env(self, dims):
        if isinstance(dims, numbers.Integral):
            self._dims_env = int(dims)
        else:
            raise TypeError("dims_env is not of a supported type: number.")
    
    @property
    def dims_tot(self) -> int:
        return self.dims_env + self.dims_cvs
    
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
            raise TypeError("dims_fls is not of a supported type: array or list.")
        # Set isfls property.
        self.isfls = dims
    
    @property
    def isfls(self) -> bool:
        return self._isfls     
    @isfls.setter
    def isfls(self, dims):
        if dims == [[],[]] or dims == None:
            self._isfls = False
        else:
            self._isfls = True
    
    @cached_property
    def isherm(self) -> bool:
        return (self.h_sys.isherm and self.h_sys_env.isherm)
    
    @cached_property
    def isgauss(self) -> bool:
        return (self.h_sys.isgauss and self.h_sys_env.isgauss)

    @cached_property
    def iscoherent(self) -> bool:
        return (self.h_sys.isherm and 
                np.all(np.abs(self.h_se) < qgauss.settings.atol))
    
    @cached_property
    def symform_sys(self) -> npt.NDArray:
        # Generates the symplectic form for the system.
        return np.kron(np.identity(self.dims_cvs), np.array([[0,1],[-1,0]]))
    
    @cached_property
    def symform_env(self) -> npt.NDArray:
        # Generates the symplectic form for the environment.
        return np.kron(np.identity(self.dims_env), np.array([[0,1],[-1,0]]))
    
    @cached_property
    def lme(self) -> QGsuper:
        # Lindblad master equation in the form of a QGsuper, equivalent to
        # the Heisenberg-Langevin equations with specified input-field correlations.
        if not (self.isherm and self.isgauss and self.input_env.isdensity):
            warnings.warn("No CPTP and Gaussian-state preserving Lindblad " \
                          "master equation can be associated with this set " \
                          "Heisenberg-Langevin equations.")
        
        _l_sys = qgauss.coherent(self.h_sys)
        if self.isfls:
            _h_temp = np.zeros((np.prod(self.dims_fls[0]),
                                np.prod(self.dims_fls[1]),
                                2*self.dims_cvs),
                                dtype=complex)
            for x in range(0,np.prod(self.dims_fls[0])):
                _h_temp[x] = self.symform_sys @ self.h_se[x,x] @ self.input_env.data_1st
            _h_drv = QGoper(data_1st = _h_temp,
                            dims_cvs = self.dims_cvs,
                            dims_fls = self.dims_fls)

            _diss = np.einsum("jklm,mn,jknp->jklp",
                              self.h_se,
                              self.input_env.data_2nd + (1j/2)*self.symform_env,
                              self.h_se.transpose([0,1,3,2]))
            
            _c_mat = np.empty([np.prod(self.dims_fls[0]),
                               2*self.dims_cvs,
                               2*self.dims_cvs], 
                               dtype=complex)
            for x in range(0,np.prod(self.dims_fls[0])):
                _rate,_jump = np.linalg.eigh(_diss[x,x])
                trim_abs(_rate), trim(_jump)
                
                if np.any(np.real(_rate) < 0):
                    warnings.warn("System has negative decay rates. " \
                                  "A CPTP Lindbladian cannot be constructed.")
                
                _c_mat[x] = np.diag(np.emath.sqrt(_rate)) @ _jump.conj().T

            _c_ops = [None for x in range(0,2*self.dims_cvs)]
            for x in range(0,2*self.dims_cvs):
                _c_temp = np.zeros((np.prod(self.dims_fls[0]),
                                    np.prod(self.dims_fls[1]),
                                    2*self.dims_cvs),
                                    dtype=complex)
                for y in range(0,np.prod(self.dims_fls[0])):
                    _c_temp[y,y] = _c_mat[y][x]
                _c_ops[x] = QGoper(data_1st = _c_temp,
                                   dims_cvs = self.dims_cvs,
                                   dims_fls = self.dims_fls)

        else:
            _h_temp = self.symform_sys @ self.h_se @ self.input_env.data_1st
            _h_drv = QGoper(data_1st = _h_temp,
                            dims_cvs = self.dims_cvs)
            
            _diss = (self.h_se
                     @ (self.input_env.data_2nd + (1j/2)*self.symform_env)
                     @ self.h_se.T)
            _rate,_jump = np.linalg.eigh(_diss)
            _jump = _jump.conj().T
            trim_abs(_rate), trim(_jump)

            if np.any(np.real(_rate) < 0):
                warnings.warn("System has negative decay rates. " \
                              "A CPTP Lindbladian cannot be constructed.")
                
            _c_ops = [QGoper(data_1st = np.emath.sqrt(_rate[x])*_jump[x],
                             dims_cvs = self.dims_cvs)
                      for x in range(0,len(_rate))]

        # Combine the coherent term from the effective drive Hamiltonian which
        # has been generated from the input-field state means with the
        # dissipation generated from the input-field state covariances.
        _l_sys_env = (qgauss.coherent(_h_drv)
                      + sum([qgauss.dissipator(x) for x in _c_ops]))
            
        return _l_sys + _l_sys_env

    '''
    ---------------
        Methods
    ---------------
    '''

    ### Addition and subtraction of QGhles ###

    def __add__(self, other: QGhle) -> QGhle:
        # Addition with self.QGhle on the left
        if isinstance(other, QGhle):
            if ((self.dims_cvs == other.dims_cvs) and 
                (self.dims_fls == other.dims_fls) and
                (self.dims_env == other.dims_env) and
                (self.input_env == other.input_env)
                ):
                return QGhle(h_sys = self.h_sys + other.h_sys,
                             h_sys_env = self.h_sys_env + other.h_sys_env,
                             input_env = self.input_env,
                             dims_cvs = self.dims_cvs,
                             dims_fls = self.dims_fls,
                             dims_env = self.dims_env)
            else:
                raise ValueError("Cannot perform addition operation between " \
                "QGhles with different dimensions or input correlations.")
        elif other == 0:
            return QGhle(self)
        else:
            raise TypeError("Cannot perform addition operation between the " \
            "types QGhle and " + type(other).__name__ + ".")

    def __radd__(self, other: QGhle) -> QGhle:
        # Addition with the self.QGhle on the right
        return self.__add__(other)

    def __sub__(self, other: QGhle) -> QGsuper:
        # Subtraction with self.QGhle on the left
        return self.__add__(other.__neg__())

    def __rsub__(self, other: QGhle) -> QGhle:
        # Subtraction with self.QGsuper on the right
        return (self.__neg__()).__add__(other)
    
    def __neg__(self) -> QGhle:
        # Negation of self.QGhle
        return QGhle(h_sys = -self.h_sys,
                     h_sys_env = -self.h_sys_env,
                     input_env = self.input_env,
                     dims_cvs = self.dims_cvs,
                     dims_fls = self.dims_fls,
                     dims_env = self.dims_env)
    
    ### Multiplication and division of QGhles ###

    def __mul__(self, other: complex) -> QGhle:
        # Multiplication of number with self.QGhle on the left
        if isinstance(other, (numbers.Number, np.number)):
            return QGhle(h_sys = other*self.h_sys,
                         h_sys_env = other*self.h_sys_env,
                         input_env = self.input_env,
                         dims_cvs = self.dims_cvs,
                         dims_fls = self.dims_fls,
                         dims_env = self.dims_env)
        else:
            raise TypeError("Cannot perform multiplication operation between " \
            "the types QGhle and " + type(other).__name__ + ".")

    def __rmul__(self, other: complex) -> QGhle:
        # Multiplication with self.QGhle on the right
        if isinstance(other, (numbers.Number, np.number)):
            return self.__mul__(other)
        else:
            raise TypeError("Cannot perform multiplication operation between " \
            "the types QGhle and " + type(other).__name__ + ".")

    def __truediv__(self, other: complex) -> QGhle:
        # Division of self.QGhle by number
        if isinstance(other, (numbers.Number,np.number)):
            return QGhle(h_sys = self.h_sys/other,
                         h_sys_env = self.h_sys_env/other,
                         input_env = self.input_env,
                         dims_cvs = self.dims_cvs,
                         dims_fls = self.dims_fls,
                         dims_env = self.dims_env)
        else:
            raise TypeError("Cannot perform division operation between the " \
            "types QGhle and " + type(other).__name__ + ".")
        
    ### Assorted Methods ###  
    
    def scattering_matrix(self,
                          freq: float = 0,
                          index: int = None
                         ) -> npt.NDArray:
        if not self.isgauss:
            raise AttributeError("System is not Gaussian-state preserving " \
            "and so no scattering matrix can be constructed.")
        
        if self.isfls:
            if index is None:
                _A = [(self.symform_sys @ self.h_sys[index,index].data_2nd
                      + (1/2)*(self.symform_sys @ self.h_se[index,index]
                               @ self.symform_env @ self.h_se[index,index].T))
                      for x in range(0,np.prod(self.dims_fls[0]))]
                QGhle._isstable(_A)

                return [(- self.symform_env
                         @ self.h_se[x,x].T
                         @ np.linalg.inv(_A[x] + 1j*freq*np.identity(2*self.dims_cvs))
                         @ self.symform_sys
                         @ self.h_se[x,x]
                         + np.identity(2*self.dims_env))
                        for x in range(0,np.prod(self.dims_fls[0]))]
            else:
                _A = (self.symform_sys @ self.h_sys.data_2nd[index,index]
                      + (1/2)*(self.symform_sys @ self.h_se[index,index]
                               @ self.symform_env @ self.h_se[index,index].T))
                QGhle._isstable(_A)

                return (- self.symform_env
                        @ self.h_se[index,index].T
                        @ np.linalg.inv(_A + 1j*freq*np.identity(2*self.dims_cvs))
                        @ self.symform_sys
                        @ self.h_se[index,index]
                        + np.identity(2*self.dims_env))
        else:
            _A = (self.symform_sys @ self.h_sys.data_2nd
                  + (1/2)*(self.symform_sys @ self.h_se
                           @ self.symform_env @ self.h_se.T))
            QGhle._isstable(_A)

            return (- self.symform_env
                    @ self.h_se.T
                    @ np.linalg.inv(_A + 1j*freq*np.identity(2*self.dims_cvs))
                    @ self.symform_sys
                    @ self.h_se 
                    + np.identity(2*self.dims_env))
        
    def transfer_matrix(self,
                        freq: float = 0,
                        index: int = None
                       ) -> npt.NDArray:
        if not self.isgauss:
            raise AttributeError("System is not Gaussian-state preserving " \
            "and so no transfer matrix can be constructed.")

        if self.isfls:
            if index is None:
                _A = [(self.symform_sys @ self.h_sys.data_2nd[x,x]
                      + (1/2)*(self.symform_sys @ self.h_se[x,x]
                               @ self.symform_env @ self.h_se[x,x].T))
                      for x in range(0,np.prod(self.dims_fls[0]))]
                QGhle._isstable(_A)

                return [(- self.symform_env
                        @ self.h_se[x,x].T
                        @ np.linalg.inv(_A[x] + 1j*freq*np.identity(2*self.dims_cvs)))
                        for x in range(0,np.prod(self.dims_fls[0]))]
            else:
                _A = (self.symform_sys @ self.h_sys.data_2nd[index,index]
                      + (1/2)*(self.symform_sys @ self.h_se[index,index]
                               @ self.symform_env @ self.h_se[index,index].T))
                QGhle._isstable(_A)

                return (- self.symform_env
                        @ self.h_se[index,index].T
                        @ np.linalg.inv(_A + 1j*freq*np.identity(2*self.dims_cvs)))
        else:
            _A = (self.symform_sys @ self.h_sys.data_2nd
                  + (1/2)*(self.symform_sys @ self.h_se
                           @ self.symform_env @ self.h_se.T))
            QGhle._isstable(_A)
            
            return (- self.symform_env
                    @ self.h_se.T
                    @ np.linalg.inv(_A + 1j*freq*np.identity(2*self.dims_cvs)))

    def output_env(self, 
                   freq: float = 0,
                   index: int = None
                  ) -> QGstate:
        if not self.isgauss:
            raise AttributeError("System is not Gaussian-state preserving " \
            "and so no output-field state can be constructed.")

        if self.isfls:
            if index is None:         
                _out_mean = np.zeros((np.prod(self.dims_fls[0]),
                                      np.prod(self.dims_fls[1]),
                                      2*dims_env))
                _out_cov = np.zeros((np.prod(self.dims_fls[0]),
                                     np.prod(self.dims_fls[1]),
                                     2*dims_env,
                                     2*dims_env))
                
                for x in range(0,np.prod(self.dims_fls[0])):
                    _smat = self.scattering_matrix(freq, x)
                    _smat_neg = self.scattering_matrix(-freq, x)
                    _tmat = self.transfer_mat(freq, x)

                    _out_mean[x,x] = \
                    (_smat @ self.input_env.data_1st
                     + _tmat @ self.symform_sys @ self.h_sys.data_1st[x,x])
                    _out_cov[x,x] = \
                    (1/2)*(_smat @ self.input_env.data_2nd @ _smat_neg.T
                           + _smat_neg @ self.input_env.data_2nd @ _smat.T)
                
                return QGstate(data_2nd = _out_cov,
                               data_1st = _out_mean,
                               dims_cvs = self.dims_env,
                               dims_fls = self.dims_fls)
            else:
                _smat = self.scattering_matrix(freq, index)
                _smat_neg = self.scattering_matrix(-freq, index)
                _tmat = self.transfer_matrix(freq, index)

                _out_mean \
                = (_smat @ self.input_env.data_1st
                   + _tmat @ self.symform_sys @ self.h_sys.data_1st[index,index])
                _out_cov = \
                (1/2)*(_smat @ self.input_env.data_2nd @ _smat_neg.T
                       + _smat_neg @ self.input_env.data_2nd @ _smat.T)
                
                return QGstate(data_2nd = _out_cov,
                               data_1st = _out_mean,
                               dims_cvs = self.dims_env)
        else:
            _smat = self.scattering_matrix(freq)
            _smat_neg = self.scattering_matrix(-freq)
            _tmat = self.transfer_matrix(freq)

            _out_mean = (_smat @ self.input_env.data_1st
                         + _tmat @ self.symform_sys @ self.h_sys.data_1st)
            _out_cov = (1/2)*(_smat @ self.input_env.data_2nd @ _smat_neg.T
                              + _smat_neg @ self.input_env.data_2nd @ _smat.T)
            
            return QGstate(data_2nd = _out_cov,
                           data_1st = _out_mean,
                           dims_cvs = self.dims_env)

    def __eq__(self, other: QGhle) -> bool:
        # Check equality of QGhle
        if (isinstance(other, QGhle) and 
            (self.dims_cvs == other.dims_cvs) and
            (self.dims_fls == other.dims_fls) and
            (self.dims_env == other.dims_env) and
            (self.h_sys == other.h_sys) and
            (self.h_sys_env == self.h_sys_env) and
            (self.input_env == other.input_env)
            ):
                return True
        else:
            return False

    def __getitem__(self, index) -> QGhle:
        # Grab CV elements from self at index in the FLS component
        # and return a QGhle with a CV component only
        if self.isfls:
            return QGhle(h_sys = self.h_sys[index],
                         h_sys_env = self.h_sys_env[index],
                         input_env = self.input_env,
                         dims_cvs = self.dims_cvs,
                         dims_env = self.dims_env)
        else:
            raise ValueError("QGhle requires an FLS component to " \
            "use this method. Access QGhle data arrays individually if " \
            "specific elements are required.")
        
    def tidyup(self, tol: float = qgauss.settings.tidyup_atol) -> QGhle:
        # Private void function to remove small magnitude elements from the data.
        self.h_se.real[np.abs(self.h_se.real) < tol] = 0
        self.h_se.imag[np.abs(self.h_se.imag) < tol] = 0

        self.h_sys.tidyup()
        self.h_sys_env.tidyup()
        self.input_env.tidyup()

    @staticmethod
    def _isstable(data: npt.NDArray):
        if np.any(np.real(np.linalg.eigvals(data)) >= 0):
            warnings.warn("System has no steady-state solution. " \
            "Program will proceed, but results will not be physically valid.")
        else:
            pass
