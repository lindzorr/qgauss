from __future__ import annotations

import numbers
import numpy.typing as npt
from functools import cached_property
import qgauss
import numpy as np

from .qgstate import QGstate
from .qgoper import QGoper
from .qgsuper import QGsuper

__all__ = ['QGhle']


class QGhle(object):

    """
    ---- Structure ----
    Write this and add more comments to code.
    
    ---- Parameters ----
    inpt : QGhle
        Create a copy of another QGhle.
    h_sys : QGoper
        System Hamiltonian, containing interactions between the CVS and FLS 
        parts of the main system. If system operators are found to be tensored 
        with bath operators, the bath modes will be dropped.
    h_sys_bath : QGoper
        System-bath Hamiltonian, representing coupling between the system and a
        Markovian bath. The Hamiltonian coupling is therefore independent of the
        frequency of the bath modes.
    in_bath : QGstate
        In-field state of the input bath operators, representing the 
        correlations of the Markovian noise from the bath. Delta-correlations
        part of the covariances are ignored.
    dims_cvs : int
        Number of continuous variable system (CVS) modes.
    dims_fls : array_like
        Dimensions of the finite-level component of the system (FLS).
    dims_bath : int
        Number of continuous variable environments, each of which is comprised
        of an infinite number of bosonic modes.
    decay_rate_mat : array_like
        Alternative to h_sys_bath. Represents the coupling to the input-fields
        in the Heisenberg-Langevin equations:
            dr/dt = A.r - sqrt(k).r_in
        The argument decay_rate_mat corresponds to the array sqrt(k), from which
        h_sys_bath is then constructed. Cannot be used for systems where 
        dissipation is dependent on the FLS system, 
            
    ---- Attributes ----
    h_sys : QGoper
    h_sys_bath : QGoper
    h_sb : array
    in_bath : QGstate
    lme : QGsuper
        Lindblad master equation equivalent to the Heisenberg-Langevin equations
        for the internal modes of the system. Both h_sys and h_sys_bath must be
        Hermitian for the Lindbladian to be CPTP, and both must be diagonal in 
        any FLS-basis, or else no Gaussian presrving superoperator will be
        constructed.
    dims_cvs : int
    dims_fls : list[list[int]]
    dims_bath : int
    symform_sys : array
        Symplectic form acting on either the system CVS modes. The
        generated matrix will have the form:
            Ω = ⊗_{j=1}^dims_cvs [[0,1],[-1,0]]
    symform_bath : array
        Symplectic form acting on either the bath degrees of freedom. The
        generated matrix will have the form:
            Ω = ⊗_{j=1}^dims_bath [[0,1],[-1,0]]

    ---- Methods ----
    scattering_mat : (QGhle, float, str) -> array
    transfer_mat : (QGhle, float, str) -> array
    out_bath : (QGhle, float, str) -> QGstate
    eq : (QGhle, QGhle) -> bool
        Check equality of two QGhles.
    
    """

    ### Quantum-Gaussian Heisenberg-Langevin Equation (QGhle) Initialisation ###
    def __init__(self, 
                 inpt: QGhle = None,
                 h_sys: QGoper = None,
                 h_sys_bath: QGoper = None,
                 in_bath: QGstate = None,
                 dims_cvs: int = None,
                 dims_fls: list[list[int]] = None,
                 dims_bath: int = None,
                 decay_rate_mat: npt.ArrayLike = None
                ):

        # QGhle as input, copy data.
        if isinstance(inpt, QGoper):
            self._dims_cvs = inpt.dims_cvs
            self._dims_fls = inpt.dims_fls
            self._dims_bath = inpt.dims_bath

            self._h_sys = inpt.h_sys
            self._h_sys_bath = inpt.h_sys_bath
            self._in_bath = inpt.in_bath

        # In other cases, specific components of QGoper must be passed as
        # arguments. The exceptions are h_sys_bath and decay_rate_mat, where
        # only one need be provided.
        elif inpt is None:
            # Set dimensions of FLS and CVS components, along with the number
            # of bath degrees of freedom. Currently, these parameters must be 
            # specified for initialization to proceed, and cannot be inferred
            # from the data.
            self.dims_fls = dims_fls
            self.dims_cvs = dims_cvs
            self.dims_bath = dims_bath

            # Set data structures from inputs
            self.h_sys = h_sys
            self.in_bath = in_bath

            # decay_rate_mat is a convenience argument to pass as an alternative
            # to constructing the system-bath Hamiltonian h_sys_bath explicitly,
            # and will only be used if no argument for h_sys_bath is passed.
            if h_sys_bath is not None:
                self.h_sys_bath = h_sys_bath
            else:
                self.h_sys_bath = decay_rate_mat

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
            # h_sys to initially tensor the system and bath operators, a h_sys 
            # with dimensions self.dims_cvs + self.dims_bath is also accepted. 
            # In this case, the bath modes will be dropped from h_sys.
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
    def h_sys_bath(self) -> QGoper:
        return self._h_sys_bath
    @h_sys_bath.setter
    def h_sys_bath(self, data) -> QGoper:
        if isinstance(data, QGoper):
            # If a QGoper is passed, assume this represents a system-bath Hamiltonian.
            # Note: The first check requires calculating cached properties, 
            # which may be slow if we are repeadedly creating instances of 
            # QGhle. Find a way around this, or else find out how such
            # couplings can be incorporated into the construction of the class.

            # First, check that only bilinear couplings are present. Couplings
            # between pairs of system modes and pairs of bath modes are 
            # currently not checked, and will simply be ignored.
            if data.is1st or data.is0th:
                raise ValueError("h_sys_bath can only contain couplings " \
                "between system and bath operators.")
            
            # Next, check that the dimensions of data agree with combined
            # system-CVS and bath dimensions of self.
            # FLS dimensions of data must either agree with those of self, 
            # or else the system-bath Hamiltonian is not coupled to an FLS.
            # In this case, data is tensored with an identity QGoper.
            if (self.dims_cvs == data.dims_tot and
                self.dims_fls == data.dims_fls
               ):
                self._h_sys_bath = data
            elif (self.dims_cvs == data.dims_tot and
                  self._isfls and not data.isfls
                 ):
                self._h_sys_bath = \
                qgauss.tensor(data,
                              QGoper(data_0th = np.identity(np.prod(self.dims_fls[0])),
                                     dims_fls = self.dims_fls))
            else:
                raise ValueError("Dimensions of h_sys_bath do not agree with stored dimensions.")     
        elif isinstance(data, (np.ndarray, list)):
            # If an array-like structure is passed, assume this represents the
            # decay-rate coupling matrix between the system and input-fields
            _data = self.symform_sys @ np.asarray(data)
            if (_data.shape[0] == 2*self.dims_cvs and
                _data.shape[1] == 2*self.dims_bath and
                _data.ndim == 2
               ):
                _data_2nd = np.block([[np.zeros((2*self.dims_cvs, 2*self.dims_cvs)),
                                       _data],
                                      [np.transpose(_data),
                                       np.zeros((2*self.dims_bath, 2*self.dims_bath))]])
                if self.isfls:
                    self._h_sys_bath = \
                    qgauss.tensor(QGoper(data_2nd = _data_2nd,
                                         dims_cvs = self.dims_tot),
                                  QGoper(data_0th = np.identity(np.prod(self.dims_fls[0])),
                                         dims_fls = self.dims_fls))
                else:
                    self._h_sys_bath = \
                    QGoper(data_2nd = _data_2nd,
                           dims_cvs = self.dims_tot)
            else:
                raise ValueError("Dimensions of decay_rate_mat cannot be " \
                "brought into agreement with stored dimensions.")
        elif data is None:
            # No argument passed, create an empty QGoper
            self._h_sys_bath = QGoper(dims_fls = self.dims_fls,
                                      dims_cvs = self.dims_tot)
        else:
           raise TypeError("Input is not of a supported type: QGoper or array-like.")
        
        # Set h_sb property immediately.
        if self.isfls == True:
            self._h_sb = \
            np.asarray([[self._h_sys_bath[x,y].data_2nd[0:2*self.dims_cvs,
                                                        2*self.dims_cvs:2*self.dims_tot]
                         for x in range(0,np.prod(self.dims_fls[0]))]
                         for y in range(0,np.prod(self.dims_fls[1]))])
        else:
            self._h_sb = \
            self._h_sys_bath.data_2nd[0:2*self.dims_cvs,
                                      2*self.dims_cvs:2*self.dims_tot]
             
    @property
    def h_sb(self) -> npt.NDArray:
        return self._h_sb
                   
    @property
    def in_bath(self) -> QGstate:
        return self._in_bath
    @in_bath.setter
    def in_bath(self, data):
        # Initialize QGstate which represents the input state of the bath.
        if isinstance(data, QGstate):
            # in_bath must have the dimensions speficied by self.dims_bath.
            if data.isfls is True:
                raise ValueError("Bath in-field state can only have a CVS component.")
            elif data.dims_cvs == self.dims_bath:
                self._in_bath = data
            else:
                raise ValueError("Dimensions of bath in-field state do not agree with stored dimensions.")
        elif data is None:
            # No argument passed for in_bath, create a vacuum QGstate.
            self._in_bath = QGstate(data_2nd = (1/2)*np.identity(2*self.dims_bath), 
                                    dims_cvs = self.dims_bath)
        else:
            raise TypeError("Bath in-field state is not of a supported type: QGstate.")

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
    def dims_bath(self) -> int:
        return self._dims_bath
    @dims_bath.setter
    def dims_bath(self, dims):
        if isinstance(dims, numbers.Integral):
            self._dims_bath = int(dims)
        else:
            raise TypeError("dims_bath is not of a supported type: number.")
    
    @property
    def dims_tot(self) -> int:
        return self.dims_bath + self.dims_cvs
    
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
        return (self.h_sys.isherm and self.h_sys_bath.isherm)
    
    @cached_property
    def isgauss(self) -> bool:
        return (self.h_sys.isgauss and self.h_sys_bath.isgauss)

    @cached_property
    def symform_sys(self) -> npt.NDArray:
        # Generates the symplectic form for the system.
        return np.kron(np.identity(self.dims_cvs), np.array([[0,1],[-1,0]]))
    
    @cached_property
    def symform_bath(self) -> npt.NDArray:
        # Generates the symplectic form for the bath.
        return np.kron(np.identity(self.dims_bath), np.array([[0,1],[-1,0]]))

    @cached_property
    def lme(self) -> QGsuper:
        # Lindblad master equation in the form of a QGsuper, equivalent to
        # the Heisenberg-Langevin equations and in-field bath correlations.
        if not (self.isherm and self.isgauss):
            raise AttributeError("No CPTP and Gaussian-state preserving " \
            "Lindblad master equation can be associated with this set of " \
            "Heisenberg-Langevin equations.")
        
        _l_sys = qgauss.coherent(self.h_sys)
        if self.isfls:
            _h_temp = np.zeros((np.prod(self.dims_fls[0]),
                                np.prod(self.dims_fls[1]),
                                2*self.dims_cvs),
                                dtype=complex)
            for x in range(0,np.prod(self.dims_fls[0])):
                _h_temp[x] = self.symform_sys @ self.h_sb[x,x] @ self.in_bath.data_1st
            _h_drv = QGoper(data_1st = _h_temp,
                            dims_cvs = self.dims_cvs,
                            dims_fls = self.dims_fls)

            _diss = np.einsum("jklm,mn,jknp->jklp",
                              self.h_sb,
                              self.in_bath.data_2nd + (1j/2)*self.symform_bath,
                              np.transpose(self.h_sb, [0,1,3,2]))
            
            _c_mat = np.empty([np.prod(self.dims_fls[0]),
                               2*self.dims_cvs,
                               2*self.dims_cvs], 
                               dtype=complex)
            for x in range(0,np.prod(self.dims_fls[0])):
                _rate,_jump = np.linalg.eigh(_diss[x,x])
                _c_mat[x] = np.diag(np.sqrt(_rate)) @ np.conj(np.transpose(_jump))

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
            _h_temp = self.symform_sys @ self.h_sb @ self.in_bath.data_1st
            _h_drv = QGoper(data_1st = _h_temp,
                            dims_cvs = self.dims_cvs)

            _diss = (self.h_sb
                     @ (self.in_bath.data_2nd + (1j/2)*self.symform_bath)
                     @ np.transpose(self.h_sb))
            _rate,_jump = np.linalg.eigh(_diss)
            _jump = np.conj(np.transpose(_jump))

            _c_ops = [QGoper(data_1st = np.sqrt(_rate[x])*_jump[x],
                             dims_cvs = self.dims_cvs)
                      for x in range(0,len(_rate))]

        # Combine the coherent term from the effective drive Hamiltonian which
        # has been generated from the in-field bath state means with the
        # dissipation generated from the in-field bath state covariances.
        _l_sys_bath = (qgauss.coherent(_h_drv)
                       + sum([qgauss.dissipator(x) for x in _c_ops]))
            
        return _l_sys + _l_sys_bath

    '''
    ---------------
        Methods
    ---------------
    '''

    ### Assorted Methods ###  
    
    def scattering_mat(self, 
                       freq: float = 0, 
                       index: int = None
                      ) -> npt.NDArray:
        if not self.isgauss:
            raise AttributeError("System is not Gaussian-state preserving " \
            "and so no scattering matrix can be constructed.")
        
        if self.isfls:
            if index is None:
                _A = [(self.symform_sys @ self.h_sys[index,index].data_2nd
                      + (1/2)*(self.symform_sys @ self.h_sb[index,index]
                               @ self.symform_bath @ np.transpose(self.h_sb[index,index])))
                      for x in range(0,np.prod(self.dims_fls[0]))]
                return [(- self.symform_bath
                         @ np.transpose(self.h_sb[x,x])
                         @ np.linalg.inv(_A[x] + 1j*freq*np.identity(2*self.dims_cvs))
                         @ self.symform_sys
                         @ self.h_sb[x,x]
                         + np.identity(2*self.dims_bath))
                        for x in range(0,np.prod(self.dims_fls[0]))]
            else:
                _A = (self.symform_sys @ self.h_sys.data_2nd[index,index]
                      + (1/2)*(self.symform_sys @ self.h_sb[index,index]
                               @ self.symform_bath @ np.transpose(self.h_sb[index,index])))
                return (- self.symform_bath
                        @ np.transpose(self.h_sb[index,index])
                        @ np.linalg.inv(_A + 1j*freq*np.identity(2*self.dims_cvs))
                        @ self.symform_sys
                        @ self.h_sb[index,index]
                        + np.identity(2*self.dims_bath))
        else:
            _A = (self.symform_sys @ self.h_sys.data_2nd
                  + (1/2)*(self.symform_sys @ self.h_sb
                           @ self.symform_bath @ np.transpose(self.h_sb)))
            return (- self.symform_bath
                    @ np.transpose(self.h_sb)
                    @ np.linalg.inv(_A + 1j*freq*np.identity(2*self.dims_cvs))
                    @ self.symform_sys
                    @ self.h_sb 
                    + np.identity(2*self.dims_bath))
        
    def transfer_mat(self, 
                     freq: float = 0,
                     index: int = None
                    ) -> npt.NDArray:
        if not self.isgauss:
            raise AttributeError("System is not Gaussian-state preserving " \
            "and so no transfer matrix can be constructed.")

        if self.isfls:
            if index is None:
                _A = [(self.symform_sys @ self.h_sys.data_2nd[x,x]
                      + (1/2)*(self.symform_sys @ self.h_sb[x,x]
                               @ self.symform_bath @ np.transpose(self.h_sb[x,x])))
                      for x in range(0,np.prod(self.dims_fls[0]))]
                return [(- self.symform_bath
                        @ np.transpose(self.h_sb[x,x])
                        @ np.linalg.inv(_A[x] + 1j*freq*np.identity(2*self.dims_cvs)))
                        for x in range(0,np.prod(self.dims_fls[0]))]
            else:
                _A = (self.symform_sys @ self.h_sys.data_2nd[index,index]
                      + (1/2)*(self.symform_sys @ self.h_sb[index,index]
                               @ self.symform_bath @ np.transpose(self.h_sb[index,index])))
                return (- self.symform_bath
                        @ np.transpose(self.h_sb[index,index])
                        @ np.linalg.inv(_A + 1j*freq*np.identity(2*self.dims_cvs)))
        else:
            _A = (self.symform_sys @ self.h_sys.data_2nd
                  + (1/2)*(self.symform_sys @ self.h_sb
                           @ self.symform_bath @ np.transpose(self.h_sb)))
            return (- self.symform_bath
                    @ np.transpose(self.h_sb)
                    @ np.linalg.inv(_A + 1j*freq*np.identity(2*self.dims_cvs)))

    def out_bath(self, 
                 freq: float = 0,
                 index: int = None
                ) -> QGstate:
        if not self.isgauss:
            raise AttributeError("System is not Gaussian-state preserving " \
            "and so no bath out-field state can be constructed.")

        if self.isfls:
            if index is None:         
                _out_mean = np.zeros((np.prod(self.dims_fls[0]),
                                      np.prod(self.dims_fls[1]),
                                      2*dims_bath))
                _out_cov = np.zeros((np.prod(self.dims_fls[0]),
                                     np.prod(self.dims_fls[1]),
                                     2*dims_bath,
                                     2*dims_bath))
                
                for x in range(0,np.prod(self.dims_fls[0])):
                    _smat = self.scattering_matrix(freq, x)
                    _smat_neg = self.scattering_matrix(-freq, x)
                    _tmat = self.transfer_mat(freq, x)

                    _out_mean[x,x] = \
                    (_smat @ self.in_bath.data_1st
                     + _tmat @ self.symform_sys @ self.h_sys.data_1st[x,x])
                    _out_cov[x,x] = \
                    (1/2)*(_smat @ self.in_bath.data_2nd[x,x] @ np.transpose(_smat_neg)
                           + _smat_neg @ self.in_bath.data_2nd[x,x] @ np.transpose(_smat))
                
                return QGstate(data_2nd = _out_cov,
                               data_1st = _out_mean,
                               dims_cvs = self.dims_bath,
                               dims_fls = self.dims_fls)
            else:
                _smat = self.scattering_matrix(freq, index)
                _smat_neg = self.scattering_matrix(-freq, index)
                _tmat = self.transfer_mat(freq, index)

                _out_mean \
                = (_smat @ self.in_bath.data_1st
                   + _tmat @ self.symform_sys @ self.h_sys.data_1st[index,index])
                _out_cov = \
                (1/2)*(_smat @ self.in_bath.data_2nd[index,index] @ np.transpose(_smat_neg)
                       + _smat_neg @ self.in_bath.data_2nd[index,index] @ np.transpose(_smat))
                
                return QGstate(data_2nd = _out_cov,
                               data_1st = _out_mean,
                               dims_cvs = self.dims_bath)
        else:
            _smat = self.scattering_matrix(freq)
            _smat_neg = self.scattering_matrix(-freq)
            _tmat = self.transfer_mat(freq)

            _out_mean = (_smat @ self.in_bath.data_1st
                         + _tmat @ self.symform_sys @ self.h_sys.data_1st)
            _out_cov = (1/2)*(_smat @ self.in_bath.data_2nd @ np.transpose(_smat_neg)
                              + _smat_neg @ self.in_bath.data_2nd @ np.transpose(_smat))
            
            return QGstate(data_2nd = _out_cov,
                           data_1st = _out_mean,
                           dims_cvs = self.dims_bath)

    def __eq__(self, other: QGhle) -> bool:
        # Check equality of QGhle
        if (isinstance(other, QGhle) and 
            (self.dims_fls == other.dims_fls) and
            (self.dims_cvs == other.dims_cvs) and
            (self.dims_bath == other.dims_bath) and
            (self.h_sys == other.h_sys) and
            (self.h_sb == other.h_sb) and
            (self.in_bath == other.in_bath)
            ):
                return True
        else:
            return False

    def tidyup(self, tol: float = qgauss.settings.tidyup_atol) -> QGhle:
        # Private void function to remove small magnitude elements from the data.
        np.real(self.h_sb)[np.abs(np.real(self.h_sb)) < tol] = 0
        np.imag(self.h_sb)[np.abs(np.imag(self.h_sb)) < tol] = 0

        self.h_sys.tidyup()
        self.h_sys_bath.tidyup()
        self.in_bath.tidyup()