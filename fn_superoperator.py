import sys
import warnings
import types
import typing
import numbers
import numpy.typing as npt

import qgauss
import numpy as np

from .qgoper import QGoper
from .qgsuper import QGsuper

__all__ = ['spre','spost','sprepost',
           'commutator_super','anticommutator_super',
           'dissipator','coherent','lindbladian'
          ]

""" Constructors used to create superoperators from operators. """

def dissipator(A: QGoper, 
               B: QGoper = None
               ) -> QGsuper:
    # Lindblad dissipation term: D[A,B](ρ) = A.ρ.B* - ½[B*.A,ρ]_+, where D[A,A] = D[A]
    if B is None:
        B = A
    return sprepost(A,B.dag()) - 0.5*spre(B.dag()*A) - 0.5*spost(B.dag()*A)


def coherent(H: QGoper) -> QGsuper:
    # Lindblad coherent evolution/von Neumann term: -i[H,ρ] = -i(H.ρ - H.ρ)
    return -1.0j*(spre(H) - spost(H))


def lindbladian(H: QGoper = None, 
                c_ops: list[QGoper]=[]
                ) -> QGsuper:
    # Lindblad superoperator, L(ρ) = -i[H,ρ] + Σ_{c_ops} D[c_ops](ρ)
    if H is not None:
        L = -1.0j*(spre(H) - spost(H))
    else:
        c0 = c_ops.pop(0)
        L = sprepost(c0,c0.dag()) - 0.5 * spre(c0.dag()*c0) - 0.5 * spost(c0.dag()*c0)
    L += sum([sprepost(c,c.dag()) - 0.5 * spre(c.dag()*c) - 0.5 * spost(c.dag()*c) for c in c_ops])
    return L


def commutator_super(H: QGoper) -> QGsuper:
    # Commutator superoperator: [H,ρ] = H.ρ - H.ρ
    return (spre(H) - spost(H))


def anticommutator_super(H: QGoper) -> QGsuper:
    # Anti-commutator superoperator: [H,ρ]_+ = H.ρ + H.ρ
    return (spre(H) + spost(H))


def spost(A: QGoper) -> QGsuper:
    # Superoperator representing post/right-multiplication of state by an operator: ρ.A
    if A.isfls:
        return QGsuper(data_2nd_r = np.einsum('jpkqyz,plqm->kljmyz',
                                              A.data_2nd[:,np.newaxis,:,np.newaxis,:,:],
                                              np.identity(np.prod(A.dims_fls[0]))[np.newaxis,:,np.newaxis,:]
                                              ).reshape(np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                        np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                        2*A.dims_cvs,
                                                        2*A.dims_cvs),
                       data_1st_r = np.einsum('jpkqz,plqm->kljmz',
                                              A.data_1st[:,np.newaxis,:,np.newaxis,:],
                                              np.identity(np.prod(A.dims_fls[0]))[np.newaxis,:,np.newaxis,:]
                                              ).reshape(np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                        np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                        2*A.dims_cvs),
                       data_0th = np.einsum('jpkq,plqm->kljm',
                                            A.data_0th[:,np.newaxis,:,np.newaxis],
                                            np.identity(np.prod(A.dims_fls[0]))[np.newaxis,:,np.newaxis,:]
                                            ).reshape(np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                      np.prod((A.dims_fls[0], A.dims_fls[1]))),
                       dims_cvs = A.dims_cvs,
                       dims_fls = [[A.dims_fls[0], A.dims_fls[1]],
                                   [A.dims_fls[0], A.dims_fls[1]]])
    else:
        return QGsuper(data_2nd_r = A.data_2nd,
                       data_1st_r = A.data_1st,
                       data_0th = A.data_0th,
                       dims_cvs = A.dims_cvs)


def spre(A: QGoper) -> QGsuper:
    # Superoperator representing pre/left-multiplication of state by an operator: A.ρ
    if A.isfls:
        return QGsuper(data_2nd_l = np.einsum('jpkq,plqmyz->jlkmyz',
                                              np.identity(np.prod(A.dims_fls[1]))[:,np.newaxis,:,np.newaxis],
                                              A.data_2nd[np.newaxis,:,np.newaxis,:,:,:]
                                              ).reshape(np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                        np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                        2*A.dims_cvs,
                                                        2*A.dims_cvs),
                       data_1st_l = np.einsum('jpkq,plqmz->jlkmz',
                                              np.identity(np.prod(A.dims_fls[1]))[:,np.newaxis,:,np.newaxis],
                                              A.data_1st[np.newaxis,:,np.newaxis,:,:]
                                              ).reshape(np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                        np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                        2*A.dims_cvs),
                       data_0th = np.einsum('jpkq,plqm->jlkm',
                                            np.identity(np.prod(A.dims_fls[1]))[:,np.newaxis,:,np.newaxis],
                                            A.data_0th[np.newaxis,:,np.newaxis,:]
                                            ).reshape(np.prod((A.dims_fls[0], A.dims_fls[1])),
                                                      np.prod((A.dims_fls[0], A.dims_fls[1]))),
                       dims_cvs = A.dims_cvs,
                       dims_fls = [[A.dims_fls[0], A.dims_fls[1]],
                                   [A.dims_fls[0], A.dims_fls[1]]])                   
    else:
        return QGsuper(data_2nd_l = A.data_2nd,
                       data_1st_l = A.data_1st,
                       data_0th = A.data_0th,
                       dims_cvs = A.dims_cvs)


def sprepost(A: QGoper, 
             B: QGoper
             ) -> QGsuper:
    # Superoperator representing pre/left and post-right-multiplication of state by an operator: A.ρ.B
    if (A.dims_cvs != B.dims_cvs) and (A.dims_fls != B.dims_fls).all():
        raise ValueError("Inputs do not have identical dimensions")

    if ((A.is2nd and B.is2nd) or
        (A.is2nd and B.is1st) or
        (A.is1st and B.is2nd)
        ):
        raise ValueError("Inputs result in superoperator which is not Gaussian")

    if A.isfls and B.isfls:
        return QGsuper(data_2nd_l = np.einsum('jpkq,plqmyz->kljmyz',
                                              B.data_0th[:,np.newaxis,:,np.newaxis],
                                              A.data_2nd[np.newaxis,:,np.newaxis,:,:,:]
                                              ).reshape(np.prod((A.dims_fls[0], B.dims_fls[1])),
                                                        np.prod((A.dims_fls[1], B.dims_fls[0])),
                                                        2*A.dims_cvs,
                                                        2*A.dims_cvs),
                       data_2nd_r = np.einsum('jpkqyz,plqm->kljmyz',
                                              B.data_2nd[:,np.newaxis,:,np.newaxis,:,:],
                                              A.data_0th[np.newaxis,:,np.newaxis,:]
                                              ).reshape(np.prod((A.dims_fls[0], B.dims_fls[1])),
                                                        np.prod((A.dims_fls[1], B.dims_fls[0])),
                                                        2*B.dims_cvs,
                                                        2*B.dims_cvs),
                       data_2nd_m = np.einsum('jpkqz,plqmy->kljmyz',
                                              B.data_1st[:,np.newaxis,:,np.newaxis,:],
                                              A.data_1st[np.newaxis,:,np.newaxis,:,:]
                                              ).reshape(np.prod((A.dims_fls[0], B.dims_fls[1])),
                                                        np.prod((A.dims_fls[1], B.dims_fls[0])),
                                                        2*A.dims_cvs,
                                                        2*B.dims_cvs),
                       data_1st_l = np.einsum('jpkq,plqmz->kljmz',
                                              B.data_0th[:,np.newaxis,:,np.newaxis],
                                              A.data_1st[np.newaxis,:,np.newaxis,:,:]
                                              ).reshape(np.prod((A.dims_fls[0], B.dims_fls[1])),
                                                        np.prod((A.dims_fls[1], B.dims_fls[0])),
                                                        2*A.dims_cvs),
                       data_1st_r = np.einsum('jpkqz,plqm->kljmz',
                                              B.data_1st[:,np.newaxis,:,np.newaxis,:],
                                              A.data_0th[np.newaxis,:,np.newaxis,:]
                                              ).reshape(np.prod((A.dims_fls[0], B.dims_fls[1])),
                                                        np.prod((A.dims_fls[1], B.dims_fls[0])),
                                                        2*B.dims_cvs),
                       data_0th = np.einsum('jpkq,plqm->kljm',
                                            B.data_0th[:,np.newaxis,:,np.newaxis],
                                            A.data_0th[np.newaxis,:,np.newaxis,:]
                                            ).reshape(np.prod((A.dims_fls[0], B.dims_fls[1])),
                                                      np.prod((A.dims_fls[1], B.dims_fls[0]))),
                       dims_cvs = A.dims_cvs,
                       dims_fls = [[A.dims_fls[0], B.dims_fls[1]],
                                   [A.dims_fls[1], B.dims_fls[0]]])
    else:
        return QGsuper(data_2nd_l = A.data_2nd*B.data_0th,
                       data_2nd_r = A.data_0th*B.data_2nd,
                       data_2nd_m = np.einsum('j,k->jk',A.data_1st,B.data_1st),
                       data_1st_l = A.data_1st*B.data_0th,
                       data_1st_r = A.data_0th*B.data_1st,
                       data_0th = A.data_0th*B.data_0th,
                       dims_cvs = A.dims_cvs)