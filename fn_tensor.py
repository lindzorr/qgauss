import types
import warnings
import numbers
import numpy as np

from .qgoper import *
from .qgstate import *

__all__ = ['tensor']

"""
Function handling tensor operation between states (QGstate) or operators (QGoper). Tensor operation for superoperators
(QGsuper) is currently not implemented. Both classes can call the tensor operation using the class method "and" (&).
Due to the fact that continuous-variable system (CVS) and finite-level system (FLS) components are handled differently, 
there are certain situation where ordering within tensor will not matter, for example:
    tensor(destroy(),sigmaz()) == tensor(sigmaz(),destroy())
    tensor(destroy(),create(),sigmaz()) == tensor(destroy(),sigmaz(),create()) == tensor(sigmaz(),destroy(),create())
However, we of course have:
    tensor(destroy(),create()) =/= tensor(create(),destroy())
It is recommended to keep ordering consistent even in cases where the tensor function is agnostic.
"""

def tensor(*args):
    if not args:
        raise TypeError("Tensor function requires at least one input argument")
    elif len(args) == 1 and isinstance(args[0], QGoper | QGstate):
        # If only one argument is provided, return copy
        return args[0]
    elif len(args) == 1 and isinstance(args[0], list):
        # List passed as arguments, convert to tuple
        args = tuple(args[0])

    # Check that every element of args is a QGoper or QGstate and call correct function
    if all(isinstance(x, QGoper) for x in args):
        return _tensor_oper(args)
    elif all(isinstance(x, QGstate) for x in args):
        return _tensor_state(args)
    else:
        raise TypeError("All operands must be of same type, either QGoper or QGstate.")


def _tensor_oper(args):
    # Check and raise error if an empty QGoper, one that is not iscv or isfls, is in args
    arg_cvs = np.where(tuple(map(lambda x: not x.iscvs, args)))[0]
    arg_fls = np.where(tuple(map(lambda x: not x.isfls, args)))[0]

    if len(set(arg_cvs) & set(arg_fls)) != 0:
        raise ValueError("Tensor product cannot be performed with empty QGopers")

    # Check and raise error if tensor will return an operator which is greater than quadratic in the quadrature basis
    # Three checks are performed:
    # (1) Whether two or more operators are quadratic in quadrature basis
    # (2) Whether three or more operators are linear in quadrature basis
    # (3) If one operator is of quadratic order, then no other operator may be of linear order
    arg_2nd = np.where(tuple(map(lambda x: x.is2nd, args)))[0]
    arg_1st = np.where(tuple(map(lambda x: x.is1st, args)))[0]

    if ((len(arg_2nd) >= 2) or 
        (len(arg_1st) >= 3) or
        (len(arg_2nd) == 1 and len(arg_1st) >= 1 and len(set(arg_2nd) ^ set(arg_1st)) > 0)
       ):
        raise ValueError("Tensor product of QGopers produces result which is beyond "
                         +"quadratic order in the quadrature operators")
        
    # Set data from first argument as initial data for output
    # Data from args will be used to update these values, with the QGoper
    # constructor called only after interating through all of the args
    _out_data_2nd = args[0].data_2nd
    _out_data_1st = args[0].data_1st
    _out_data_0th = args[0].data_0th
    
    _out_dims_cvs = args[0].dims_cvs
    _out_dims_fls = args[0].dims_fls
    _out_isfls = args[0].isfls

    for _elm in args[1:]:
        if not _out_isfls and not _elm.isfls:
            _out_data_2nd = (_elm.data_0th*np.pad(_out_data_2nd, 
                                                  ((0, 2*_elm.dims_cvs), (0, 2*_elm.dims_cvs)))
                             + _out_data_0th*np.pad(_elm.data_2nd, 
                                                    ((2*_out_dims_cvs, 0), (2*_out_dims_cvs, 0)))
                             + 2.*np.einsum("j,k->jk",
                                            np.pad(_out_data_1st, 
                                                   (0, 2*_elm.dims_cvs)),
                                            np.pad(_elm.data_1st, 
                                                   (2*_out_dims_cvs, 0)))
                            )
            _out_data_1st = (_elm.data_0th*np.pad(_out_data_1st, 
                                                  (0, 2*_elm.dims_cvs))
                             + _out_data_0th*np.pad(_elm.data_1st, 
                                                    (2*_out_dims_cvs, 0))
                            )
            _out_data_0th = _out_data_0th*_elm.data_0th
            _out_dims_cvs += _elm.dims_cvs

        elif _out_isfls and not _elm.isfls:
            _out_data_2nd = (_elm.data_0th*np.pad(_out_data_2nd, 
                                                  ((0, 0), (0, 0), (0, 2*_elm.dims_cvs), (0, 2*_elm.dims_cvs)))
                             + np.einsum("jk,lm->jklm",
                                         _out_data_0th,
                                         np.pad(_elm.data_2nd, 
                                                ((2*_out_dims_cvs, 0), (2*_out_dims_cvs, 0)))
                                        )
                             + 2.*np.einsum("jkl,m->jklm",
                                            np.pad(_out_data_1st, 
                                                   ((0, 0),(0, 0),(0, 2*_elm.dims_cvs))),
                                            np.pad(_elm.data_1st, 
                                                   (2*_out_dims_cvs, 0))
                                           )
                            )
            _out_data_1st = (_elm.data_0th*np.pad(_out_data_1st, 
                                                  ((0, 0), (0, 0), (0, 2*_elm.dims_cvs)))
                             + np.einsum("jk,l->jkl",
                                         _out_data_0th,
                                         np.pad(_elm.data_1st, 
                                                (2*_out_dims_cvs, 0))
                                        )
                            )
            _out_data_0th = _out_data_0th*_elm.data_0th
            _out_dims_cvs += _elm.dims_cvs
        
        elif not _out_isfls and _elm.isfls:
            _out_data_2nd = (np.einsum("lm,jk->jklm",
                                       np.pad(_out_data_2nd, 
                                              ((0, 2*_elm.dims_cvs), (0, 2*_elm.dims_cvs))),
                                       _elm.data_0th
                                      )
                             + _out_data_0th*np.pad(_elm.data_2nd, 
                                                    ((0, 0), (0, 0), (2*_out_dims_cvs, 0), (2*_out_dims_cvs, 0)))
                             + 2.*np.einsum("l,jkm->jklm",
                                            np.pad(_out_data_1st, 
                                                   (0, 2*_elm.dims_cvs)),
                                            np.pad(_elm.data_1st, 
                                                   ((0, 0), (0, 0), (2*_out_dims_cvs, 0)))
                                           )
                            )
            _out_data_1st = (np.einsum("l,jk->jkl",
                                       np.pad(_out_data_1st, 
                                              (0, 2*_elm.dims_cvs)),
                                       _elm.data_0th
                                      )
                            + _out_data_0th*np.pad(_elm.data_1st, 
                                                   ((0, 0), (0, 0), (2*_out_dims_cvs, 0)))
                            )
            _out_data_0th = _out_data_0th*_elm.data_0th
            _out_dims_cvs += _elm.dims_cvs
            _out_dims_fls = _elm.dims_fls
            
        else:
            _out_data_2nd = (np.einsum("jpkqyz,plqm->jlkmyz",
                                       np.pad(_out_data_2nd, 
                                              ((0, 0), (0, 0), (0, 2*_elm.dims_cvs), (0, 2*_elm.dims_cvs))
                                             )[:,np.newaxis,:,np.newaxis,:,:],
                                       _elm.data_0th[np.newaxis,:,np.newaxis,:]
                                      ).reshape(np.prod((_out_dims_fls[0], _elm.dims_fls[0])),
                                                np.prod((_out_dims_fls[1], _elm.dims_fls[1])),
                                                2*(_out_dims_cvs + _elm.dims_cvs),
                                                2*(_out_dims_cvs + _elm.dims_cvs))
                             + np.einsum("jpkq,plqmyz->jlkmyz",
                                         _out_data_0th[:,np.newaxis,:,np.newaxis],
                                         np.pad(_elm.data_2nd, 
                                                ((0, 0), (0, 0), (2*_out_dims_cvs, 0), (2*_out_dims_cvs, 0))
                                               )[np.newaxis,:,np.newaxis,:,:,:]
                                        ).reshape(np.prod((_out_dims_fls[0], _elm.dims_fls[0])),
                                                  np.prod((_out_dims_fls[1], _elm.dims_fls[1])),
                                                  2*(_out_dims_cvs + _elm.dims_cvs),
                                                  2*(_out_dims_cvs + _elm.dims_cvs))
                             + 2.*np.einsum("jpkqy,plqmz->jlkmyz",
                                            np.pad(_out_data_1st, 
                                                   ((0, 0),(0, 0),(0, 2*_elm.dims_cvs))
                                                  )[:,np.newaxis,:,np.newaxis,:],
                                            np.pad(_elm.data_1st, 
                                                   ((0, 0),(0, 0),(2*_out_dims_cvs, 0))
                                                  )[np.newaxis,:,np.newaxis,:,:]
                                           ).reshape(np.prod((_out_dims_fls[0], _elm.dims_fls[0])),
                                                     np.prod((_out_dims_fls[1], _elm.dims_fls[1])),
                                                     2*(_out_dims_cvs + _elm.dims_cvs),
                                                     2*(_out_dims_cvs + _elm.dims_cvs))
                            )
            _out_data_1st = (np.einsum("jpkqz,plqm->jlkmz",
                                       np.pad(_out_data_1st, 
                                              ((0, 0), (0, 0), (0, 2*_elm.dims_cvs))
                                             )[:,np.newaxis,:,np.newaxis,:],
                                       _elm.data_0th[np.newaxis,:,np.newaxis,:]
                                      ).reshape(np.prod((_out_dims_fls[0], _elm.dims_fls[0])),
                                                np.prod((_out_dims_fls[1], _elm.dims_fls[1])),
                                                2*(_out_dims_cvs + _elm.dims_cvs))
                             + np.einsum("jpkq,plqmz->jlkmz",
                                         _out_data_0th[:,np.newaxis,:,np.newaxis],
                                         np.pad(_elm.data_1st, 
                                                ((0, 0), (0, 0), (2*_out_dims_cvs, 0))
                                               )[np.newaxis,:,np.newaxis,:,:]
                                        ).reshape(np.prod((_out_dims_fls[0], _elm.dims_fls[0])),
                                                  np.prod((_out_dims_fls[1], _elm.dims_fls[1])),
                                                  2*(_out_dims_cvs + _elm.dims_cvs))
                            )
            _out_data_0th = np.einsum("jpkq,plqm->jlkm",
                                      _out_data_0th[:,np.newaxis,:,np.newaxis],
                                      _elm.data_0th[np.newaxis,:,np.newaxis,:]
                                      ).reshape(np.prod((_out_dims_fls[0], _elm.dims_fls[0])),
                                                np.prod((_out_dims_fls[1], _elm.dims_fls[1])))
            _out_dims_cvs += _elm.dims_cvs
            _out_dims_fls = [_out_dims_fls[0] + _elm.dims_fls[0],
                             _out_dims_fls[1] + _elm.dims_fls[1]]
            
        # Update whether output has an FLS component
        if _out_dims_fls != [[],[]]:
            _out_isfls = True
     
    return QGoper(data_2nd = _out_data_2nd,
                  data_1st = _out_data_1st,
                  data_0th = _out_data_0th,
                  dims_cvs = _out_dims_cvs,
                  dims_fls = _out_dims_fls
                 )


def _tensor_state(args):
    # Check and raise error if an empty QGstate, one that is not iscv or isfls, is in args
    arg_cvs = np.where(tuple(map(lambda x: not x.iscvs, args)))[0]
    arg_fls = np.where(tuple(map(lambda x: not x.isfls, args)))[0]

    if len(set(arg_cvs) & set(arg_fls)) != 0:
        raise ValueError("Tensor product cannot be performed with empty QGstates")
    
    # Set data from first argument as initial data for output
    # Data from args will be used to update these values, with the QGstate
    # constructor called only after interating through all of the args
    _out_data_2nd = args[0].data_2nd
    _out_data_1st = args[0].data_1st
    _out_data_0th = args[0].data_0th
    
    _out_dims_cvs = args[0].dims_cvs
    _out_dims_fls = args[0].dims_fls
    _out_isfls = args[0].isfls

    for _elm in args[1:]:
        if not _out_isfls and not _elm.isfls:
            _out_data_2nd = (np.pad(_out_data_2nd, 
                                    ((0, 2*_elm.dims_cvs), (0, 2*_elm.dims_cvs)))
                             + np.pad(_elm.data_2nd, 
                                      ((2*_out_dims_cvs, 0), (2*_out_dims_cvs, 0)))
                            )
            _out_data_1st = (np.pad(_out_data_1st, 
                                    (0, 2*_elm.dims_cvs))
                             + np.pad(_elm.data_1st, 
                                      (2*_out_dims_cvs, 0))
                            )
            _out_data_0th = _out_data_0th*_elm.data_0th
            _out_dims_cvs += _elm.dims_cvs

        elif _out_isfls and not _elm.isfls:
            _out_data_2nd = (np.multiply(np.where(_elm.data_0th!=0,1,0),
                                         np.pad(_out_data_2nd,
                                                ((0, 0), (0, 0), (0, 2*_elm.dims_cvs), (0, 2*_elm.dims_cvs))) 
                                        )
                             + np.einsum("jk,lm->jklm",
                                         np.where(_out_data_0th!=0,1,0),
                                         np.pad(_elm.data_2nd, 
                                                ((2*_out_dims_cvs, 0), (2*_out_dims_cvs, 0)))
                                        )
                            )
            _out_data_1st = (np.multiply(np.where(_elm.data_0th!=0,1,0),
                                         np.pad(_out_data_1st,
                                                ((0, 0), (0, 0), (0, 2*_elm.dims_cvs)))
                                        )
                             + np.einsum("jk,l->jkl",
                                         np.where(_out_data_0th!=0,1,0),
                                         np.pad(_elm.data_1st, 
                                                (2*_out_dims_cvs, 0))
                                        )
                            )
            _out_data_0th = _out_data_0th*_elm.data_0th
            _out_dims_cvs += _elm.dims_cvs
        
        elif not _out_isfls and _elm.isfls:
            _out_data_2nd = (np.einsum("lm,jk->jklm",
                                       np.pad(_out_data_2nd, 
                                              ((0, 2*_elm.dims_cvs), (0, 2*_elm.dims_cvs))),
                                       np.where(_elm.data_0th!=0,1,0)
                                      )
                             + np.multiply(np.where(_out_data_0th!=0,1,0),
                                           np.pad(_elm.data_2nd, 
                                                  ((0, 0), (0, 0), (2*_out_dims_cvs, 0), (2*_out_dims_cvs, 0)))
                                           )
                            )
            _out_data_1st = (np.einsum("l,jk->jkl",
                                       np.pad(_out_data_1st, 
                                              (0, 2*_elm.dims_cvs)),
                                       np.where(_elm.data_0th!=0,1,0)
                                      )
                            + np.multiply(np.where(_out_data_0th!=0,1,0),
                                          np.pad(_elm.data_1st,
                                                 ((0, 0), (0, 0), (2*_out_dims_cvs, 0)))
                                          )
                            )
            _out_data_0th = _out_data_0th*_elm.data_0th
            _out_dims_cvs += _elm.dims_cvs
            _out_dims_fls = _elm.dims_fls
            
        else:
            _out_data_2nd = (np.einsum("jpkqyz,plqm->jlkmyz",
                                       np.pad(_out_data_2nd,
                                              ((0, 0), (0, 0), (0, 2*_elm.dims_cvs), (0, 2*_elm.dims_cvs))
                                             )[:,np.newaxis,:,np.newaxis,:,:],
                                       np.where(_elm.data_0th!=0,1,0)[np.newaxis,:,np.newaxis,:]
                                      ).reshape(np.prod((_out_dims_fls[0], _elm.dims_fls[0])),
                                                np.prod((_out_dims_fls[1], _elm.dims_fls[1])),
                                                2*(_out_dims_cvs + _elm.dims_cvs),
                                                2*(_out_dims_cvs + _elm.dims_cvs))
                             + np.einsum("jpkq,plqmyz->jlkmyz",
                                         np.where(_out_data_0th!=0,1,0)[:,np.newaxis,:,np.newaxis],
                                         np.pad(_elm.data_2nd,
                                                ((0, 0), (0, 0), (2*_out_dims_cvs, 0), (2*_out_dims_cvs, 0))
                                               )[np.newaxis,:,np.newaxis,:,:,:]
                                        ).reshape(np.prod((_out_dims_fls[0], _elm.dims_fls[0])),
                                                  np.prod((_out_dims_fls[1], _elm.dims_fls[1])),
                                                  2*(_out_dims_cvs + _elm.dims_cvs),
                                                  2*(_out_dims_cvs + _elm.dims_cvs))
                            )
            _out_data_1st = (np.einsum("jpkqz,plqm->jlkmz",
                                       np.pad(_out_data_1st,
                                              ((0, 0), (0, 0), (0, 2*_elm.dims_cvs))
                                             )[:,np.newaxis,:,np.newaxis,:],
                                       np.where(_elm.data_0th!=0,1,0)[np.newaxis,:,np.newaxis,:]
                                      ).reshape(np.prod((_out_dims_fls[0], _elm.dims_fls[0])),
                                                np.prod((_out_dims_fls[1], _elm.dims_fls[1])),
                                                2*(_out_dims_cvs + _elm.dims_cvs))
                             + np.einsum("jpkq,plqmz->jlkmz",
                                         np.where(_out_data_0th!=0,1,0)[:,np.newaxis,:,np.newaxis],
                                         np.pad(_elm.data_1st,
                                                ((0, 0), (0, 0), (2*_out_dims_cvs, 0))
                                               )[np.newaxis,:,np.newaxis,:,:]
                                        ).reshape(np.prod((_out_dims_fls[0], _elm.dims_fls[0])),
                                                  np.prod((_out_dims_fls[1], _elm.dims_fls[1])),
                                                  2*(_out_dims_cvs + _elm.dims_cvs))
                            )
            _out_data_0th = np.einsum("jpkq,plqm->jlkm",
                                      _out_data_0th[:,np.newaxis,:,np.newaxis],
                                      _elm.data_0th[np.newaxis,:,np.newaxis,:]
                                      ).reshape(np.prod((_out_dims_fls[0], _elm.dims_fls[0])),
                                                np.prod((_out_dims_fls[1], _elm.dims_fls[1])))
            _out_dims_cvs += _elm.dims_cvs
            _out_dims_fls = [_out_dims_fls[0] + _elm.dims_fls[0],
                             _out_dims_fls[1] + _elm.dims_fls[1]]
            
        # Update whether output has an FLS component
        if _out_dims_fls != [[],[]]:
            _out_isfls = True
     
    return QGstate(data_2nd = _out_data_2nd,
                   data_1st = _out_data_1st,
                   data_0th = _out_data_0th,
                   dims_cvs = _out_dims_cvs,
                   dims_fls = _out_dims_fls
                   )