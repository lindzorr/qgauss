import numpy.typing as npt
import qgauss
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from ..core.qgstate import QGstate
from ..dev.qghle import QGhle
from ..calc.utilities import *

__all__ = ['snr']

def snr(HLE0: QGhle,
        rho0: QGstate,
        tlist: list[float] | npt.NDArray[float],
        pointers: tuple[int,int],
        meas_mode: int | list[int],
        noise_rest: float = 0,
        **options
        ):
    """
    ---- Procedure ----
    Test function for the SNR of Wholly time-independent systems. No default
    values accepted, as this function purely exists to test the accuracy of
    calculting the SNR by numerically solving a set of the differential 
    equations. 

    ---- Parameters ----
    HLE0 : QGhle
    rho0 : QGstate
    tlist : list(float)
    pointers : string
    meas_mode : int or list(int)

    ---- Returns ----
    snr : list[float] or list[array[float]]
    meas_signal : list[float] or list[array[float]]
    meas_noise : list[float] or list[array[float]]
    """
    _defaults = {'atol': qgauss.settings.atol,
                 'rtol': qgauss.settings.rtol, 
                 'method': 'RK45'}
    options = {**_defaults, **options}
    _tol = options['atol']

    # Generate dynamical matrices for each pointer state by generating the
    # corresponding Lindbladian and extracting the element from there
    _dims_sys = HLE0.dims_cvs
    _dims_env = HLE0.dims_env
    _id_sys = np.identity(2*_dims_sys)
    _id_env = np.identity(2*_dims_env)
    _lme_A = HLE0[pointers[0], pointers[0]].lme
    _lme_B = HLE0[pointers[1], pointers[1]].lme
    _dyn_mat_A = _lme_A.wigner_2nd_rdr
    _dyn_mat_B = _lme_B.wigner_2nd_rdr
    _drv_vec_A = _lme_A.wigner_1st_dr
    _drv_vec_B = _lme_B.wigner_1st_dr
    
    # Initialize array of mode indices to be monitored
    if isinstance(meas_mode, int): 
        meas_mode = [meas_mode]
    _mode_index = np.zeros(2*HLE0.dims_env)
    for i in meas_mode: _mode_index[2*i-2:2*i] = 1

    # Define some commonly used arrays to make future expressions more compact
    _h_se_A = HLE0[pointers[0], pointers[0]].h_se
    _h_se_B = HLE0[pointers[1], pointers[1]].h_se
    sfs = HLE0.symform_sys
    sfb = HLE0.symform_env

    _J_A = sfb @ _h_se_A.T
    _J_B = sfb @ _h_se_B.T
    _K_A = sfs @ _h_se_A
    _K_B = sfs @ _h_se_B

    input_cov = HLE0.input_env.data_2nd
    input_mean = HLE0.input_env.data_1st
    intra_cov = rho0.data_2nd
    intra_mean = rho0.data_1st

    def weights(t):
        return ((-_J_A @ (expm((t-tlist[0])*_dyn_mat_A) @ intra_mean 
                          + t*expm(-tlist[0]*_dyn_mat_A) @ expm_int(t*_dyn_mat_A) @ _drv_vec_A)
                 +_J_B @ (expm((t-tlist[0])*_dyn_mat_B) @ intra_mean
                          + t*expm(-tlist[0]*_dyn_mat_B) @ expm_int(t*_dyn_mat_B) @ _drv_vec_B)
                ) * _mode_index).real

    def de_meas_signal(t, X, Hsb, dyn_mat, drv_vec):
        return weights(t) @ (input_mean 
                             - sfb @ Hsb.T @ (expm((t-tlist[0])*dyn_mat) @ intra_mean
                                              + t*expm(-tlist[0]*dyn_mat) @ expm_int(t*dyn_mat) @ drv_vec)
                            )

    _XS_At = solve_ivp(fun = de_meas_signal,
                       t_span = (tlist[0],tlist[-1]),
                       y0 = np.zeros(2*_dims_env),
                       t_eval = tlist,
                       args = (_h_se_A,_dyn_mat_A,_drv_vec_A),
                       **options)
    _XS_Bt = solve_ivp(fun = de_meas_signal,
                       t_span = (tlist[0],tlist[-1]), 
                       y0 = np.zeros(2*_dims_env), 
                       t_eval = tlist,
                       args = (_h_se_B,_dyn_mat_B,_drv_vec_B),
                       **options)
    
    signal_A = np.array([np.real(_XS_At.y[0,t]) for t in range(0,len(tlist))])
    signal_B = np.array([np.real(_XS_Bt.y[0,t]) for t in range(0,len(tlist))])
    
    def de_meas_noise_1(t, X, Hsb, A):
        _wt = weights(t)
        _Vt = expm((t-tlist[0])*A)
        _X1 = X[1:2*_dims_env+1]
        _X2 = X[2*_dims_env+1:4*_dims_env+1]
        _X3 = vec_to_mat(X[4*_dims_env+1:4*_dims_env*(_dims_env+1)+1])
        return np.concatenate(([_wt @ (input_cov + noise_rest*_id_env) @ _wt
                                + _wt @ sfb @ Hsb.T @ (-_Vt @ intra_cov @ _X1 + _X2)
                                - (-_X1.T @ intra_cov @ _Vt.T + _X2.T) @ Hsb @ sfb @ _wt],
                                _Vt.T @ Hsb @ sfb @ _wt,
                                A @ _X2 + (sfs @ Hsb @ input_cov + _X3 @ Hsb @ sfb) @ _wt,
                                mat_to_vec(A @ _X3 + _X3 @ A.T + sfs @ Hsb @ input_cov @ Hsb.T @ sfs)
                               ))
    
    def de_meas_noise_2(t, X, Hsb, A):
        _wt = weights(t)
        _Vt = expm((t-tlist[0])*A)
        _X1 = X[1:2*_dims_env+1]
        _X2 = X[2*_dims_env+1:4*_dims_env+1]
        _X3 = vec_to_mat(X[4*_dims_env+1:4*_dims_env*(_dims_env+1)+1])
        return np.concatenate(([_wt @ (input_cov + noise_rest*_id_env) @ _wt
                                + 2*_wt @ sfb @ Hsb.T @ (-_Vt @ intra_cov @ _X1 + _X2)],
                                _Vt.T @ Hsb @ sfb @ _wt,
                                A @ _X2 + (sfs @ Hsb @ input_cov + _X3 @ Hsb @ sfb) @ _wt,
                                mat_to_vec(A @ _X3 + _X3 @ A.T + sfs @ Hsb @ input_cov @ Hsb.T @ sfs)
                               ))

    _XN_At = solve_ivp(fun = de_meas_noise_1,
                       t_span = (tlist[0],tlist[-1]), 
                       y0 = np.zeros(4*HLE0.dims_env*(HLE0.dims_env+1)+1), 
                       t_eval = tlist,
                       args = (_h_se_A,_dyn_mat_A),
                       **options)
    _XN_Bt = solve_ivp(fun = de_meas_noise_2,
                       t_span = (tlist[0],tlist[-1]), 
                       y0 = np.zeros(4*HLE0.dims_env*(HLE0.dims_env+1)+1), 
                       t_eval = tlist,
                       args = (_h_se_B,_dyn_mat_B),
                       **options)

    noise_A = np.array([np.real(_XN_At.y[0,t]) for t in range(0,len(tlist))])
    noise_B = np.array([np.real(_XN_Bt.y[0,t]) for t in range(0,len(tlist))])
    
    _signal = [np.abs(signal_A[j] - signal_B[j]) for j in range(0,len(tlist))]
    _noise = [(noise_A[j] + noise_B[j]) for j in range(0,len(tlist))]
    snrsq = np.array([0] + [0.25*(_signal[j]**2/_noise[j]) for j in range(1,len(tlist))])

    return snrsq,signal_A,signal_B,noise_A,noise_B

    
"""
    def mean_dyn_1(t):
        return (expm((t-tlist[0])*_dyn_mat_A) @ intra_mean
                + t*expm(-tlist[0]*_dyn_mat_A) @ expm_int(t*_dyn_mat_A) @ _drv_vec_A)
    def mean_dyn_2(t,X):
        return _dyn_mat_A @ X + _drv_vec_A
    mean_test = solve_ivp(fun = mean_dyn_2,
                          t_span = (tlist[0],tlist[-1]),
                          y0 = intra_mean,
                          t_eval = tlist)

    def weights_de(t, X):
        _Xw = X[0:2*_dims]
        _XA = X[2*_dims:4*_dims]
        _XB = X[4*_dims:6*_dims]
        return np.concatenate(((-_J_A @ (_dyn_mat_A @ _XA + _drv_vec_A) 
                                +_J_B @ (_dyn_mat_B @ _XB + _drv_vec_B)) * _mode_index,
                               _dyn_mat_A @ _XA + _drv_vec_A,
                               _dyn_mat_B @ _XB + _drv_vec_B))
    
    _XW0 = np.concatenate(((-_J_A + _J_B) @ intra_mean * _mode_index, intra_mean, intra_mean))

    _XWt = solve_ivp(fun = weights_de,
                     t_span = (tlist[0],tlist[-1]),
                     y0 = _XW0,
                     t_eval = tlist,
                     **options)
    weights_num = [np.real(_XWt.y[0:2*_dims,t]) for t in range(0,len(tlist))]

    def de_meas_signal_2(t, X, J, dyn_mat, drv_vec):
        _Xm = X[1:2*_dims+1]
        return np.concatenate((weights(t) @ (input_mean - J @ _Xm),
                               dyn_mat @ _Xm + drv_vec))
"""