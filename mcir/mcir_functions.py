import numpy as np
import scipy.optimize as sopt

def mcir_fun(x, data, TI):
    """
    Calculates the residual for the multi-component IR function with K components. The IR function can either be:
    (1) M[j] = Sum(on i) of m0[i] * (1-2*exp(-TI[j]/t1[i])
    where M[j] is the signal at TI[j], t1[i] is the t1 of component i, and m0[i] is a measure of component's proton density
        or
    (2) M[j] = m0 * (Sum(on i) of f[i] * (1-2*exp(-TI[j]/t1[i]) )
    where M[j] is the signal at TI[j], m0 is the voxel's proton density, t1[i] is the t1 of component i, and f[i] is the component's volume fraction of the voxel
    :param x: array, containing the independent variables. For an K-component IR fit, len(x) must equal to 2*K (for IR function 1) or 2*K+1 (for IR function 2).
                function 1: x[0:K] - t1 variables; x[K:2*K] - m0 variables
                function 2: x[0:K] - t1 variables; x[K:2*K] - f variables; x[-1] - m0 variable
    :param data: array, containing the measured IR signal. N=len(data) is the number of measured TI points
    :param TI: array of length N, specifying the TI point at which the data was measured
    :return: array of length N with the residuals of the IR function
    """
    K = int(x.size // 2)  # number of components to be fitted
    t1 = x[0:K].reshape(K, 1)  # get t1 variables
    m0 = x[K: (2 * K)].reshape(K, 1)  # get m0 variables
    M = (m0 * (1 - 2 * np.exp(-TI / t1))).sum(0)  # calculate IR function
    return M.__abs__() - data  # compute residuals and return them


def mcir_jac(x, data, TI):
    """
    Calculates the jacobian for the multi-component IR function with K components. The IR function can either be:
    (1) M[j] = Sum(on i) of m0[i] * (1-2*exp(-TI[j]/t1[i])
    where M[j] is the signal at TI[j], t1[i] is the t1 of component i, and m0[i] is a measure of component's proton density
        or
    (2) M[j] = m0 * (Sum(on i) of f[i] * (1-2*exp(-TI[j]/t1[i]) )
    where M[j] is the signal at TI[j], m0 is the voxel's proton density, t1[i] is the t1 of component i, and f[i] is the component's volume fraction of the voxel
    :param x: array, containing the independent variables. For an K-component IR fit, len(x) must equal to 2*K (for IR function 1) or 2*K+1 (for IR function 2).
                function 1: x[0:K] - t1 variables; x[K:2*K] - m0 variables
                function 2: x[0:K] - t1 variables; x[K:2*K] - f variables; x[-1] - m0 variable
    :param data: array, containing the measured IR signal. N=len(data) is the number of measured TI points
    :param TI: array of length N, specifying the TI point at which the data was measured
    :return: array of length N with the residuals of the IR function
    """
    K = int(x.size // 2)  # number of components to be fitted
    t1 = x[0:K].reshape(K, 1)  # get t1 variables
    m0 = x[K: (2 * K)].reshape(K, 1)  # get m0 variables
    decay = np.exp(-TI / t1)
    M = (m0 * (1 - 2 * decay)).sum(0)  # calculate IR function
    dM0j = (1 - 2 * decay).T
    dT1j = (-2 * m0 * TI * decay / (t1 ** 2)).T
    return np.sign(M)[:, np.newaxis] * np.concatenate((dT1j, dM0j), axis=1)  # compute jacobian and return them
