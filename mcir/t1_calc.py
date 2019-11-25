import numpy as np
import scipy.optimize as sopt
import multiprocessing as mp
import itertools
import sys


class T1Calc:
    def __init__(
        self,
        n_components=10,
        max_t1=3000,
        min_t1=200,
        max_m0=6000,
        min_m0=0,
        outlier_range=10,
        parallel_processing=False,
        processes=4,
        iterative_optimization=False,
        init_t1=None,
        init_m0="dependent",
        optimization_range=None,
        t1_lower_bounds=None,
        t1_upper_bounds=None,
        m0_lower_bounds="dependent",
        m0_upper_bounds="dependent",
        gm_range=None,
        n_gm_components=None,
    ):

        self.n_components = n_components
        self.max_t1 = max_t1
        self.min_t1 = min_t1
        self.max_m0 = max_m0
        self.min_m0 = min_m0
        self.outlier_range = outlier_range
        self.parallel_processing = parallel_processing
        self.processes = processes
        self.iterative_optimization = iterative_optimization
        self.init_t1 = init_t1
        self.init_m0 = init_m0
        self.optimization_range = optimization_range
        self.t1_lower_bounds = t1_lower_bounds
        self.t1_upper_bounds = t1_upper_bounds
        self.m0_lower_bounds = m0_lower_bounds
        self.m0_upper_bounds = m0_upper_bounds
        self.gm_range = gm_range
        self.n_gm_components = n_gm_components

    def prep_ir_data(self, ir_data):
        return ir_data.astype(float)

    def prep_ti_list(self, ti_list):
        return ti_list.astype(float).ravel()

    def prep_mask(self, mask):
        return mask > 0

    def get_dims(self, ir_data):
        return ir_data.shape[0:3], ir_data.shape[-1]

    def calc_t1(self, ir_data, ti_list, mask=None):
        # flatten IR matrix to allow for multiprocessing:
        ir_data = self.prep_ir_data(ir_data)
        ti_list = self.prep_ti_list(ti_list)
        mask = self.prep_mask(ir_data) if mask is None else self.prep_mask(mask)
        im_dims, ir_dim = self.get_dims(ir_data)
        flat_ir_data = ir_data.reshape(-1, ir_dim)
        flat_mask = mask.reshape(-1)  # flatten binary mask

        optimization_params = self.get_optimization_params(self.n_components, self.iterative_optimization, self.max_t1,
                                                           self.min_t1, self.max_m0, self.min_m0, self.init_t1,
                                                           self.init_m0, self.optimization_range, self.t1_lower_bounds,
                                                           self.t1_upper_bounds, self.m0_lower_bounds,
                                                           self.m0_upper_bounds, self.gm_range, self.n_gm_components)
        self.optimization_results, self.t1_matrix, self.m0_matrix, self.norm_m0_matrix = self.analyze_t1(
            ir_data, ti_list, mask, im_dims, self.n_components, ir_dim, optimization_params,
            self.iterative_optimization, self.parallel_processing, self.processes,
        )

    def get_optimization_params(
            self,
            n_components,
            iterative_optimization,
            max_t1,
            min_t1,
            max_m0,
            min_m0,
            init_t1,
            init_m0,
            optimization_range,
            t1_lower_bounds,
            t1_upper_bounds,
            m0_lower_bounds,
            m0_upper_bounds,
            gm_range,
            n_gm_components,
    ):
        if iterative_optimization:
            optimization_params = self.get_iterative_params(max_t1, min_t1, max_m0, min_m0)
        else:
            optimization_params = self.get_noniterative_params(n_components, max_t1, min_t1, max_m0, min_m0,
                                                            init_t1, init_m0, optimization_range, t1_lower_bounds,
                                                            t1_upper_bounds, m0_lower_bounds, m0_upper_bounds, gm_range,
                                                            n_gm_components)
        return optimization_params

    def get_iterative_params(
            self,
            max_t1,
            min_t1,
            max_m0,
            min_m0,
    ):
        return {'max_t1': max_t1, 'min_t1': min_t1, 'max_m0': max_m0, 'min_m0': min_m0}

    def get_noniterative_params(
            self,
            n_components,
            max_t1,
            min_t1,
            max_m0,
            min_m0,
            init_t1,
            init_m0,
            optimization_range,
            t1_lower_bounds,
            t1_upper_bounds,
            m0_lower_bounds,
            m0_upper_bounds,
            gm_range,
            n_gm_components,
    ):
        t1_lower_bounds = t1_lower_bounds if t1_lower_bounds is not None else self.set_bounds(n_components, min_t1)
        t1_upper_bounds = t1_upper_bounds if t1_upper_bounds is not None else self.set_bounds(n_components, max_t1)
        m0_lower_bounds = m0_lower_bounds if m0_lower_bounds is not None else self.set_bounds(n_components, min_m0)
        m0_upper_bounds = m0_upper_bounds if m0_upper_bounds is not None else self.set_bounds(n_components, max_m0)
        init_t1 = init_t1 if init_t1 is not None else ((t1_lower_bounds + t1_upper_bounds)/2)
        try:
            init_m0 = init_m0 if init_m0 is not None else ((m0_lower_bounds + m0_upper_bounds)/2)
        except:
            init_m0 = "dependent"
        return {"init_t1": init_t1, "init_m0": init_m0, "t1_lower_bounds": t1_lower_bounds,
                "t1_upper_bounds": t1_upper_bounds, "m0_lower_bounds": m0_lower_bounds,
                "m0_upper_bounds": m0_upper_bounds}

    def set_bounds(self, n_components, value):
        return np.tile(value, n_components)

    def analyze_t1(
            self,
            ir_data,
            ti_list,
            mask,
            im_dims,
            n_components,
            ir_dim,
            optimization_params,
            iterative_optimization=False,
            parallel_processing=False,
            processes=4,
    ):
        optimization_results = self.get_optimization_results(
            ir_data, ti_list, mask, im_dims, n_components, ir_dim, optimization_params, iterative_optimization,
            parallel_processing, processes,
        )
        t1_matrix = optimization_results[:, :, :, 0:n_components]
        m0_matrix = optimization_results[:, :, :, n_components:]
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_m0_matrix = np.nan_to_num(m0_matrix / m0_matrix.sum(axis=3, keepdims=True))
        return optimization_results, t1_matrix, m0_matrix, norm_m0_matrix

    def get_optimization_results(
            self,
            ir_data,
            ti_list,
            mask,
            im_dims,
            n_components,
            ir_dim,
            optimization_params,
            iterative_optimization=False,
            parallel_processing=False,
            processes=4,
        ):
        if parallel_processing:
            optimization_results = self.parallel_optimize_t1(
                ir_data, ti_list, mask, im_dims, n_components, ir_dim, optimization_params, iterative_optimization,
                processes,
            )
        else:
            optimization_results = self.optimize_t1(
                ir_data, ti_list, mask, im_dims, n_components, ir_dim, optimization_params, iterative_optimization
            )
        return optimization_results

    def optimize_t1(
            self,
            ir_data,
            ti_list,
            mask,
            im_dims,
            n_components,
            ir_dim,
            optimization_params,
            iterative_optimization=False,
    ):
        if iterative_optimization:
            optimization_results = np.zeros((im_dims[0], im_dims[1], im_dims[2], 2 * n_components))
            pass
        else:
            optimization_results = self.optimize_t1_noniterative(
                ir_data, ti_list, mask, im_dims, n_components, ir_dim, optimization_params
            )

        return optimization_results

    def parallel_optimize_t1(
            self,
            ir_data,
            ti_list,
            mask,
            im_dims,
            n_components,
            ir_dim,
            optimization_params,
            iterative_optimization=False,
            processes=4
    ):
        if iterative_optimization:
            optimization_results = np.zeros((im_dims[0], im_dims[1], im_dims[2], 2 * n_components))
            pass
        else:
            optimization_results = self.parallel_optimize_t1_noniterative(
                ir_data, ti_list, mask, im_dims, n_components, ir_dim, optimization_params
            )

        return optimization_results

    def parallel_optimize_t1_noniterative(self, ir_data, ti_list, mask, im_dims, n_components, ir_dim, optimization_params):
        ir_data = ir_data.reshape(-1, ir_dim)
        mask = mask.reshape(-1, 1)

        if sys.platform == "win32":
            # mp.spawn.set_executable(_winapi.GetModuleFileName(0))
            import _winapi
            mp.set_executable(_winapi.GetModuleFileName(0))
        pool = mp.Pool(processes=self.processes)  # initialize multiprocess
        # run multiprocess and convert to numpy array
        print("Starting optimization")
        results = np.array(
            pool.starmap(
                self.single_voxel_parallel_optimize,
                [
                    (ir_data[i, :], ti_list, mask[i, :], n_components, optimization_params)
                    for i in range(mask.shape[0])
                ],
            )
        )
        pool.close()
        pool.terminate()
        return results.reshape((im_dims[0], im_dims[1], im_dims[2], 2 * n_components))

    def optimize_t1_noniterative(self, ir_data, ti_list, mask, im_dims, n_components, ir_dim, optimization_params):
        optimization_results = np.zeros((im_dims[0], im_dims[1], im_dims[2], 2*n_components))
        for i, j, k in itertools.product(range(im_dims[0]), range(im_dims[1]), range(im_dims[2])):
            if mask[i, j, k]:
                try:
                    s = self.opt_mcir_fun(ir_data[i, j, k], ti_list, n_components,
                                          optimization_params["init_t1"], optimization_params["init_m0"],
                                          optimization_params["t1_lower_bounds"], optimization_params["t1_upper_bounds"],
                                          optimization_params["m0_lower_bounds"], optimization_params["m0_upper_bounds"])
                    optimization_results[i, j, k, :] = s.x
                except:
                    pass
        return optimization_results

    def single_voxel_parallel_optimize(self, data, ti_list, mask, n_components, optimization_params):
        if mask:
            try:
                s = self.opt_mcir_fun(data, ti_list, n_components, optimization_params["init_t1"],
                                      optimization_params["init_m0"], optimization_params["t1_lower_bounds"],
                                      optimization_params["t1_upper_bounds"], optimization_params["m0_lower_bounds"],
                                      optimization_params["m0_upper_bounds"])
                return s.x
            except:
                return np.zeros(2 * n_components)
        else:
            return np.zeros(2 * n_components)


    def mcir_fun(self, x, data, TI):
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
        m0 = x[K : (2 * K)].reshape(K, 1)  # get m0 variables
        M = (m0 * (1 - 2 * np.exp(-TI / t1))).sum(0)  # calculate IR function
        """return (
            np.absolute(M.__abs__() - data)
        ).sum()  # compute residuals and return them"""
        return M.__abs__() - data  # compute residuals and return them

    def mcir_jac(self, x, data, TI):
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
        m0 = x[K : (2 * K)].reshape(K, 1)  # get m0 variables
        decay = np.exp(-TI / t1)
        M = (m0 * (1 - 2 * decay)).sum(0)  # calculate IR function
        dM0j = (1 - 2 * decay).T
        dT1j = (-2 * m0 * TI * decay / (t1 ** 2)).T
        return np.sign(M)[:, np.newaxis] * np.concatenate((dT1j, dM0j), axis=1)  # compute jacobian and return them

    def opt_mcir_fun(self, ir_data, ti_list, n_components, init_t1, init_m0, lb_t1, ub_t1, lb_m0, ub_m0):
        # initialize starting point and lower and upper bounds
        '''x0 = np.concatenate(
            (np.linspace(800, 2000, nc), np.tile(np.absolute(data).max() / nc, nc))
        )
        lb = np.concatenate((np.tile(self.min_t1, nc), np.tile(self.min_m0, nc)))
        ub = np.concatenate((np.tile(self.max_t1, nc), np.tile(min(self.max_m0, 1.5*data.max()), nc)))'''
        max_ir = ir_data.max()
        init_m0 = np.tile(max_ir/n_components, n_components) if init_m0 == "dependent" else init_m0
        lb_m0 = np.zeros(n_components) if lb_m0 == "dependent" else lb_m0
        ub_m0 = np.tile(max_ir, n_components) if ub_m0 == "dependent" else ub_m0

        x0 = np.concatenate((init_t1, init_m0))
        lb = np.concatenate((lb_t1, lb_m0))
        ub = np.concatenate((ub_t1, ub_m0))

        x0[(lb > x0) | (ub < x0)] = (
            lb[(lb > x0) | (ub < x0)] + ub[(lb > x0) | (ub < x0)]
        ) / 2
        # return optimization struct
        """return sopt.minimize(
            self.mcir_fun,
            x0,
            method="L-BFGS-B",
            args=(data, ti),
            bounds=sopt.Bounds(lb, ub),
        )"""
        return sopt.least_squares(
            self.mcir_fun,
            x0,
            jac=self.mcir_jac,
            bounds=(lb, ub),
            method="trf",
            args=(ir_data, ti_list),
            loss="linear",
            xtol=0.1,
            ftol=0.1,
            gtol=0.1,
            diff_step=0.01
        )
