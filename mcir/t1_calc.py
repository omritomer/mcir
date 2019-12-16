import numpy as np
import scipy.optimize as sopt
import multiprocessing as mp
import itertools
import sys
from mcir_functions import mcir_fun, mcir_jac


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

        optimization_params = self.get_optimization_params(self.n_components, self.max_t1,
                                                           self.min_t1, self.max_m0, self.min_m0, self.init_t1,
                                                           self.init_m0, self.t1_lower_bounds, self.t1_upper_bounds,
                                                           self.m0_lower_bounds, self.m0_upper_bounds)
        self.optimization_results, self.t1_matrix, self.m0_matrix, self.norm_m0_matrix = self.analyze_t1(
            ir_data, ti_list, mask, im_dims, self.n_components, ir_dim, optimization_params,
            self.parallel_processing, self.processes,
        )

    def get_optimization_params(
            self,
            n_components,
            max_t1,
            min_t1,
            max_m0,
            min_m0,
            init_t1,
            init_m0,
            t1_lower_bounds,
            t1_upper_bounds,
            m0_lower_bounds,
            m0_upper_bounds,
    ):
        optimization_params = self.get_params(n_components, max_t1, min_t1, max_m0, min_m0,
                                                            init_t1, init_m0, t1_lower_bounds,
                                                            t1_upper_bounds, m0_lower_bounds, m0_upper_bounds)
        return optimization_params

    def get_params(
            self,
            n_components,
            max_t1,
            min_t1,
            max_m0,
            min_m0,
            init_t1,
            init_m0,
            t1_lower_bounds,
            t1_upper_bounds,
            m0_lower_bounds,
            m0_upper_bounds,
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
            parallel_processing=False,
            processes=4,
    ):
        optimization_results = self.get_optimization_results(
            ir_data, ti_list, mask, im_dims, n_components, ir_dim, optimization_params,
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
            parallel_processing=False,
            processes=4,
        ):
        if parallel_processing:
            optimization_results = self.parallel_optimize_t1(
                ir_data, ti_list, mask, im_dims, n_components, ir_dim, optimization_params,
                processes,
            )
        else:
            optimization_results = self.optimize_t1(
                ir_data, ti_list, mask, im_dims, n_components, ir_dim, optimization_params
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
    ):
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

    def parallel_optimize_t1(
            self,
            ir_data,
            ti_list,
            mask,
            im_dims,
            n_components,
            ir_dim,
            optimization_params,
            processes=4
    ):
        ir_data = ir_data.reshape(-1, ir_dim)
        mask = mask.reshape(-1, 1)

        if sys.platform == "win32":
            # mp.spawn.set_executable(_winapi.GetModuleFileName(0))
            import _winapi
            mp.set_executable(_winapi.GetModuleFileName(0))
        pool = mp.Pool(processes=processes)  # initialize multiprocess
        # run multiprocess and convert to numpy array
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

        return sopt.least_squares(
            mcir_fun,
            x0,
            jac=mcir_jac,
            bounds=(lb, ub),
            method="trf",
            args=(ir_data, ti_list),
            loss="linear",
            xtol=0.1,
            ftol=0.1,
            gtol=0.1,
            diff_step=0.01
        )
