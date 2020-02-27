import numpy as np
import scipy.optimize as sopt
import multiprocessing as mp
import itertools
import sys
from mcir_functions import mcir_fun, mcir_jac


class T1Calc:
    def __init__(
        self,
        n_max=10,
        max_t1=4000,
        min_t1=200,
        max_m0=4000,
        min_m0=0,
        parallel_processing=False,
        processes=4,
    ):

        self.n_max = n_max
        self.max_t1 = max_t1
        self.min_t1 = min_t1
        self.max_m0 = max_m0
        self.min_m0 = min_m0
        self.parallel_processing = parallel_processing
        self.processes = processes

    def prep_ir_data(self, ir_data):
        return ir_data.astype(float)

    def prep_ti_list(self, ti_list):
        return ti_list.astype(float).ravel()

    def prep_mask(self, mask):
        return mask > 0

    def get_dims(self, ir_data):
        return ir_data.shape[0:3], ir_data.shape[-1]

    def get_bounds(self):
        lower_bounds = [np.array([self.min_t1] * n + [self.min_m0] * n)
                        for n in range(2, self.n_max + 1)]
        upper_bounds = [np.array([self.max_t1] * n + [self.max_m0] * n)
                        for n in range(2, self.n_max + 1)]
        return lower_bounds, upper_bounds

    def calc_t1(self, ir_data, ti_list, mask=None):
        ir_data = self.prep_ir_data(ir_data)
        ti_list = self.prep_ti_list(ti_list)
        mask = self.prep_mask(ir_data) if mask is None else self.prep_mask(mask)
        im_dims, ir_dim = self.get_dims(ir_data)
        lower_bounds, upper_bounds = self.get_bounds()

        self.optimization_results, self.t1_matrix, self.m0_matrix, self.norm_m0_matrix = self.analyze_t1(
            ir_data, ti_list, mask, im_dims, self.n_max, ir_dim,
            lower_bounds, upper_bounds, self.parallel_processing, self.processes,
        )

    def analyze_t1(
        self,
        ir_data,
        ti_list,
        mask,
        im_dims,
        n_max,
        ir_dim,
        lower_bounds,
        upper_bounds,
        parallel_processing=False,
        processes=4,
    ):
        optimization_results = self.get_optimization_results(
            ir_data, ti_list, mask, im_dims, n_max, ir_dim,
            lower_bounds, upper_bounds,
            parallel_processing, processes,
        )
        t1_matrix = optimization_results[:, :, :, 0:n_max]
        m0_matrix = optimization_results[:, :, :, n_max:]
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_m0_matrix = np.nan_to_num(m0_matrix / m0_matrix.sum(axis=3, keepdims=True))
        return optimization_results, t1_matrix, m0_matrix, norm_m0_matrix

    def get_optimization_results(
        self,
        ir_data,
        ti_list,
        mask,
        im_dims,
        n_max,
        ir_dim,
        lower_bounds,
        upper_bounds,
        parallel_processing=False,
        processes=4,
    ):
        if parallel_processing:
            optimization_results = self.parallel_optimize_t1(
                ir_data, ti_list, mask, im_dims, n_max, ir_dim,
                lower_bounds, upper_bounds, processes,
            )
        else:
            optimization_results = self.optimize_t1(
                ir_data, ti_list, mask, im_dims, n_max, ir_dim, lower_bounds, upper_bounds,
            )
        return optimization_results

    def optimize_t1(
        self,
        ir_data,
        ti_list,
        mask,
        im_dims,
        n_max,
        ir_dim,
        lower_bounds,
        upper_bounds,
    ):
        optimization_results = np.zeros((im_dims[0], im_dims[1], im_dims[2], 2 * n_max))
        for i, j, k in itertools.product(range(im_dims[0]), range(im_dims[1]), range(im_dims[2])):
            if mask[i, j, k]:
                try:
                    s = self.opt_mcir_fun(ir_data[i, j, k], ti_list, n_max,
                                          lower_bounds, upper_bounds)
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
        n_max,
        ir_dim,
        lower_bounds,
        upper_bounds,
        processes=4,
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
                    (ir_data[i, :], ti_list, mask[i, :], n_max, lower_bounds, upper_bounds)
                    for i in range(mask.shape[0])
                ],
            )
        )
        pool.close()
        pool.terminate()
        return results.reshape((im_dims[0], im_dims[1], im_dims[2], 2 * n_max))

    def single_voxel_parallel_optimize(self, data, ti_list, mask, n_max, lower_bounds, upper_bounds):
        if mask:
            try:
                s = self.opt_mcir_fun(data, ti_list, n_max, lower_bounds, upper_bounds)
                return s.x
            except:
                return np.zeros(2 * n_max)
        else:
            return np.zeros(2 * n_max)

    def opt_mcir_fun(self, data, ti, n_max, lower_bounds, upper_bounds):
        max_ir = data.max()
        init_x0 = [np.array([1200] + list(np.linspace(600, 2500, n - 1, False)) + [max_ir / n] * n)
                   for n in range(2, n_max + 1)]
        cost = np.inf
        for i in range(0, n_max - 1):
            lb = lower_bounds[i]
            ub = upper_bounds[i]
            x0 = init_x0[i]
            res = sopt.least_squares(
                mcir_fun,
                x0,
                jac=mcir_jac,
                bounds=(lb, ub),
                method="trf",
                args=(data, ti),
                loss="soft_l1",
                xtol=1e-4,
                ftol=1e-2,
                diff_step=1e-2,
                gtol=1e-4,
            )
            if res.cost < cost:
                cost = res.cost
                x = res.x
        k = int(x.shape[0] / 2)
        return np.concatenate((x[:k], np.zeros(n_max - k), x[k:], np.zeros(n_max - k)))
