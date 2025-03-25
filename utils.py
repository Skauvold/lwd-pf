import numpy as np


class CovarianceFunction:
    def __init__(self, params: dict):
        self.params = params

    def cov(self, x1, x2):
        raise NotImplementedError

    def cov_matrix(self, xs):
        n = len(xs)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.cov(xs[i], xs[j])
        return K

    def cross_cov(self, x, xs):
        n = len(xs)
        K = np.zeros(n)
        for i in range(n):
            K[i] = self.cov(x, xs[i])
        return K


class GenExp(CovarianceFunction):
    def cov(self, x1, x2):
        return self.params["standard_deviation"] ** 2 * np.exp(
            -np.power(
                np.abs(x1 - x2) / self.params["correlation_range"],
                self.params["exponent"],
            )
        )


class GaussianProcess:
    def __init__(self, cov_func: CovarianceFunction, mean_func: callable, domain: dict):
        self.cov_func = cov_func
        self.xvals = np.linspace(domain["xmin"], domain["xmax"], domain["n"])
        self.K = self.cov_func.cov_matrix(self.xvals)
        self.mean_vec = mean_func(self.xvals)

    def mean(self) -> tuple:
        return self.mean_vec

    def limits(self, n_std=3):
        sig = np.sqrt(np.diag(self.K))
        lower = self.mean_vec - n_std * sig
        upper = self.mean_vec + n_std * sig
        return min(lower), max(upper)

    def draw_full(self):
        return np.random.multivariate_normal(self.mean_vec, self.K)

    def draw_conditional(self, cond_indices: list, cond_vals: list):
        K = self.K
        K11 = K[np.ix_(cond_indices, cond_indices)]
        K12 = K[
            np.ix_(cond_indices, [i for i in range(len(K)) if i not in cond_indices])
        ]
        K21 = K[
            np.ix_([i for i in range(len(K)) if i not in cond_indices], cond_indices)
        ]
        K22 = K[
            np.ix_(
                [i for i in range(len(K)) if i not in cond_indices],
                [i for i in range(len(K)) if i not in cond_indices],
            )
        ]
        mu1 = self.mean_vec[cond_indices]
        mu2 = self.mean_vec[[i for i in range(len(K)) if i not in cond_indices]]
        mu = mu2 + np.dot(K21, np.linalg.solve(K11, np.array(cond_vals) - mu1))
        cov = K22 - np.dot(K21, np.linalg.solve(K11, K12))
        realization = np.random.multivariate_normal(mu, cov)

        # Combine the conditional realization with the known values
        full_realization = np.zeros(len(K))
        full_realization[cond_indices] = cond_vals
        full_realization[[i for i in range(len(K)) if i not in cond_indices]] = (
            realization
        )

        return full_realization

    def draw_left_conditional(self, index: int, cond_vals: list):
        # Draw single element at index conditional on everything to the left
        K = self.K
        K11 = K[:index, :index]
        K12 = K[:index, index]
        K21 = K[index, :index]
        K22 = K[index, index]
        mu1 = self.mean_vec[:index]
        mu2 = self.mean_vec[index]
        mu = mu2 + np.dot(K21, np.linalg.solve(K11, np.array(cond_vals) - mu1))
        cov = K22 - np.dot(K21, np.linalg.solve(K11, K12))
        return np.random.normal(mu, cov)


class TopBaseProcessPair:
    def __init__(self, base_gp: GaussianProcess, thickness_gp: GaussianProcess):
        self.base_gp = base_gp
        self.thickness_gp = thickness_gp
        self.xvals = self.base_gp.xvals

    def draw_full(self):
        base_sample = self.base_gp.draw_full()
        thickness_sample = self.thickness_gp.draw_full()
        top_sample = base_sample - thickness_sample
        return {"top": top_sample, "base": base_sample}

    def mean_top(self):
        return self.base_gp.mean() - self.thickness_gp.mean()

    def mean_base(self):
        return self.base_gp.mean()

    def limits(self, n_std=3):
        top_limits = self.base_gp.limits(n_std=n_std)
        base_limits = self.thickness_gp.limits(n_std=n_std)
        return min(top_limits[0], base_limits[0]), max(top_limits[1], base_limits[1])

    def draw_conditional(self, cond_indices: list, cond_vals: dict):
        base_sample = self.base_gp.draw_conditional(cond_indices, cond_vals["base"])
        cond_vals_thickness = [
            cv_base - cv_top
            for cv_base, cv_top in zip(cond_vals["base"], cond_vals["top"])
        ]
        thickness_sample = self.thickness_gp.draw_conditional(
            cond_indices, cond_vals_thickness
        )
        top_sample = base_sample - thickness_sample
        return {"top": top_sample, "base": base_sample}

    def draw_left_conditional(self, index: int, cond_vals: dict):
        base_sample = self.base_gp.draw_left_conditional(index, cond_vals["base"])
        cond_vals_thickness = [
            cv_base - cv_top
            for cv_base, cv_top in zip(cond_vals["base"], cond_vals["top"])
        ]
        thickness_sample = self.thickness_gp.draw_left_conditional(
            index, cond_vals_thickness
        )
        top_sample = base_sample - thickness_sample
        return {"top": top_sample, "base": base_sample}

    # TODO:
    # - draw conditional value at an x-value given a list of x-values and their z-values
