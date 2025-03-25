import numpy as np
import scipy.stats as stats


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

    def cross_cov_one_to_many(self, x, xs):
        n = len(xs)
        K = np.zeros(n)
        for i in range(n):
            K[i] = self.cov(x, xs[i])
        return K

    def cross_cov_many_to_many(self, xs, ys):
        n = len(xs)
        m = len(ys)
        K = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                K[i, j] = self.cov(xs[i], ys[j])
        return K


class GenExp(CovarianceFunction):
    def cov(self, x1: float, x2: float) -> float:
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
        self.mean_func = mean_func

    def mean(self) -> tuple:
        return self.mean_vec

    def limits(self, n_std=3):
        sig = np.sqrt(np.diag(self.K))
        lower = self.mean_vec - n_std * sig
        upper = self.mean_vec + n_std * sig
        return min(lower), max(upper)

    def draw_full(self):
        return np.random.multivariate_normal(self.mean_vec, self.K)

    def draw_single(self, x: float) -> float:
        var = self.cov_func.cov_matrix([x])[0, 0]
        mean = self.mean_func([x])[0]
        return np.random.normal(mean, np.sqrt(var))

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

    # - draw conditional value at an x-value given a list of x-values and their z-values
    def draw_conditional_points(
        self, xvals_query: list, xvals_cond: list, zvals_cond: list
    ):
        K = self.K
        K11 = self.cov_func.cov_matrix(xvals_cond)
        K12 = self.cov_func.cross_cov_many_to_many(xvals_query, xvals_cond)
        K21 = self.cov_func.cross_cov_many_to_many(xvals_cond, xvals_query)
        K22 = self.cov_func.cov_matrix(xvals_query)
        mu1 = self.mean_func(np.array(xvals_cond))
        mu2 = self.mean_func(np.array(xvals_query))
        mu = mu2 + np.matmul(K21.T, np.linalg.solve(K11, np.array(zvals_cond) - mu1))
        cov = K22 - np.matmul(K21.T, np.linalg.solve(K11, K12.T))
        realization = np.random.multivariate_normal(mu, cov)
        return realization


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

    def draw_single(self, x: float) -> dict:
        base_sample = self.base_gp.draw_single(x)
        thickness_sample = self.thickness_gp.draw_single(x)
        top_sample = base_sample - thickness_sample
        return {"top": top_sample, "base": base_sample}

    def mean_top(self):
        return self.base_gp.mean() - self.thickness_gp.mean()

    def mean_base(self):
        return self.base_gp.mean()

    def interpolate_sample(self, xs: list, sample: dict):
        zs_base = sample["base"]
        zs_top = sample["top"]
        zs_thickness = [z_base - z_top for z_base, z_top in zip(zs_base, zs_top)]
        thickness_sample = np.interp(xs, self.xvals, zs_thickness)
        base_sample = np.interp(xs, self.xvals, zs_base)
        top_sample = base_sample - thickness_sample
        return {"top": top_sample, "base": base_sample}

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

    # draw conditional value at an x-value given a list of x-values and their z-values
    def draw_conditional_points(
        self, xvals_query: list, xvals_cond: list, zvals_cond: dict
    ):
        zvals_cond_base = zvals_cond["base"]
        zvals_cond_top = zvals_cond["top"]
        thickness_cond_as_list = [
            cv_base - cv_top for cv_base, cv_top in zip(zvals_cond_base, zvals_cond_top)
        ]
        base_cond_as_list = [cv_base for cv_base in zvals_cond_base]

        base_sample = self.base_gp.draw_conditional_points(
            xvals_query, xvals_cond, base_cond_as_list
        )
        thickness_sample = self.thickness_gp.draw_conditional_points(
            xvals_query, xvals_cond, thickness_cond_as_list
        )

        top_sample = base_sample - thickness_sample
        return {"top": top_sample, "base": base_sample}


class ZonalField:
    def __init__(self, trend_zrel: list, trend_vals: list):
        self.trend_zrel = trend_zrel
        self.trend_vals = trend_vals

    def eval(self, z, noise_params: dict = None):
        signal = np.interp(z, self.trend_zrel, self.trend_vals)
        if noise_params is not None:
            noise = np.random.normal(noise_params["mean"], noise_params["sd"], len(z))
            return signal + noise
        else:
            return signal

    def triplet_trend_prediction(
        self, z_well: float, z_top: float, z_base: float
    ) -> float:
        rel_thickness = (z_well - z_top) / (z_base - z_top)
        return np.interp(rel_thickness, self.trend_zrel, self.trend_vals)

    def triplet_likelihood(
        self,
        z_well: float,
        z_top: float,
        z_base: float,
        observed_value: float,
        standard_deviation: float,
    ) -> float:
        predicted_value = self.triplet_trend_prediction(z_well, z_top, z_base)
        return stats.norm.pdf(
            observed_value, loc=predicted_value, scale=standard_deviation
        )


class Particle:
    def __init__(self, tb_sample: dict, xvals: list, weight: float):
        self.x = xvals
        self.z_top = tb_sample["top"]
        self.z_base = tb_sample["base"]
        self.weight = weight

    def tb(self):
        return {"top": self.z_top, "base": self.z_base}

    def extend(self, tb_sample: dict, xs, overlap: int = 1):
        self.x.extend(xs[overlap:])
        self.z_top.extend(tb_sample["top"][overlap:])
        self.z_base.extend(tb_sample["base"][overlap:])


class ParticleFilter:
    def __init__(
        self,
        n_particles,
        tbpp: TopBaseProcessPair,
        zf: ZonalField,
        noise_params: dict,
        well_data: dict,
        domain: dict,
    ):
        self.n_particles = n_particles
        self.tbpp = tbpp
        self.gr_trend = zf
        self.noise_params = noise_params

        self.x_obs = well_data["x"]
        self.z_obs = well_data["z"]
        self.gr_obs = well_data["gr"]

        self.domain = domain
        self.particles = self.init_particles()

    def init_particles(self):
        particles = []

        x0 = self.x_obs[0]
        weight = 1.0 / self.n_particles

        for i in range(self.n_particles):
            # TODO: Draw multiple values covering interval from left end to x0
            tb_sample = self.tbpp.draw_single(x0)
            particles.append(Particle(tb_sample, [x0], weight))

        return particles

    def forecast(self, obs_index: int):
        nx_new = 10
        nx_past = 10
        xs_new = np.linspace(
            self.x_obs[obs_index - 1], self.x_obs[obs_index], nx_new
        ).tolist()
        delta_x = xs_new[-1] - xs_new[0]
        x_past_start = self.x_obs[obs_index - 1] - delta_x
        x_past_end = self.x_obs[obs_index - 1]
        xs_past = np.linspace(x_past_start, x_past_end, nx_past).tolist()

        for p in self.particles:
            tb_past = self.tbpp.interpolate_sample(xs_past, p.tb())
            tb_new = self.tbpp.draw_conditional_points(xs_new, xs_past, tb_past)
            p.extend(tb_new, xs_new)

    def compute_weights(self, obs_index: int):
        weights = np.zeros(self.n_particles)
        for i, p in enumerate(self.particles):
            weights[i] = self.gr_trend.triplet_likelihood(
                z_well=self.z_obs[obs_index],
                z_top=p.z_top[-1],
                z_base=p.z_base[-1],
                observed_value=self.gr_obs[obs_index],
                standard_deviation=self.noise_params["sd_gr"],
            )
        weights = weights / np.sum(weights)
        for i, p in enumerate(self.particles):
            p.weight = weights[i]

    def resample(self):
        indices = np.random.choice(
            range(self.n_particles),
            size=self.n_particles,
            p=[p.weight for p in self.particles],
        )
        self.particles = [self.particles[i].copy() for i in indices]
