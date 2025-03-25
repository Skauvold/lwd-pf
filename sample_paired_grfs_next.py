import numpy as np
import matplotlib.pyplot as plt

from utils import GenExp, GaussianProcess, TopBaseProcessPair

domain = {"xmin": 0, "xmax": 1, "n": 40}

# Define the covariance function
cov_func_base = GenExp(
    {"standard_deviation": 0.15, "correlation_range": 0.30, "exponent": 1.9}
)

cov_func_thickness = GenExp(
    {"standard_deviation": 0.10, "correlation_range": 0.40, "exponent": 1.8}
)


# Define the mean functions
def mean_func_base(x):
    a = 0.67
    b = 0.5
    return a + b * x


def mean_func_thickness(x):
    return np.full_like(x, 0.5)


# Create process pair
gp_base = GaussianProcess(cov_func_base, mean_func_base, domain)
gp_thickness = GaussianProcess(cov_func_thickness, mean_func_thickness, domain)
tbpp = TopBaseProcessPair(gp_base, gp_thickness)

# Draw a realization
ref = tbpp.draw_full()

# Draw some conditional samples
x_now = 0.5
delta_x_forward = 0.1
delta_x_backward = 0.05
x_obs = np.linspace(x_now - delta_x_backward, x_now, 5).tolist()
z_obs = tbpp.interpolate_sample(x_obs, ref)

x_eval = np.linspace(x_now, x_now + delta_x_forward, 5).tolist()

samples = [tbpp.draw_conditional_points(x_eval, x_obs, z_obs) for _ in range(40)]


plt.figure()

# Reference
plt.plot(tbpp.xvals, ref["top"], "g-", alpha=0.5)
plt.plot(tbpp.xvals, ref["base"], "r-", alpha=0.5)

# Observations
plt.plot(x_obs, z_obs["top"], "gs")
plt.plot(x_obs, z_obs["base"], "rs")

# Cond. samples
for sample in samples:
    plt.plot(x_eval, sample["top"], "g-", alpha=0.25)
    plt.plot(x_eval, sample["base"], "r-", alpha=0.25)

plt.plot(tbpp.xvals, tbpp.mean_top(), "g-", linewidth=1)
plt.plot(tbpp.xvals, tbpp.mean_base(), "r-", linewidth=1)

plt.xlabel("x")
plt.ylabel("z")

plt.ylim(tbpp.limits(n_std=4))

plt.gca().invert_yaxis()
plt.show()
