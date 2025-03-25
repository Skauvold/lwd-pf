import numpy as np
import matplotlib.pyplot as plt

from utils import GenExp, GaussianProcess, TopBaseProcessPair, ZonalField

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

samples = [tbpp.draw_conditional_points(x_eval, x_obs, z_obs) for _ in range(250)]

# Evaluate triple likelihood of each sample
x_w = x_now + delta_x_forward
tb_w = tbpp.interpolate_sample(x_w, ref)
z_w = (tb_w["top"] + tb_w["base"]) / 2
gr_obs = 0.75
sd_gr = 0.1

gr_field = ZonalField([0.0, 0.4, 0.6, 1.0], [0.9, 0.33, 0.67, 0.1])

likelihood_values = [
    gr_field.triplet_likelihood(
        z_w, sample["top"][-1], sample["base"][-1], gr_obs, sd_gr
    )
    for sample in samples
]

normalization_constant = max(likelihood_values)
weights_01 = [l / normalization_constant for l in likelihood_values]

# Define colormap
cmap = plt.get_cmap("jet")

plt.figure()

# Reference
plt.plot(tbpp.xvals, ref["top"], "g-", alpha=0.5)
plt.plot(tbpp.xvals, ref["base"], "r-", alpha=0.5)

# Observations
plt.plot(x_obs, z_obs["top"], "gs")
plt.plot(x_obs, z_obs["base"], "rs")
plt.plot([x_w], [z_w], "bs")

# Cond. samples
for sample, weight_01 in zip(samples, weights_01):
    plt.plot(x_eval, sample["top"], "-", color=cmap(weight_01), alpha=0.5)
    plt.plot(x_eval, sample["base"], "-", color=cmap(weight_01), alpha=0.5)

plt.plot(tbpp.xvals, tbpp.mean_top(), "g-", linewidth=1)
plt.plot(tbpp.xvals, tbpp.mean_base(), "r-", linewidth=1)

plt.xlabel("x")
plt.ylabel("z")

plt.ylim(tbpp.limits(n_std=4))

plt.gca().invert_yaxis()
plt.show()
