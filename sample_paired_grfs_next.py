import numpy as np
import matplotlib.pyplot as plt

from utils import CovarianceFunction, GenExp, GaussianProcess, TopBaseProcessPair

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


# Create a Gaussian processes
gp_base = GaussianProcess(cov_func_base, mean_func_base, domain)
gp_thickness = GaussianProcess(cov_func_thickness, mean_func_thickness, domain)
tbpp = TopBaseProcessPair(gp_base, gp_thickness)

# Draw a realization
ref = tbpp.draw_full()

# Sample single element given everything to the left
index = 16
ref_truncated = {k: v[:index] for k, v in ref.items()}
samples = [tbpp.draw_left_conditional(index, ref_truncated) for _ in range(100)]

# Plot the samples
plt.figure()

plt.plot(tbpp.xvals[:index], ref["top"][:index], "g-", alpha=0.5)
plt.plot(tbpp.xvals[:index], ref["base"][:index], "r-", alpha=0.5)

# Plot the samples
# x_index_list = [tbpp.xvals[index]] * len(samples)
# plt.scatter(x_index_list, [s["top"] for s in samples], c="g", marker="o")
# plt.scatter(x_index_list, [s["base"] for s in samples], c="r", marker="o")
for sample in samples:
    plt.plot(
        tbpp.xvals[index - 1 : index + 1],
        [ref["top"][index - 1], sample["top"]],
        "g-",
        alpha=0.5,
    )
    plt.plot(
        tbpp.xvals[index - 1 : index + 1],
        [ref["base"][index - 1], sample["base"]],
        "r-",
        alpha=0.5,
    )

plt.plot(tbpp.xvals, tbpp.mean_top(), "g-", linewidth=1)
plt.plot(tbpp.xvals, tbpp.mean_base(), "r-", linewidth=1)

plt.xlabel("x")
plt.ylabel("f(x)")

plt.ylim(tbpp.limits(n_std=4))

plt.gca().invert_yaxis()
plt.show()
