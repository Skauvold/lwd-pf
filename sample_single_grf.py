import numpy as np
import matplotlib.pyplot as plt

from utils import CovarianceFunction, GenExp, GaussianProcess

# Define the domain of the function
domain = {"xmin": 0, "xmax": 1, "n": 100}

# Define the covariance function
cov_func = GenExp(
    {"standard_deviation": 0.5, "correlation_range": 0.33, "exponent": 1.9}
)


# Define the mean function
def mean_func(x):
    a = 0.67
    b = -1.5
    return a + b * x


# Create a Gaussian process
gp = GaussianProcess(cov_func, mean_func, domain)

# Draw some samples from the Gaussian process
# samples = [gp.draw_full() for _ in range(10)]
ic = [42]
vc = [1.0]
samples = [gp.draw_conditional(ic, vc) for _ in range(10)]

# Plot the samples
plt.figure()

for sample in samples:
    plt.plot(gp.xvals, sample, "k-", alpha=0.25)

plt.plot(gp.xvals, gp.mean(), "r-", linewidth=2)

plt.xlabel("x")
plt.ylabel("z")

plt.ylim(gp.limits(n_std=4))
plt.gca().invert_yaxis()
plt.show()
