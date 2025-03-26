import numpy as np
import matplotlib.pyplot as plt

from utils import ZonalField

trend_values = [
    (0.0, 0.97),
    (0.05, 0.95),
    (0.45, 0.27),
    (0.5, 0.25),
    (0.55, 0.6),
    (0.68, 0.55),
    (0.7, 0.3),
    (0.75, 0.2),
    (1.0, 0.04),
]

zf = ZonalField(
    trend_zrel=[z for (z, _) in trend_values],
    trend_vals=[v for (_, v) in trend_values],
    )

# Visualize depth trend and likelihood for a given observed trend value
zvals = np.linspace(0, 1, 250)
tvals = zf.eval(zvals)

t_obs = 0.8
sigma_t = 0.2
likelihood_values = [zf.triplet_likelihood(z, 0.0, 1.0, t_obs, sigma_t) for z in zvals]

plt.figure()
plt.subplot(121)
plt.plot(zf.trend_vals, zf.trend_zrel, "ks")
plt.plot(tvals, zvals, "r-", lw=2)
plt.plot([t_obs, t_obs], [0, 1], "k--")

plt.xlabel("trend")
plt.ylabel("z")

plt.gca().invert_yaxis()

plt.subplot(122)
plt.plot(likelihood_values, zvals, "k-")
plt.xlabel("likelihood")
plt.ylabel("z")

plt.gca().invert_yaxis()

plt.show()
