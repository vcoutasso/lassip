import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    bscan = np.load("bscan.npy")

    n, tc = bscan.shape

    height = 50
    width = 50
    origin = (int(height / 2), -int(width / 2))
    resolution = (1000, tc)

    z = np.linspace(origin[0] + origin[0] + height, resolution[0])
    x = np.linspace(origin[1] + origin[1] + width, resolution[1])

    dt = 5e-9
    dx = 100e-6

    c = 0.35

    f = 5e6
    samples_per_period = 30

    T = samples_per_period / f

    xt = np.arange(tc) * dx

    saft = np.zeros((height, width))
    xx, zz = np.meshgrid(x, z)

    for elem in range(tc):
        dist = np.sqrt(np.power(xx - xt[elem], 2) + np.power(zz, 2))
        delay = dt * 2 * dist / c
        sample = np.round(delay / T).astype(int)
        print(sample)
        ascan = np.pad(bscan[:, elem], (0, 1))
        saft += np.take(ascan, sample, mode="clip")

    plt.imshow(saft)
    plt.show()
