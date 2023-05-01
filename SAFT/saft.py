import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    bscan = np.load("test.npy")

    n, tc = bscan.shape
    t = np.arange(n)
    c = 0.35
    roi = 250

    samples_per_period = 36
    f = 1e6

    T = f / samples_per_period
    print(T)

    xs = np.arange(tc)
    xs += (int(roi / 2)) - xs[int(tc / 2)]
    z = c * t / 2

    saft = np.zeros((roi, roi))
    indices = np.arange(roi)
    xx, zz = np.meshgrid(indices, z)

    for elem in range(tc):
        dist = np.sqrt(np.power(xx - xs[elem], 2) + np.power(zz, 2))
        delay = 2 * dist / c
        sample = np.round(delay / T).astype(int)
        ascan = np.pad(bscan[:, elem], (0, 1))
        saft += np.take(ascan, sample, mode="clip")

    saft /= tc

    plt.imshow(saft)
    plt.show()
