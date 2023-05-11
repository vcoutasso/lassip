import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    bscan = np.load("bscan3.npy")

    n, tc = bscan.shape

    height = 50
    width = tc
    origin = (25, -int(width / 2))
    resolution = (50, tc)

    z = np.linspace(origin[0], origin[0] + height, resolution[0])
    x = np.linspace(origin[1], origin[1] + width, resolution[1])

    dt = 5e-9
    dx = 100e-6

    c = 0.29

    f = 5e6

    xt = np.arange(tc) - int(width / 2)

    saft = np.zeros(resolution)
    xx, zz = np.meshgrid(x, z)

    gating = 0

    sample = None
    dist = None

    for elem in range(tc):
        dist = np.sqrt((xx - xt[elem]) ** 2 + zz ** 2)
        delay = 2 * dist / c
        sample = np.minimum(np.round(np.maximum(0, delay - gating)).astype(int), n-1)
        #ascan = np.pad(bscan[:, elem], (0, 1))
        saft += bscan[sample, elem]

    plt.imshow(saft, aspect="auto", extent=(0, tc, resolution[0] + origin[0], origin[0]))
    plt.show()
